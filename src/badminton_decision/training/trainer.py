from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from badminton_decision.constants import DEFAULT_MODEL_ID
from badminton_decision.data.dataset import ChoiceVideoDataset, DecisionCollator
from badminton_decision.models.smolvlm_decision import SmolVLMChoiceModel
from badminton_decision.utils import seed_everything


def _load_yaml(path: str | Path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _make_loader(manifest, video_root, split, processor, cfg, max_samples=None):
    video_cfg, choice_cfg = cfg["video"], cfg["choice"]
    dataset = ChoiceVideoDataset(
        manifest_path=manifest,
        video_root=video_root,
        split=split,
        num_frames=video_cfg["num_frames"],
        duration_seconds=video_cfg["duration_seconds"],
        cache_size=video_cfg["cache_size"],
        jitter_seconds=video_cfg["temporal_jitter_seconds"],
        min_options=choice_cfg["min_options_train"] if split == "train" else 2,
        max_options=choice_cfg["max_options_train"] if split == "train" else choice_cfg["max_options_eval"],
        shuffle_options=choice_cfg["shuffle_train_options"],
        max_samples=max_samples,
        seed=cfg["project"]["seed"],
    )
    collator = DecisionCollator(
        processor,
        context_max_tokens=choice_cfg["context_max_tokens"],
        option_max_tokens=choice_cfg["option_max_tokens"],
        task_prompt=choice_cfg["task_prompt"],
        image_longest_edge=video_cfg["image_longest_edge"],
    )
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["per_device_batch_size"],
        shuffle=split == "train",
        num_workers=cfg["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=False,
    )


def _build_accelerator(cfg):
    try:
        from accelerate import Accelerator
    except ImportError as exc:
        raise RuntimeError("accelerate is required: pip install accelerate") from exc
    precision = cfg["training"].get("mixed_precision", "fp16")
    return Accelerator(
        gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
        mixed_precision=precision,
    )


def _eval_loss_accuracy(model, loader, accelerator, max_batches=None):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            labels = batch["correct_index"].to(accelerator.device)
            with accelerator.autocast():
                logits = model(
                    pixel_values=batch["pixel_values"],
                    pixel_attention_mask=batch["pixel_attention_mask"],
                    context_ids=batch["context_ids"], option_ids=batch["option_ids"],
                    task_type_ids=batch["task_type_names"],
                )
                loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)
            gathered_labels = accelerator.gather(labels.detach())
            gathered_correct = accelerator.gather((preds == labels).long())
            gathered_loss = accelerator.gather(loss.detach().view(1))
            total_loss += float(gathered_loss.mean().item()) * int(gathered_labels.numel())
            total_correct += int(gathered_correct.sum().item())
            total += int(gathered_labels.numel())
    model.train()
    denom = max(total, 1)
    return {"loss": total_loss / denom, "accuracy": total_correct / denom, "count": total}


def train(config_path: str | Path, run_name: str = "badminton-decision"):
    cfg = _load_yaml(config_path)
    seed_everything(int(cfg["project"]["seed"]))
    accelerator = _build_accelerator(cfg)
    run_dir = Path(cfg["project"]["output_dir"]) / run_name
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    model_cfg = cfg["model"]
    dtype = model_cfg.get("torch_dtype", "float16")
    model = SmolVLMChoiceModel(
        model_id=model_cfg.get("name", DEFAULT_MODEL_ID),
        dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "eager"),
        lora_cfg=cfg["lora"],
        head_cfg=cfg["head"],
        gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True),
        local_files_only=model_cfg.get("local_files_only", False),
    )
    if accelerator.is_main_process:
        print("Trainable parameters:", model.trainable_parameter_counts())
        print("Decoder hidden size:", model.hidden_dim)
        print("Vision encoder output tokens per frame are determined by the loaded checkpoint.")

    data_dir = Path(cfg["project"]["processed_dir"])
    raw_dir = Path(cfg["project"]["data_dir"])
    train_loader = _make_loader(data_dir / "train.jsonl", raw_dir / "videos", "train", model.processor, cfg,
                                cfg["training"].get("max_train_samples"))
    val_loader = _make_loader(data_dir / "validation.jsonl", raw_dir / "videos", "validation", model.processor, cfg,
                              cfg["training"].get("max_eval_samples"))

    lora_params = [p for p in model.vlm.model.text_model.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": float(cfg["training"]["learning_rate_lora"])},
         {"params": head_params, "lr": float(cfg["training"]["learning_rate_head"])}],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    steps_per_epoch = math.ceil(len(train_loader) / int(cfg["training"]["gradient_accumulation_steps"]))
    total_steps = max(1, steps_per_epoch * int(cfg["training"]["epochs"]))
    warmup_steps = int(float(cfg["training"]["warmup_ratio"]) * total_steps)
    try:
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("Transformers scheduler import failed. Update transformers.") from exc
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    global_step = 0
    best_accuracy = -1.0
    log_path = run_dir / "metrics.jsonl"
    if accelerator.is_main_process:
        print(f"Train examples: {len(train_loader.dataset):,}; validation examples: {len(val_loader.dataset):,}")
        print(f"Optimizer steps/epoch: {steps_per_epoch:,}; planned total: {total_steps:,}")

    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        running_loss, seen = 0.0, 0
        progress = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"epoch {epoch+1}")
        optimizer.zero_grad(set_to_none=True)
        for batch in progress:
            with accelerator.accumulate(model):
                labels = batch["correct_index"].to(accelerator.device)
                with accelerator.autocast():
                    logits = model(
                        pixel_values=batch["pixel_values"],
                        pixel_attention_mask=batch["pixel_attention_mask"],
                        context_ids=batch["context_ids"], option_ids=batch["option_ids"],
                        task_type_ids=batch["task_type_names"],
                    )
                    loss = F.cross_entropy(logits, labels)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(cfg["training"]["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().item())
            seen += 1
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % int(cfg["training"]["logging_steps"]) == 0 and accelerator.is_main_process:
                    record = {"step": global_step, "epoch": epoch + 1,
                              "train_loss": running_loss / max(seen, 1),
                              "lr_lora": optimizer.param_groups[0]["lr"],
                              "lr_head": optimizer.param_groups[1]["lr"]}
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    progress.set_postfix(loss=f"{running_loss/max(seen,1):.4f}")
                if global_step % int(cfg["training"]["eval_steps"]) == 0:
                    val_metrics = _eval_loss_accuracy(model, val_loader, accelerator,
                                                      cfg["training"].get("max_eval_samples"))
                    if accelerator.is_main_process:
                        print(f"step={global_step} validation={val_metrics}")
                        record = {"step": global_step, "epoch": epoch + 1, "validation": val_metrics}
                        with log_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record) + "\n")
                        if val_metrics["accuracy"] > best_accuracy:
                            best_accuracy = val_metrics["accuracy"]
                            _save_model(accelerator.unwrap_model(model), run_dir / "checkpoints" / "best", cfg, global_step, val_metrics)
                if global_step % int(cfg["training"]["save_steps"]) == 0 and accelerator.is_main_process:
                    _save_model(accelerator.unwrap_model(model), run_dir / "checkpoints" / "latest", cfg, global_step, {})
        accelerator.wait_for_everyone()
        val_metrics = _eval_loss_accuracy(model, val_loader, accelerator, cfg["training"].get("max_eval_samples"))
        if accelerator.is_main_process:
            print(f"epoch={epoch+1} validation={val_metrics}")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": global_step, "epoch": epoch + 1, "validation_epoch": val_metrics}) + "\n")
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                _save_model(accelerator.unwrap_model(model), run_dir / "checkpoints" / "best", cfg, global_step, val_metrics)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_model(accelerator.unwrap_model(model), run_dir / "checkpoints" / "final", cfg, global_step, {"best_validation_accuracy": best_accuracy})
        print(f"Training complete. Checkpoints: {run_dir / 'checkpoints'}")
    accelerator.end_training()
    return run_dir


def _save_model(model, checkpoint_dir: Path, cfg: dict, step: int, metrics: dict):
    model_config = {
        "name": cfg["model"]["name"],
        "dtype": cfg["model"].get("torch_dtype", "float16"),
        "lora": cfg["lora"],
        "head": cfg["head"],
    }
    metadata = {
        "global_step": step,
        "model_config": model_config,
        "choice_config": cfg["choice"],
        "video_config": cfg["video"],
        "metrics": metrics,
    }
    model.save_checkpoint(checkpoint_dir, metadata)
    (checkpoint_dir / "training_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
