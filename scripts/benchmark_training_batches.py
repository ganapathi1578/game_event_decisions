from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from badminton_decision.data.dataset import ChoiceVideoDataset, DecisionCollator
from badminton_decision.data.feature_cache import VisualFeatureCache
from badminton_decision.data.feature_extraction import visual_cache_metadata
from badminton_decision.models.smolvlm_decision import SmolVLMChoiceModel
from badminton_decision.utils import read_jsonl


def _parse_batch_sizes(value: str) -> list[int]:
    sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated positive integers")
    return sizes


def _make_batch(dataset, collator, batch_size: int, offset: int):
    indices = [(offset + index) % len(dataset) for index in range(batch_size)]
    return collator([dataset[index] for index in indices])


def _run_step(model, batch, device: torch.device, use_amp: bool) -> float:
    features = batch["visual_features"].to(device=device, non_blocking=True)
    labels = batch["correct_index"].to(device=device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(use_amp and device.type == "cuda"),
    ):
        logits = model(
            visual_features=features,
            context_ids=batch["context_ids"],
            option_ids=batch["option_ids"],
            task_type_ids=batch["task_type_names"],
        )
        loss = F.cross_entropy(logits, labels)
    loss.backward()
    return float(loss.detach().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark real cached-video decoder training batches on the current GPU."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=_parse_batch_sizes("1,2,4,8"))
    parser.add_argument("--steps", type=int, default=3, help="Measured forward/backward steps per batch size.")
    parser.add_argument("--output", default="runs/batch_benchmark.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for a useful VRAM/throughput benchmark.")
    device = torch.device("cuda")

    cache_cfg = cfg.get("visual_cache", {})
    if not bool(cache_cfg.get("enabled", False)):
        raise SystemExit("Set visual_cache.enabled: true to benchmark using cached visual features.")
    cache = VisualFeatureCache(
        cache_cfg.get("directory", "data/cache/visual_features"),
        visual_cache_metadata(cfg),
    )
    processed_dir = Path(cfg["project"]["processed_dir"])
    train_rows = read_jsonl(processed_dir / "train.jsonl")
    cache.assert_complete(sorted({str(row["event_id"]) for row in train_rows}))

    model_cfg = cfg["model"]
    model = SmolVLMChoiceModel(
        model_id=model_cfg["name"],
        dtype=model_cfg.get("torch_dtype", "float16"),
        attn_implementation=model_cfg.get("attn_implementation", "eager"),
        lora_cfg=cfg["lora"],
        head_cfg=cfg["head"],
        gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True),
        local_files_only=model_cfg.get("local_files_only", False),
    ).to(device)
    model.train()

    video_cfg, choice_cfg = cfg["video"], cfg["choice"]
    dataset = ChoiceVideoDataset(
        processed_dir / "train.jsonl",
        Path(cfg["project"]["data_dir"]) / "videos",
        "train",
        num_frames=video_cfg["num_frames"],
        duration_seconds=video_cfg["duration_seconds"],
        cache_size=video_cfg.get("cache_size", 8),
        jitter_seconds=0.0,
        min_options=choice_cfg["min_options_train"],
        max_options=choice_cfg["max_options_train"],
        shuffle_options=choice_cfg["shuffle_train_options"],
        max_samples=cfg["training"].get("max_train_samples"),
        seed=int(cfg["project"]["seed"]),
        feature_cache=cache,
    )
    collator = DecisionCollator(
        model.processor,
        context_max_tokens=choice_cfg["context_max_tokens"],
        option_max_tokens=choice_cfg["option_max_tokens"],
        task_prompt=choice_cfg["task_prompt"],
        image_longest_edge=video_cfg["image_longest_edge"],
        use_visual_cache=True,
    )

    results = []
    for batch_size in args.batch_sizes:
        if batch_size > len(dataset):
            continue
        torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            # Warm-up catches shape/kernel initialization costs before timing.
            warmup = _make_batch(dataset, collator, batch_size, 0)
            _run_step(model, warmup, device, use_amp=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

            durations = []
            losses = []
            for step in range(max(1, args.steps)):
                batch = _make_batch(dataset, collator, batch_size, (step + 1) * batch_size)
                start = time.perf_counter()
                losses.append(_run_step(model, batch, device, use_amp=True))
                torch.cuda.synchronize(device)
                durations.append(time.perf_counter() - start)
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            mean_seconds = statistics.mean(durations)
            result = {
                "batch_size": batch_size,
                "status": "ok",
                "peak_memory_gib": peak_gib,
                "mean_seconds_per_batch": mean_seconds,
                "examples_per_second": batch_size / max(mean_seconds, 1e-9),
                "mean_loss": statistics.mean(losses),
            }
            print(
                f"batch={batch_size}: peak={peak_gib:.2f} GiB, "
                f"{result['examples_per_second']:.2f} examples/s"
            )
            results.append(result)
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            result = {"batch_size": batch_size, "status": "cuda_oom"}
            print(f"batch={batch_size}: CUDA OOM")
            results.append(result)
        finally:
            model.zero_grad(set_to_none=True)

    successful = [row for row in results if row["status"] == "ok"]
    recommended = max(successful, key=lambda row: row["examples_per_second"])["batch_size"] if successful else 1
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "total_vram_gib": torch.cuda.get_device_properties(device).total_memory / (1024**3),
        "recommended_batch_size_by_throughput": recommended,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Recommended batch size: {recommended}")
    print(f"Report: {output_path}")
    print("Set training.per_device_batch_size to the selected integer in your YAML config.")


if __name__ == "__main__":
    main()
