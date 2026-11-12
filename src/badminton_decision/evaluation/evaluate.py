from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from badminton_decision.data.dataset import ChoiceVideoDataset, DecisionCollator
from badminton_decision.data.feature_cache import VisualFeatureCache
from badminton_decision.data.feature_extraction import visual_cache_metadata
from badminton_decision.utils import read_jsonl
from badminton_decision.evaluation.metrics import classification_metrics
from badminton_decision.models.smolvlm_decision import load_decision_model


def evaluate_checkpoint(checkpoint_dir: str | Path, processed_dir: str | Path, video_root: str | Path,
                        split: str = "test", permutation_trials: int = 5, max_samples: int | None = None,
                        device: str | None = None, output_path: str | Path | None = None):
    checkpoint_dir, processed_dir, video_root = Path(checkpoint_dir), Path(processed_dir), Path(video_root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_decision_model(checkpoint_dir, device=device)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((checkpoint_dir / "training_config.yaml").read_text(encoding="utf-8"))
    vcfg, ccfg = metadata["video_config"], metadata["choice_config"]
    feature_cache = None
    cache_cfg = cfg.get("visual_cache", {})
    cache_dir = Path(cache_cfg.get("directory", "data/cache/visual_features"))
    if bool(cache_cfg.get("enabled", False)) and (cache_dir / "features.npy").exists():
        feature_cache = VisualFeatureCache(cache_dir, visual_cache_metadata(cfg))
        split_rows = read_jsonl(processed_dir / f"{split}.jsonl")
        feature_cache.assert_complete(sorted({str(row["event_id"]) for row in split_rows}))
    elif bool(cache_cfg.get("enabled", False)):
        print(f"Visual cache not found at {cache_dir}; evaluation will decode videos online.")
    dataset = ChoiceVideoDataset(
        processed_dir / f"{split}.jsonl", video_root, split,
        num_frames=vcfg["num_frames"], duration_seconds=vcfg["duration_seconds"],
        cache_size=vcfg.get("cache_size", 8), jitter_seconds=0.0,
        min_options=2, max_options=ccfg.get("max_options_eval", 16), shuffle_options=False,
        max_samples=max_samples, seed=42, feature_cache=feature_cache,
    )
    collator = DecisionCollator(
        model.processor, ccfg["context_max_tokens"], ccfg["option_max_tokens"],
        ccfg["task_prompt"], image_longest_edge=vcfg.get("image_longest_edge", 512),
        use_visual_cache=feature_cache is not None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator)
    y_true, y_pred, confidences = [], [], []
    order_diffs, order_same = [], []
    predictions = []
    rng = random.Random(42)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate:{split}"):
            model_inputs = {
                "context_ids": batch["context_ids"],
                "option_ids": batch["option_ids"],
                "task_type_ids": batch["task_type_names"],
            }
            if batch.get("visual_features") is not None:
                logits = model(
                    visual_features=batch["visual_features"].to(device), **model_inputs
                )[0]
            else:
                batch["pixel_values"] = batch["pixel_values"].to(device)
                batch["pixel_attention_mask"] = batch["pixel_attention_mask"].to(device)
                logits = model(
                    pixel_values=batch["pixel_values"],
                    pixel_attention_mask=batch["pixel_attention_mask"],
                    **model_inputs,
                )[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            options = batch["options_text"][0]
            pred_idx = int(torch.argmax(logits).item())
            answer = batch["answers"][0]
            y_true.append(answer)
            y_pred.append(options[pred_idx])
            confidences.append(float(probs[pred_idx]))
            perm_consistency = []
            perm_differences = []
            if permutation_trials > 1 and len(options) > 1:
                base_map = {opt: probs[i] for i, opt in enumerate(options)}
                for _ in range(permutation_trials - 1):
                    perm = list(range(len(options)))
                    rng.shuffle(perm)
                    perm_options = [options[i] for i in perm]
                    perm_option_ids = [batch["option_ids"][0][i] for i in perm]
                    if batch.get("visual_features") is not None:
                        perm_logits = model(
                            visual_features=batch["visual_features"].to(device),
                            context_ids=batch["context_ids"], option_ids=[perm_option_ids],
                            task_type_ids=batch["task_type_names"],
                        )[0]
                    else:
                        perm_logits = model(
                            pixel_values=batch["pixel_values"],
                            pixel_attention_mask=batch["pixel_attention_mask"],
                            context_ids=batch["context_ids"], option_ids=[perm_option_ids],
                            task_type_ids=batch["task_type_names"],
                        )[0]
                    perm_probs = torch.softmax(perm_logits, dim=-1).cpu().tolist()
                    perm_map = {opt: perm_probs[i] for i, opt in enumerate(perm_options)}
                    perm_pred = perm_options[int(torch.argmax(perm_logits).item())]
                    perm_consistency.append(perm_pred == options[pred_idx])
                    perm_differences.extend(abs(base_map[k] - perm_map[k]) for k in base_map)
            if perm_consistency:
                order_same.extend(perm_consistency)
                order_diffs.extend(perm_differences)
            predictions.append({
                "sample_id": batch["sample_ids"][0], "match_id": batch["match_ids"][0],
                "task_type": batch["task_type_names"][0], "question": batch["questions"][0],
                "options": options, "answer": answer, "prediction": options[pred_idx],
                "confidence": float(probs[pred_idx]), "probabilities": probs,
            })
    metrics = classification_metrics(y_true, y_pred, confidences)
    by_task = {}
    task_names = sorted({row["task_type"] for row in predictions})
    for task in task_names:
        idxs = [i for i, row in enumerate(predictions) if row["task_type"] == task]
        by_task[task] = classification_metrics([y_true[i] for i in idxs], [y_pred[i] for i in idxs], [confidences[i] for i in idxs])
    metrics["by_task"] = by_task
    metrics["option_order"] = {
        "prediction_consistency": sum(order_same) / len(order_same) if order_same else None,
        "mean_abs_probability_change": sum(order_diffs) / len(order_diffs) if order_diffs else None,
        "permutation_comparisons": len(order_same),
    }
    result = {"split": split, "checkpoint": str(checkpoint_dir), "metrics": metrics, "predictions": predictions}
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--video-root", default="data/raw/finebadminton-20k/videos")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--permutation-trials", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.processed_dir, args.video_root, args.split,
                                 args.permutation_trials, args.max_samples, args.device, args.output)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
