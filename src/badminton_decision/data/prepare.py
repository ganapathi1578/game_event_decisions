from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from badminton_decision.data.annotations import parse_dataset
from badminton_decision.utils import write_json, write_jsonl


def split_by_match(events: list[dict[str, Any]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/validation/test ratios must sum to 1.0")
    matches = sorted({event["match_id"] for event in events})
    random.Random(seed).shuffle(matches)
    n = len(matches)
    if n < 3:
        raise ValueError(f"At least three distinct matches are required for group splitting; found {n}.")
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Assign the remainder to test so all groups are accounted for.
    n_test = n - n_train - n_val
    if n >= 3:
        n_train = max(1, min(n - 2, n_train))
        n_val = max(1, min(n - n_train - 1, n_val))
        n_test = n - n_train - n_val
    return {
        "train": set(matches[:n_train]),
        "validation": set(matches[n_train:n_train + n_val]),
        "test": set(matches[n_train + n_val:]),
    }


def make_qa_records(events, split_by_match_map, templates, include_tasks, min_class_count=2):
    train_events = [e for e in events if e["match_id"] in split_by_match_map["train"]]
    task_fields = {"hit_type": "hit_type", "subtype": "subtype", "hitter": "hitter", "ball_area": "ball_area"}
    labels_by_task = {}
    for task, field in task_fields.items():
        counter = Counter()
        for event in train_events:
            value = event.get(field)
            if task == "subtype":
                if isinstance(value, list) and len(value) == 1:
                    counter[value[0]] += 1
            elif value:
                counter[str(value)] += 1
        labels_by_task[task] = sorted(label for label, count in counter.items() if count >= min_class_count)

    output = {name: [] for name in ("train", "validation", "test")}
    skipped = Counter()
    for event in events:
        split = next((name for name, ids in split_by_match_map.items() if event["match_id"] in ids), None)
        if split is None:
            continue
        for task in include_tasks:
            if task not in task_fields:
                skipped[f"unsupported_task:{task}"] += 1
                continue
            raw = event.get(task_fields[task])
            if task == "subtype":
                if not isinstance(raw, list) or len(raw) != 1:
                    skipped["subtype_missing_or_ambiguous"] += 1
                    continue
                answer = raw[0]
            else:
                answer = raw
            if not answer:
                skipped[f"missing_{task}"] += 1
                continue
            answer = str(answer)
            classes = labels_by_task.get(task, [])
            if len(classes) < 2 or answer not in classes:
                skipped[f"insufficient_or_unseen_class_{task}"] += 1
                continue
            train_templates = templates.get(task, {})
            template_set = train_templates.get(split, [])
            if not template_set:
                skipped[f"no_templates_{task}_{split}"] += 1
                continue
            output[split].append({
                "sample_id": f"{event['event_id']}__{task}",
                "event_id": event["event_id"],
                "match_id": event["match_id"],
                "video_name": event["video_name"],
                "video_path": event["video_path"],
                "fps": event["fps"],
                "target_frame": event["target_frame"],
                "clip_duration_seconds": 3.0,
                "task_type": task,
                "question_templates": list(template_set),
                "options": classes,
                "answer": answer,
                "event_labels": {
                    "hit_type": event.get("hit_type"),
                    "subtype": event.get("subtype", []),
                    "hitter": event.get("hitter"),
                    "ball_area": event.get("ball_area"),
                },
            })
    return output, {"label_vocabularies": labels_by_task, "skipped": dict(skipped)}


def prepare(data_dir: str | Path, output_dir: str | Path, config_path: str | Path, templates_path: str | Path, seed: int = 42):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    templates = yaml.safe_load(Path(templates_path).read_text(encoding="utf-8"))
    events, source_summary = parse_dataset(data_dir)
    split_map = split_by_match(
        events,
        cfg["data"]["train_ratio"],
        cfg["data"]["validation_ratio"],
        cfg["data"]["test_ratio"],
        seed,
    )
    qa, qa_summary = make_qa_records(
        events, split_map, templates,
        include_tasks=cfg["data"]["include_tasks"],
        min_class_count=cfg["data"]["min_class_count"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "events.jsonl", events)
    for split_name, rows in qa.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)
    split_report = {name: sorted(ids) for name, ids in split_map.items()}
    summary = {
        "source": source_summary,
        "match_splits": {k: len(v) for k, v in split_map.items()},
        "qa_counts": {k: len(v) for k, v in qa.items()},
        "qa": qa_summary,
        "split_match_ids": split_report,
        "seed": seed,
        "note": "Splits are assigned by full match_id to prevent same-match leakage.",
    }
    write_json(output_dir / "prepare_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Parse annotations, split by match, and make choice QA rows.")
    parser.add_argument("--data-dir", default="data/raw/finebadminton-20k")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--templates", default="configs/question_templates.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare(args.data_dir, args.output_dir, args.config, args.templates, args.seed)
    print(f"Prepared data: {args.output_dir}")
    print(f"QA counts: {summary['qa_counts']}")
    print(f"Labels: {summary['qa']['label_vocabularies']}")


if __name__ == "__main__":
    main()
