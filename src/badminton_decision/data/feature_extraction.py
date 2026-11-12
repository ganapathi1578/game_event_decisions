from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from badminton_decision.data.dataset import DecisionCollator
from badminton_decision.data.feature_cache import VisualFeatureCache
from badminton_decision.data.video import VideoFrameReader
from badminton_decision.utils import read_jsonl


def visual_cache_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg["model"]
    video_cfg = cfg["video"]
    cache_cfg = cfg.get("visual_cache", {})
    return {
        "model_id": model_cfg.get("name"),
        "num_frames": int(video_cfg["num_frames"]),
        "duration_seconds": float(video_cfg["duration_seconds"]),
        "image_longest_edge": int(video_cfg["image_longest_edge"]),
        "feature_dtype": str(cache_cfg.get("feature_dtype", "float16")),
        "extractor": "SmolVLM2 vision_model + connector",
    }


def collect_unique_events(processed_dir: str | Path) -> list[dict[str, Any]]:
    processed_dir = Path(processed_dir)
    unique: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        manifest = processed_dir / f"{split}.jsonl"
        if not manifest.exists():
            continue
        for row in read_jsonl(manifest):
            event_id = row.get("event_id")
            if not event_id:
                raise ValueError(
                    f"Manifest row {row.get('sample_id', '<unknown>')} has no event_id. "
                    "Regenerate manifests with scripts/prepare_data.py before feature extraction."
                )
            unique.setdefault(str(event_id), row)
    if not unique:
        raise FileNotFoundError(
            f"No event records found in {processed_dir}/{{train,validation,test}}.jsonl. "
            "Run scripts/prepare_data.py first."
        )
    return [unique[key] for key in sorted(unique)]


def choose_extraction_batch_size(device: torch.device, cache_cfg: dict[str, Any]) -> int:
    requested = cache_cfg.get("extraction_batch_size", "auto")
    maximum = max(1, int(cache_cfg.get("max_extraction_batch_size", 8)))
    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        return max(1, min(maximum, int(requested)))
    if str(requested).lower() != "auto":
        raise ValueError("visual_cache.extraction_batch_size must be a positive integer or 'auto'")
    if device.type != "cuda" or not torch.cuda.is_available():
        print("Visual extraction auto-batch: CUDA unavailable; using micro-batch 1.")
        return 1

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / (1024**3)
    fraction = float(cache_cfg.get("memory_fraction", 0.50))
    reserve_gib = float(cache_cfg.get("reserve_gib", 2.5))
    estimated_gib_per_event = max(0.25, float(cache_cfg.get("estimated_gib_per_event", 1.5)))
    usable_gib = max(0.0, free_gib * fraction - reserve_gib)
    estimate = max(1, math.floor(usable_gib / estimated_gib_per_event))
    selected = min(maximum, estimate)
    print(
        "Visual extraction GPU memory: "
        f"{free_gib:.1f}/{total_bytes / (1024**3):.1f} GiB free; "
        f"starting micro-batch={selected} (cap={maximum})."
    )
    return selected


def _is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: memory allocation" in message


def extract_visual_features(
    cfg: dict[str, Any],
    model,
    *,
    device: str | torch.device | None = None,
    force: bool = False,
) -> VisualFeatureCache:
    """Extract deterministic visual features once per unique event and save to disk.

    Extraction is a standalone/offline stage. The returned cache is disk-backed and
    can be reopened by the later training process; the full feature set is never kept
    in GPU memory or loaded into RAM at once. Completed events are skipped on resume.
    """
    project_cfg = cfg["project"]
    video_cfg = cfg["video"]
    cache_cfg = cfg.get("visual_cache", {})
    cache_dir = Path(cache_cfg.get("directory", "data/cache/visual_features"))
    if force and cache_dir.exists():
        shutil.rmtree(cache_dir)

    events = collect_unique_events(project_cfg["processed_dir"])
    event_ids = [str(row["event_id"]) for row in events]
    expected_metadata = visual_cache_metadata(cfg)
    cache = VisualFeatureCache(
        cache_dir, expected_metadata, event_ids=event_ids, create=True
    )
    missing = cache.missing(event_ids)
    if not missing:
        print(f"Visual cache already complete: {len(cache):,} events at {cache_dir}")
        return cache

    by_id = {str(row["event_id"]): row for row in events}
    pending_rows = [by_id[event_id] for event_id in missing]
    video_root = Path(project_cfg["data_dir"]) / "videos"
    reader = VideoFrameReader(
        num_frames=int(video_cfg["num_frames"]),
        duration_seconds=float(video_cfg["duration_seconds"]),
        cache_size=int(video_cfg.get("cache_size", 8)),
        jitter_seconds=0.0,
    )
    image_collator = DecisionCollator(
        model.processor,
        context_max_tokens=int(cfg["choice"]["context_max_tokens"]),
        option_max_tokens=int(cfg["choice"]["option_max_tokens"]),
        task_prompt=str(cfg["choice"]["task_prompt"]),
        image_longest_edge=int(video_cfg["image_longest_edge"]),
    )

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = choose_extraction_batch_size(target_device, cache_cfg)
    vision_model = model.vlm.model.vision_model
    connector = model.vlm.model.connector
    original_vision_device = next(vision_model.parameters()).device
    original_connector_device = next(connector.parameters()).device
    was_training = model.training

    print(
        f"Extracting {len(pending_rows):,} missing event features with micro-batch={batch_size}.\n"
        f"Disk cache: {cache_dir} (one FP16 memmap file; resumable)"
    )
    progress = tqdm(total=len(pending_rows), desc="offline visual extraction", unit="event")
    try:
        vision_model.to(target_device)
        connector.to(target_device)
        model.eval()
        cursor = 0
        while cursor < len(pending_rows):
            candidate_rows = pending_rows[cursor : cursor + batch_size]
            processed = []
            for row in candidate_rows:
                video_path = Path(str(row["video_path"]))
                if not video_path.is_absolute():
                    video_path = video_root / video_path
                frames = reader.read(
                    video_path,
                    float(row["target_frame"]),
                    float(row.get("fps", 30.0)),
                    training=False,
                )
                processed.append(image_collator._process_frames(frames))

            pixel_shapes = {tuple(pixel_values.shape) for pixel_values, _ in processed}
            mask_shapes = {tuple(pixel_mask.shape) for _, pixel_mask in processed}
            if len(pixel_shapes) != 1 or len(mask_shapes) != 1:
                raise RuntimeError(
                    "The image processor returned variable image tensor shapes across this batch. "
                    f"Pixel shapes={pixel_shapes}; mask shapes={mask_shapes}. "
                    "Use one fixed aspect ratio or normalize input frame dimensions before extraction."
                )
            pixel_values = torch.stack([item[0] for item in processed], dim=0)
            pixel_attention_mask = torch.stack([item[1] for item in processed], dim=0)

            try:
                with torch.inference_mode():
                    features = model._encode_visual(pixel_values, pixel_attention_mask)
                shapes = {tuple(item.shape) for item in features}
                if len(features) != len(candidate_rows) or len(shapes) != 1:
                    raise RuntimeError(
                        "Visual encoder returned inconsistent feature groups: "
                        f"{len(features)} examples, shapes={shapes}."
                    )
                cache.write_batch(
                    [str(row["event_id"]) for row in candidate_rows], features
                )
            except RuntimeError as exc:
                if target_device.type == "cuda" and _is_cuda_oom(exc) and len(candidate_rows) > 1:
                    batch_size = max(1, (len(candidate_rows) + 1) // 2)
                    print(f"CUDA OOM during extraction; retrying with micro-batch={batch_size}.")
                    del pixel_values, pixel_attention_mask
                    del processed
                    torch.cuda.empty_cache()
                    continue
                if target_device.type == "cuda" and _is_cuda_oom(exc):
                    raise RuntimeError(
                        "Visual extraction ran out of GPU memory even at micro-batch 1. "
                        "Reduce video.num_frames or video.image_longest_edge, or close other GPU processes."
                    ) from exc
                raise

            cursor += len(candidate_rows)
            progress.update(len(candidate_rows))
            del pixel_values, pixel_attention_mask, features, processed
    finally:
        progress.close()
        vision_model.to(original_vision_device)
        connector.to(original_connector_device)
        model.train(was_training)
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    cache.assert_complete(event_ids)
    size_gib = cache.features_path.stat().st_size / (1024**3)
    print(
        f"Offline visual extraction complete: {len(cache):,}/{len(event_ids):,} events; "
        f"features file={cache.features_path} ({size_gib:.2f} GiB)."
    )
    return cache
