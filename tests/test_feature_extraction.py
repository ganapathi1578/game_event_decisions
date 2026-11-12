from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

import badminton_decision.data.feature_extraction as extraction
from badminton_decision.data.feature_cache import VisualFeatureCache


class _FakeReader:
    def __init__(self, num_frames, duration_seconds, cache_size, jitter_seconds):
        self.num_frames = num_frames

    def read(self, *args, **kwargs):
        return [object() for _ in range(self.num_frames)]


class _FakeImageProcessor:
    def __init__(self):
        self.size = {}
        self.max_image_size = {}
        self.do_image_splitting = True

    def __call__(self, images, return_tensors):
        return {
            "pixel_values": torch.ones(len(images), 3, 4, 4),
            "pixel_attention_mask": torch.ones(len(images), 4, 4, dtype=torch.bool),
        }


class _FakeVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 1)


class _FakeFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = SimpleNamespace(
            tokenizer=object(), image_processor=_FakeImageProcessor()
        )
        self.vlm = nn.Module()
        self.vlm.model = nn.Module()
        self.vlm.model.vision_model = _FakeVisionModel()
        self.vlm.model.connector = _FakeVisionModel()

    def _encode_visual(self, pixel_values, pixel_attention_mask):
        return [torch.full((8, 4), float(index + 1)) for index in range(pixel_values.shape[0])]


def _write_manifest(path: Path, event_ids: list[str]) -> None:
    rows = []
    for event_id in event_ids:
        rows.append({
            "event_id": event_id,
            "sample_id": f"{event_id}__hit_type",
            "video_path": "dummy.mp4",
            "target_frame": 10,
            "fps": 30,
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_offline_extraction_writes_one_disk_row_per_unique_event(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _write_manifest(processed_dir / "train.jsonl", ["event-a", "event-b"])
    _write_manifest(processed_dir / "validation.jsonl", ["event-b"])
    _write_manifest(processed_dir / "test.jsonl", ["event-c"])
    monkeypatch.setattr(extraction, "VideoFrameReader", _FakeReader)

    cfg = {
        "project": {"data_dir": str(tmp_path / "raw"), "processed_dir": str(processed_dir)},
        "model": {"name": "test-model"},
        "video": {
            "num_frames": 8, "duration_seconds": 3.0, "image_longest_edge": 512,
            "cache_size": 2,
        },
        "visual_cache": {
            "directory": str(tmp_path / "visual-cache"), "feature_dtype": "float16",
            "extraction_batch_size": 2, "max_extraction_batch_size": 2,
            "items_per_shard": 2,
        },
        "choice": {"context_max_tokens": 32, "option_max_tokens": 8, "task_prompt": "Choose."},
    }
    model = _FakeFeatureModel()

    cache = extraction.extract_visual_features(cfg, model, device="cpu")

    assert len(cache) == 3
    assert cache.event_ids == ["event-a", "event-b", "event-c"]
    assert cache.features_path.exists()
    assert torch.equal(cache.get("event-a"), torch.ones(8, 4, dtype=torch.float16))
    assert torch.equal(cache.get("event-c"), torch.ones(8, 4, dtype=torch.float16))
    reopened = VisualFeatureCache(cache.cache_dir, extraction.visual_cache_metadata(cfg))
    reopened.assert_complete(cache.event_ids)


def test_auto_extraction_batch_uses_available_gpu_memory(monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (10 * gib, 16 * gib))

    selected = extraction.choose_extraction_batch_size(
        torch.device("cuda"),
        {
            "extraction_batch_size": "auto",
            "max_extraction_batch_size": 8,
            "memory_fraction": 0.5,
            "reserve_gib": 2.0,
            "estimated_gib_per_event": 1.5,
        },
    )

    # floor((10 GiB * 0.5 - 2 GiB) / 1.5 GiB per event) == 2.
    assert selected == 2
