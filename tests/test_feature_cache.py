from __future__ import annotations

import torch

from badminton_decision.data.feature_cache import VisualFeatureCache


def test_disk_cache_round_trip_and_resume(tmp_path):
    metadata = {
        "model_id": "test-model",
        "num_frames": 8,
        "feature_dtype": "float16",
    }
    event_ids = ["event-a", "event-b", "event-c"]
    cache = VisualFeatureCache(tmp_path / "visual", metadata, event_ids=event_ids, create=True)

    cache.write_batch(
        ["event-a", "event-c"],
        [torch.full((5, 4), 1.25), torch.full((5, 4), 3.5)],
    )
    assert len(cache) == 2
    assert cache.missing(event_ids) == ["event-b"]
    assert torch.allclose(cache.get("event-a"), torch.full((5, 4), 1.25, dtype=torch.float16))

    # Reopening the cache simulates a separate process after offline extraction.
    reopened = VisualFeatureCache(tmp_path / "visual", metadata)
    assert len(reopened) == 2
    assert reopened.missing(event_ids) == ["event-b"]
    assert torch.allclose(reopened.get("event-c"), torch.full((5, 4), 3.5, dtype=torch.float16))
    reopened.write_batch(["event-b"], [torch.full((5, 4), 2.0)])
    reopened.assert_complete(event_ids)
    assert len(reopened) == 3


def test_cache_rejects_configuration_mismatch(tmp_path):
    cache_dir = tmp_path / "visual"
    cache = VisualFeatureCache(
        cache_dir,
        {"model_id": "test-model", "num_frames": 8, "feature_dtype": "float16"},
        event_ids=["event-a"],
        create=True,
    )
    cache.write_batch(["event-a"], [torch.zeros(2, 3)])

    try:
        VisualFeatureCache(cache_dir, {"model_id": "different-model", "num_frames": 8, "feature_dtype": "float16"})
    except RuntimeError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("Expected an incompatible-cache error")


def test_cache_checks_available_disk_space_before_allocating(tmp_path, monkeypatch):
    from collections import namedtuple

    import badminton_decision.data.feature_cache as feature_cache_module

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        feature_cache_module.shutil,
        "disk_usage",
        lambda path: Usage(total=1024**3, used=1024**3 - 1024, free=1024),
    )
    cache = VisualFeatureCache(
        tmp_path / "visual",
        {"model_id": "test-model", "num_frames": 8, "feature_dtype": "float16"},
        event_ids=["event-a"],
        create=True,
    )

    try:
        cache.write_batch(["event-a"], [torch.zeros(512, 576)])
    except OSError as error:
        assert "Insufficient local disk space" in str(error)
    else:
        raise AssertionError("Expected a disk-space preflight failure")
