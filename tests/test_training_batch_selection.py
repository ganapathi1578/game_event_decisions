from __future__ import annotations

import torch
import torch.nn as nn

from badminton_decision.training.trainer import _resolve_training_batch_size


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1024))


def test_auto_training_batch_is_conservative_for_a_16_gb_gpu(monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (15 * gib, 16 * gib))
    cfg = {
        "training": {
            "per_device_batch_size": "auto",
            "max_auto_batch_size": 8,
            "auto_batch_reserve_gib": 4.0,
            "auto_batch_estimated_gib_per_example": 4.0,
        }
    }

    selected = _resolve_training_batch_size(cfg, torch.device("cuda"), _TinyModel())

    assert selected == 2


def test_explicit_training_batch_size_is_respected():
    cfg = {"training": {"per_device_batch_size": 3}}
    assert _resolve_training_batch_size(cfg, torch.device("cpu"), _TinyModel()) == 3
