from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from badminton_decision.data.feature_extraction import extract_visual_features
from badminton_decision.models.smolvlm_decision import SmolVLMChoiceModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-extract frozen SmolVLM2 visual embeddings once per unique event "
            "and save them to a resumable disk-backed cache."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the visual cache.")
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    model = SmolVLMChoiceModel(
        model_id=model_cfg.get("name"),
        dtype=model_cfg.get("torch_dtype", "float16"),
        attn_implementation=model_cfg.get("attn_implementation", "eager"),
        lora_cfg={"enabled": False},
        head_cfg=cfg["head"],
        gradient_checkpointing=False,
        local_files_only=model_cfg.get("local_files_only", False),
    )
    target = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extraction device: {target}")
    extract_visual_features(cfg, model, device=target, force=args.force)


if __name__ == "__main__":
    main()
