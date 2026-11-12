#!/usr/bin/env bash
set -euo pipefail

# End-to-end setup helper. Runs the real model only when --train is supplied.
DOWNLOAD=1
PREPARE=1
TRAIN=0
CONFIG="configs/default.yaml"
RUN_NAME="badminton-jev-smolvlm2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download) DOWNLOAD=0; shift ;;
    --skip-prepare) PREPARE=0; shift ;;
    --train) TRAIN=1; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --name) RUN_NAME="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/quickstart.sh [--skip-download] [--skip-prepare] [--train] [--config PATH] [--name RUN]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

python scripts/check_environment.py
pytest -q
if [[ "$DOWNLOAD" == "1" ]]; then
  python scripts/download_data.py --output-dir data/raw/finebadminton-20k
fi
if [[ "$PREPARE" == "1" ]]; then
  python scripts/inspect_data.py --data-dir data/raw/finebadminton-20k
  python scripts/prepare_data.py --data-dir data/raw/finebadminton-20k --output-dir data/processed \
    --config "$CONFIG" --templates configs/question_templates.yaml --seed 42
fi
if [[ "$TRAIN" == "1" ]]; then
  CACHE_ENABLED="$(python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("true" if config.get("visual_cache", {}).get("enabled", False) else "false")
PY
)"
  if [[ "$CACHE_ENABLED" == "true" ]]; then
    python scripts/extract_visual_features.py --config "$CONFIG"
  fi
  accelerate launch --num_processes 1 scripts/train.py --config "$CONFIG" --name "$RUN_NAME"
else
  echo "Data preparation complete. Add --train to start fine-tuning."
fi
