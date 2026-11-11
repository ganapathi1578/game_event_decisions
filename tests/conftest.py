"""Make the src-layout package importable when tests are run from a checkout.

Editable installation is still recommended for CLI entry points. This small
bootstrap also lets `pytest` collect tests before the project is installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if SRC_DIR.is_dir():
    src_path = str(SRC_DIR)
    if src_path in sys.path:
        sys.path.remove(src_path)
    sys.path.insert(0, src_path)
