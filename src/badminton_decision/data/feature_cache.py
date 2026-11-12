from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


_NUMPY_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "bfloat16": np.float16,  # NumPy has no native bfloat16; bfloat16 cache requests use FP16.
}
_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.float16,
}


class VisualFeatureCache:
    """Disk-backed event-level visual feature cache.

    Features are stored in one NumPy ``.npy`` memmap, with shape
    ``[num_events, visual_tokens, hidden_dim]``. Each item is read on demand, so
    the complete cache does not need to fit in system RAM or GPU VRAM. A small
    JSON index maps stable event IDs to rows; a completion bitmap allows safe
    extraction resume after interruption.
    """

    FORMAT_VERSION = 2

    def __init__(
        self,
        cache_dir: str | Path,
        expected_metadata: dict[str, Any] | None = None,
        *,
        event_ids: list[str] | None = None,
        create: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / "metadata.json"
        self.index_path = self.cache_dir / "index.json"
        self.features_path = self.cache_dir / "features.npy"
        self.completed_path = self.cache_dir / "completed.npy"
        self._features: np.memmap | None = None
        self._completed: np.memmap | None = None

        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if expected_metadata:
                self._validate_metadata(expected_metadata)
        elif create:
            self.metadata = dict(expected_metadata or {})
            self.metadata["format_version"] = self.FORMAT_VERSION
            self._write_json_atomic(self.metadata_path, self.metadata)
        else:
            raise FileNotFoundError(
                f"Visual cache metadata not found: {self.metadata_path}. "
                "Run scripts/extract_visual_features.py first."
            )

        if self.metadata.get("format_version") != self.FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported visual cache format {self.metadata.get('format_version')}; "
                f"expected {self.FORMAT_VERSION}. Rebuild with --force."
            )

        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        elif create and event_ids is not None:
            self.index = self._make_index(event_ids)
            self._write_json_atomic(self.index_path, self.index)
        else:
            raise FileNotFoundError(f"Visual cache index not found: {self.index_path}")

        if event_ids is not None:
            requested = [str(item) for item in event_ids]
            existing = self.index.get("event_ids", [])
            if requested != existing:
                raise RuntimeError(
                    "The event list differs from the one used to create this visual cache. "
                    "This can happen if the prepared manifests changed. Use a new cache directory "
                    "or rerun extraction with --force."
                )
        self.event_ids: list[str] = list(self.index.get("event_ids", []))
        self.event_to_row: dict[str, int] = {
            str(key): int(value) for key, value in self.index.get("event_to_row", {}).items()
        }
        if len(self.event_ids) != len(self.event_to_row):
            raise RuntimeError("Visual cache index is inconsistent: event IDs and row mapping differ")

        if self.features_path.exists() and self.completed_path.exists():
            self._open_storage(mode="r")
        elif not create:
            raise FileNotFoundError(
                f"Visual feature storage is missing from {self.cache_dir}. "
                "Run scripts/extract_visual_features.py first."
            )

    @staticmethod
    def _make_index(event_ids: list[str]) -> dict[str, Any]:
        normalized = [str(item) for item in event_ids]
        if len(normalized) != len(set(normalized)):
            raise ValueError("event_ids must be unique")
        return {
            "event_ids": normalized,
            "event_to_row": {event_id: index for index, event_id in enumerate(normalized)},
        }

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    def _validate_metadata(self, expected: dict[str, Any]) -> None:
        mismatched = {
            key: (self.metadata.get(key), value)
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                "Visual cache extraction configuration does not match the current project. "
                f"Mismatches: {mismatched}. Use a new cache directory or re-extract with --force."
            )

    def _open_storage(self, mode: str) -> None:
        feature_shape = self.metadata.get("feature_shape")
        if not feature_shape:
            raise RuntimeError("Visual cache storage exists but feature_shape is absent from metadata")
        dtype_name = str(self.metadata.get("feature_dtype", "float16"))
        if dtype_name not in _NUMPY_DTYPES:
            raise RuntimeError(f"Unsupported feature_dtype in visual cache: {dtype_name}")
        shape = (len(self.event_ids), int(feature_shape[0]), int(feature_shape[1]))
        self._features = np.load(self.features_path, mmap_mode=mode)
        completed_mode = "r" if mode == "r" else "r+"
        self._completed = np.load(self.completed_path, mmap_mode=completed_mode)
        if tuple(self._features.shape) != shape:
            raise RuntimeError(
                f"Visual cache feature file has shape {self._features.shape}; expected {shape}."
            )
        if tuple(self._completed.shape) != (len(self.event_ids),):
            raise RuntimeError("Visual cache completion bitmap has an invalid shape")

    def initialize_storage(self, feature_shape: tuple[int, int]) -> None:
        """Create the on-disk arrays after the first encoder batch reveals output shape."""
        feature_shape = (int(feature_shape[0]), int(feature_shape[1]))
        if min(feature_shape) <= 0:
            raise ValueError(f"Invalid visual feature shape: {feature_shape}")
        if self.features_path.exists() and self.completed_path.exists():
            recorded = tuple(self.metadata.get("feature_shape", []))
            if recorded != feature_shape:
                raise RuntimeError(
                    f"Existing cache shape {recorded} differs from extracted shape {feature_shape}. "
                    "Rebuild with --force."
                )
            if self._features is None or getattr(self._features, "mode", "r") == "r":
                self._features = None
                self._completed = None
                self._open_storage(mode="r+")
            return
        if self.features_path.exists() or self.completed_path.exists():
            # Recover from interruption during the two-file initialization.
            self.features_path.unlink(missing_ok=True)
            self.completed_path.unlink(missing_ok=True)
            self.metadata.pop("feature_shape", None)
            self._write_json_atomic(self.metadata_path, self.metadata)

        dtype_name = str(self.metadata.get("feature_dtype", "float16"))
        if dtype_name not in _NUMPY_DTYPES:
            raise ValueError(f"Unsupported feature_dtype: {dtype_name}")
        shape = (len(self.event_ids), *feature_shape)
        required_bytes = int(np.prod(shape)) * np.dtype(_NUMPY_DTYPES[dtype_name]).itemsize
        required_bytes += len(self.event_ids)  # Boolean completion bitmap.
        free_bytes = shutil.disk_usage(self.cache_dir).free
        safety_reserve = 512 * 1024**2
        if required_bytes > max(0, free_bytes - safety_reserve):
            required_gib = required_bytes / (1024**3)
            available_gib = max(0, free_bytes - safety_reserve) / (1024**3)
            raise OSError(
                "Insufficient local disk space for the visual-feature cache: "
                f"need approximately {required_gib:.2f} GiB, have {available_gib:.2f} GiB "
                "after a 0.5 GiB safety reserve. Choose a cache directory with more free space."
            )
        feature_tmp = self.features_path.with_suffix(".npy.tmp")
        completed_tmp = self.completed_path.with_suffix(".npy.tmp")
        # Persist the expected shape before publishing the data files. If extraction
        # is interrupted during file creation, the next --force-free run can recover.
        self.metadata["feature_shape"] = list(feature_shape)
        self._write_json_atomic(self.metadata_path, self.metadata)
        feature_map = np.lib.format.open_memmap(
            feature_tmp, mode="w+", dtype=_NUMPY_DTYPES[dtype_name], shape=shape
        )
        completed_map = np.lib.format.open_memmap(
            completed_tmp, mode="w+", dtype=np.bool_, shape=(len(self.event_ids),)
        )
        completed_map[:] = False
        feature_map.flush()
        completed_map.flush()
        del feature_map, completed_map
        os.replace(feature_tmp, self.features_path)
        os.replace(completed_tmp, self.completed_path)
        self._open_storage(mode="r+")

    def __len__(self) -> int:
        if self._completed is None:
            return 0
        return int(np.asarray(self._completed).sum())

    def contains(self, event_id: str) -> bool:
        if self._completed is None:
            return False
        row = self.event_to_row.get(str(event_id))
        return row is not None and bool(self._completed[row])

    def missing(self, event_ids: list[str]) -> list[str]:
        return [str(event_id) for event_id in event_ids if not self.contains(str(event_id))]

    def write_batch(self, event_ids: list[str], features: list[torch.Tensor] | torch.Tensor) -> None:
        if not event_ids:
            return
        if isinstance(features, torch.Tensor):
            tensor = features.detach().cpu()
        else:
            shapes = {tuple(item.shape) for item in features}
            if len(shapes) != 1:
                raise ValueError(f"Visual features in a batch must have equal shapes; got {shapes}")
            tensor = torch.stack([item.detach().cpu() for item in features], dim=0)
        if tensor.ndim != 3 or tensor.shape[0] != len(event_ids):
            raise ValueError(
                f"Expected [batch, tokens, hidden] features for {len(event_ids)} events; "
                f"got {tuple(tensor.shape)}"
            )
        if len({str(item) for item in event_ids}) != len(event_ids):
            raise ValueError("A cache write batch contains duplicate event IDs")
        self.initialize_storage((int(tensor.shape[1]), int(tensor.shape[2])))
        if self._features is None or self._completed is None:
            raise RuntimeError("Visual feature storage was not initialized")

        rows = []
        for event_id in event_ids:
            row = self.event_to_row.get(str(event_id))
            if row is None:
                raise KeyError(f"Event {event_id!r} is absent from the visual cache index")
            if bool(self._completed[row]):
                raise ValueError(f"Refusing to overwrite completed visual features for {event_id!r}")
            rows.append(row)

        dtype_name = str(self.metadata.get("feature_dtype", "float16"))
        np_dtype = _NUMPY_DTYPES[dtype_name]
        values = tensor.float().numpy().astype(np_dtype, copy=False)
        self._features[rows] = values
        self._features.flush()
        self._completed[rows] = True
        self._completed.flush()

    def get(self, event_id: str) -> torch.Tensor:
        key = str(event_id)
        row = self.event_to_row.get(key)
        if row is None or self._completed is None or not bool(self._completed[row]):
            raise KeyError(
                f"Event {key!r} is not present in the completed visual cache {self.cache_dir}. "
                "Run scripts/extract_visual_features.py first."
            )
        if self._features is None:
            self._open_storage(mode="r")
        # Copy this row out of the read-only memmap before wrapping it as a tensor.
        array = np.array(self._features[row], copy=True)
        dtype_name = str(self.metadata.get("feature_dtype", "float16"))
        return torch.from_numpy(array).to(dtype=_TORCH_DTYPES[dtype_name])

    def assert_complete(self, event_ids: list[str]) -> None:
        missing = self.missing(event_ids)
        if missing:
            preview = ", ".join(missing[:8])
            raise RuntimeError(
                f"Visual cache is missing {len(missing):,} required events (e.g. {preview}). "
                "Run: python scripts/extract_visual_features.py --config configs/default.yaml"
            )
