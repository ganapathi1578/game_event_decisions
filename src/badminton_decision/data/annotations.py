from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from badminton_decision.utils import get_ci, norm_key, read_json


def iter_annotation_records(payload: Any) -> Iterator[dict[str, Any]]:
    """Yield rally/video records from the released per-match JSON files."""
    if isinstance(payload, list):
        for item in payload:
            yield from iter_annotation_records(item)
        return
    if not isinstance(payload, dict):
        return
    if get_ci(payload, "hitting", "hits", "hit_annotations") is not None:
        yield payload
        return
    # Handle common wrapper layouts without assuming one fixed export schema.
    for key in ("data", "annotations", "rallies", "records", "items"):
        child = get_ci(payload, key)
        if isinstance(child, (list, dict)):
            yield from iter_annotation_records(child)


def _find_section(hit: dict[str, Any], section_name: str) -> dict[str, Any]:
    value = get_ci(hit, section_name, default={})
    return value if isinstance(value, dict) else {}


def _clean_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return None
    value = " ".join(value.strip().split())
    if not value or value.lower() in {"unknown", "none", "null", "n/a", "na", "-"}:
        return None
    return value


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    labels = []
    for item in value:
        cleaned = _clean_label(item)
        if cleaned and cleaned not in labels:
            labels.append(cleaned)
    return labels


def build_video_index(video_root: str | Path) -> dict[str, Path]:
    video_root = Path(video_root)
    index: dict[str, Path] = {}
    if not video_root.exists():
        return index
    for path in video_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
            resolved = path.resolve()
            index[path.name.lower()] = resolved
            index[path.stem.lower()] = resolved
    return index


def resolve_video_path(video_name: str, video_root: Path, index: dict[str, Path]) -> Path | None:
    raw = Path(video_name)
    candidates = [video_root / raw, video_root / raw.name, video_root / f"{raw.stem}.mp4"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return index.get(raw.name.lower()) or index.get(raw.stem.lower())


def parse_match_file(json_path: str | Path, video_root: str | Path, video_index: dict[str, Path] | None = None) -> list[dict[str, Any]]:
    json_path = Path(json_path)
    video_root = Path(video_root)
    video_index = video_index or build_video_index(video_root)
    payload = read_json(json_path)
    events: list[dict[str, Any]] = []
    record_index = 0
    for record in iter_annotation_records(payload):
        record_index += 1
        video_name = get_ci(record, "video", "video_name", "file_name", default=json_path.stem + ".mp4")
        video_path = resolve_video_path(str(video_name), video_root, video_index)
        fps = get_ci(record, "fps", "frame_rate", default=30.0)
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = 30.0
        if fps <= 0:
            fps = 30.0
        hits = get_ci(record, "hitting", "hits", "hit_annotations", default=[])
        if isinstance(hits, dict):
            hits = list(hits.values())
        if not isinstance(hits, list):
            continue
        match_id = Path(str(video_name)).stem or json_path.stem
        for hit_index, hit in enumerate(hits):
            if not isinstance(hit, dict):
                continue
            start_frame = get_ci(hit, "start_frame", "start", default=None)
            end_frame = get_ci(hit, "end_frame", "end", default=None)
            if start_frame is None or end_frame is None:
                continue
            try:
                start_frame, end_frame = int(start_frame), int(end_frame)
            except (TypeError, ValueError):
                continue
            if end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame
            foundation = _find_section(hit, "Foundational Actions Level")
            tactical = _find_section(hit, "Tactical Semantics Level")
            event = {
                "event_id": f"{match_id}_r{record_index:04d}_h{hit_index:04d}",
                "match_id": match_id,
                "video_name": Path(str(video_name)).name,
                # Store a path relative to the dataset's videos/ directory so the
                # prepared manifests remain portable when the dataset is moved.
                "video_path": (video_path.resolve().relative_to(video_root.resolve()).as_posix()
                               if video_path else None),
                "fps": fps,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "target_frame": (start_frame + end_frame) / 2.0,
                "player_name": _clean_label(get_ci(hit, "player", default=None)),
                "hit_type": _clean_label(get_ci(foundation, "hit type", "hit_type", default=None)),
                "subtype": _clean_list(get_ci(foundation, "subtype", default=[])),
                "hitter": _clean_label(get_ci(foundation, "hitter", default=None)),
                "ball_area": _clean_label(get_ci(foundation, "ball area", "ball_area", default=None)),
                "player_actions": _clean_list(get_ci(tactical, "player actions", default=[])),
                "shot_characteristics": _clean_list(get_ci(tactical, "shot characteristics", default=[])),
                "source_annotation": str(json_path),
            }
            if event["video_path"] is not None:
                events.append(event)
    return events


def parse_dataset(data_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_dir = Path(data_dir)
    ann_dir = data_dir / "finebadminton-20K"
    video_root = data_dir / "videos"
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    if not video_root.exists():
        raise FileNotFoundError(f"Video directory not found: {video_root}")
    video_index = build_video_index(video_root)
    json_files = sorted(ann_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json annotation files found beneath {ann_dir}")
    all_events = []
    file_reports = []
    for path in json_files:
        rows = parse_match_file(path, video_root, video_index)
        all_events.extend(rows)
        file_reports.append({"file": str(path), "parsed_hits": len(rows)})
    counts = {
        "hit_type": Counter(e["hit_type"] for e in all_events if e["hit_type"]),
        "hitter": Counter(e["hitter"] for e in all_events if e["hitter"]),
        "ball_area": Counter(e["ball_area"] for e in all_events if e["ball_area"]),
        "subtype_single": Counter(e["subtype"][0] for e in all_events if len(e["subtype"]) == 1),
    }
    summary = {
        "annotation_files": len(json_files),
        "indexed_videos": len({str(p) for p in video_index.values()}),
        "parsed_events": len(all_events),
        "matches": len({e["match_id"] for e in all_events}),
        "label_counts": {name: dict(counter.most_common()) for name, counter in counts.items()},
        "files": file_reports,
    }
    return all_events, summary
