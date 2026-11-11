from pathlib import Path
import json

from badminton_decision.data.annotations import parse_match_file


def test_parse_preview_style_annotation(tmp_path: Path):
    videos = tmp_path / "videos"
    ann = tmp_path / "finebadminton-20K"
    videos.mkdir(); ann.mkdir()
    (videos / "0001.mp4").write_bytes(b"placeholder")
    payload = [{
        "video": "0001.mp4", "fps": 30, "start_frame": 100, "end_frame": 250,
        "hitting": [{
            "start_frame": 120, "end_frame": 130, "player": "Player A",
            "Foundational Actions Level": {
                "hit type": "serve", "subtype": ["short serve"],
                "hitter": "bottom", "ball area": "mid front court"
            },
            "Tactical Semantics Level": {"player actions": ["serve"], "shot characteristics": []}
        }]
    }]
    path = ann / "0001.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = parse_match_file(path, videos)
    assert len(rows) == 1
    assert rows[0]["hit_type"] == "serve"
    assert rows[0]["subtype"] == ["short serve"]
    assert rows[0]["hitter"] == "bottom"
    assert rows[0]["video_path"] == "0001.mp4"
