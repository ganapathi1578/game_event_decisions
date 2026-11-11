from pathlib import Path

from badminton_decision.data.prepare import make_qa_records, split_by_match


def _event(match_id: str, index: int, label: str):
    return {
        "event_id": f"{match_id}_{index}",
        "match_id": match_id,
        "video_name": f"{match_id}.mp4",
        "video_path": f"{match_id}.mp4",
        "fps": 30.0,
        "target_frame": float(index * 10),
        "hit_type": label,
        "subtype": [f"{label} subtype"],
        "hitter": "top" if index % 2 else "bottom",
        "ball_area": "mid court" if index % 2 else "back court",
    }


def test_make_qa_records_creates_separate_template_banks():
    events = []
    for match_index in range(10):
        for event_index in range(4):
            events.append(_event(f"m{match_index}", event_index, "smash" if event_index % 2 else "clear"))

    split = split_by_match(events, 0.8, 0.1, 0.1, 7)
    templates = {
        "hit_type": {
            "train": ["Train template A", "Train template B"],
            "validation": ["Validation template"],
            "test": ["Test template"],
        },
        "subtype": {
            "train": ["Train subtype"],
            "validation": ["Validation subtype"],
            "test": ["Test subtype"],
        },
        "hitter": {
            "train": ["Train hitter"],
            "validation": ["Validation hitter"],
            "test": ["Test hitter"],
        },
        "ball_area": {
            "train": ["Train area"],
            "validation": ["Validation area"],
            "test": ["Test area"],
        },
    }
    rows, summary = make_qa_records(
        events, split, templates,
        include_tasks=["hit_type", "subtype", "hitter", "ball_area"],
        min_class_count=2,
    )
    assert rows["train"] and rows["validation"] and rows["test"]
    assert set(r["match_id"] for r in rows["train"]).isdisjoint(r["match_id"] for r in rows["test"])
    assert all("Train" in r["question_templates"][0] for r in rows["train"])
    assert all("Validation" in r["question_templates"][0] for r in rows["validation"])
    assert all("Test" in r["question_templates"][0] for r in rows["test"])
    assert summary["label_vocabularies"]["hit_type"] == ["clear", "smash"]
