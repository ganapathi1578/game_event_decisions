from badminton_decision.data.prepare import split_by_match


def test_split_is_match_disjoint_and_complete():
    events = [{"match_id": f"m{i}"} for i in range(70) for _ in range(3)]
    splits = split_by_match(events, .8, .1, .1, 42)
    assert len(splits["train"]) == 56
    assert len(splits["validation"]) == 7
    assert len(splits["test"]) == 7
    assert not (splits["train"] & splits["validation"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["validation"] & splits["test"])
