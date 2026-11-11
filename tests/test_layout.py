import torch

from badminton_decision.decision.layout import build_choice_layout, to_additive_mask


def make_layout(options):
    return build_choice_layout(
        context_length=5,
        option_token_ids=[[10] * n for n in options],
        option_open_ids=[1], option_close_ids=[2], answer_token_ids=[3, 4],
    )


def test_options_share_logical_position_start():
    layout = make_layout([2, 3, 1])
    starts = [layout.position_ids[start].item() for start, _ in layout.option_spans]
    assert starts == [5, 5, 5]


def test_options_cannot_attend_to_each_other():
    layout = make_layout([2, 2, 2])
    for i, (start_i, end_i) in enumerate(layout.option_spans):
        for j, (start_j, end_j) in enumerate(layout.option_spans):
            if i != j:
                assert not layout.allowed_attention[start_i:end_i, start_j:end_j].any()


def test_each_option_can_attend_to_context_and_itself():
    layout = make_layout([2, 3])
    for start, end in layout.option_spans:
        assert layout.allowed_attention[start:end, :5].all()
        assert torch.equal(layout.allowed_attention[start:end, start:end], torch.tril(torch.ones(end-start, end-start, dtype=torch.bool)))


def test_answer_can_attend_to_all_options():
    layout = make_layout([2, 4, 1])
    answer_start = layout.answer_end_index - 1
    assert layout.allowed_attention[answer_start:, :answer_start].all()


def test_additive_mask_shape_and_values():
    layout = make_layout([2, 2])
    mask = to_additive_mask(layout.allowed_attention)
    assert mask.shape == (1, 1, layout.position_ids.numel(), layout.position_ids.numel())
    assert torch.isfinite(mask[0, 0][layout.allowed_attention]).all()
    assert torch.isneginf(mask[0, 0][~layout.allowed_attention]).all()


def test_choice_read_index_matches_closing_marker_end():
    layout = make_layout([2, 3])
    for read_idx, (_, end) in zip(layout.choice_read_indices.tolist(), layout.option_spans):
        assert read_idx == end - 1
