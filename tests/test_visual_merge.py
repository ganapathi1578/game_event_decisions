import pytest
import torch

from badminton_decision.models.smolvlm_decision import (
    _group_legacy_visual_features,
    merge_visual_features_into_context,
)


def test_visual_features_replace_only_image_placeholder_embeddings():
    context_ids = torch.tensor([10, 99, 99, 11])
    context = torch.zeros(4, 3)
    features = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    merged = merge_visual_features_into_context(context, context_ids, features, image_token_id=99)
    assert torch.equal(merged[0], torch.zeros(3))
    assert torch.equal(merged[1:3], features)
    assert torch.equal(merged[3], torch.zeros(3))


def test_visual_merge_rejects_placeholder_feature_mismatch():
    with pytest.raises(RuntimeError, match="placeholder/feature mismatch"):
        merge_visual_features_into_context(
            torch.zeros(3, 4), torch.tensor([1, 99, 2]), torch.zeros(2, 4), image_token_id=99
        )


def test_legacy_visual_features_are_regrouped_by_batch_example():
    # Example 0 contains two real frames and one padded frame; example 1 has one.
    pixel_values = torch.zeros(2, 3, 3, 2, 2)
    pixel_values[0, 0] = 1.0
    pixel_values[0, 1] = 2.0
    pixel_values[1, 0] = 3.0
    flat_features = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8)

    grouped = _group_legacy_visual_features(flat_features, pixel_values, hidden_dim=8)

    assert len(grouped) == 2
    assert grouped[0].shape == (8, 8)
    assert grouped[1].shape == (4, 8)
    assert torch.equal(grouped[0], flat_features[:2].reshape(8, 8))
    assert torch.equal(grouped[1], flat_features[2:].reshape(4, 8))


def test_legacy_visual_feature_grouping_rejects_encoder_count_mismatch():
    pixel_values = torch.ones(2, 2, 3, 2, 2)
    flat_features = torch.zeros(3, 4, 8)
    with pytest.raises(RuntimeError, match="real images"):
        _group_legacy_visual_features(flat_features, pixel_values, hidden_dim=8)
