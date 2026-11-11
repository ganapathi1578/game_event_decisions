import pytest
import torch

from badminton_decision.models.smolvlm_decision import merge_visual_features_into_context


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
