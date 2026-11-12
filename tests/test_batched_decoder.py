from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from badminton_decision.models.choice_head import ChoiceHead
from badminton_decision.models.smolvlm_decision import SmolVLMChoiceModel


class _FakeTextModel(nn.Module):
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(256, hidden_dim)
        self.calls = 0
        self.last_attention_shape = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, position_ids, use_cache, return_dict):
        self.calls += 1
        self.last_attention_shape = tuple(attention_mask.shape)
        assert position_ids.shape[:2] == inputs_embeds.shape[:2]
        assert use_cache is False
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class _FakeVLMCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = _FakeTextModel()


class _FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeVLMCore()


def _make_uninitialized_model():
    model = object.__new__(SmolVLMChoiceModel)
    nn.Module.__init__(model)
    model.vlm = _FakeVLM()
    model.head = ChoiceHead(hidden_dim=8, head_dim=12, num_layers=1, num_task_types=4)
    model.hidden_dim = 8
    model.image_token_id = 99
    model.option_open_ids = [1]
    model.option_close_ids = [2]
    model.answer_prompt_ids = [3, 4]
    return model


def test_batched_decoder_executes_one_padded_decoder_call_for_independent_layouts():
    model = _make_uninitialized_model()
    visual = [torch.randn(2, 8), torch.randn(3, 8)]
    contexts = [torch.tensor([10, 99, 99, 11]), torch.tensor([10, 99, 99, 99, 11])]
    options = [
        [torch.tensor([20]), torch.tensor([21, 22])],
        [torch.tensor([30]), torch.tensor([31]), torch.tensor([32, 33])],
    ]

    logits = model._batched_text_forward(visual, contexts, options, ["hit_type", "subtype"])

    assert logits.shape == (2, 3)
    assert torch.isneginf(logits[0, 2])
    assert torch.isfinite(logits[0, :2]).all()
    assert torch.isfinite(logits[1, :3]).all()
    assert model.vlm.model.text_model.calls == 1
    assert model.vlm.model.text_model.last_attention_shape[0] == 2
    assert model.vlm.model.text_model.last_attention_shape[1] == 1
    assert model.vlm.model.text_model.last_attention_shape[2] == model.vlm.model.text_model.last_attention_shape[3]
