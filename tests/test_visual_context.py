import torch

from badminton_decision.data.dataset import build_smolvlm_context_ids


class DummyTokenizer:
    def __init__(self):
        self.special = {
            "<image>": 99,
            "<fake_token_around_image>": 98,
            "<global-img>": 97,
            "<end_of_utterance>": 96,
            "<|im_start|>": 95,
        }

    def encode(self, text, add_special_tokens=False):
        # A minimal tokenizer for testing that retains special-token IDs.
        import re
        parts = re.findall(r'<[^>]+>|[^<>]+', text)
        ids = []
        for part in parts:
            if part in self.special:
                ids.append(self.special[part])
            elif part.strip():
                ids.extend([1] * len(part.split()))
        return ids

    def convert_tokens_to_ids(self, token):
        return self.special[token]


class DummyProcessor:
    image_seq_len = 64
    image_token_id = 99
    image_token = "<image>"
    fake_image_token = "<fake_token_around_image>"
    global_image_token = "<global-img>"
    end_of_utterance_token = "<end_of_utterance>"


def test_visual_context_contains_one_image_token_sequence_per_frame():
    tokenizer = DummyTokenizer()
    processor = DummyProcessor()
    ids = build_smolvlm_context_ids(
        tokenizer, processor, num_frames=4, duration_seconds=3.0,
        question="Which stroke is performed?", task_prompt="Choose the best option.",
        context_max_tokens=32,
    )
    assert int((ids == processor.image_token_id).sum()) == 4 * processor.image_seq_len
    assert int((ids == 98).sum()) == 8
    assert int((ids == 97).sum()) == 4
    assert ids[-1].item() == 1


def test_visual_context_caps_only_question_text_not_image_placeholders():
    tokenizer = DummyTokenizer()
    processor = DummyProcessor()
    question = "word " * 200
    ids = build_smolvlm_context_ids(
        tokenizer, processor, num_frames=2, duration_seconds=3.0,
        question=question, task_prompt="Pick one.", context_max_tokens=8,
    )
    assert int((ids == processor.image_token_id).sum()) == 2 * processor.image_seq_len
