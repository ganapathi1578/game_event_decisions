from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from badminton_decision.data.video import VideoFrameReader
from badminton_decision.utils import read_jsonl


def build_smolvlm_context_ids(
    tokenizer, processor, num_frames: int, duration_seconds: float, question: str,
    task_prompt: str, context_max_tokens: int,
) -> torch.Tensor:
    """Build a SmolVLM-native frame-placeholder prompt for already sampled frames.

    SmolVLM2 replaces each ``<image>`` placeholder with exactly ``image_seq_len``
    connector features. This project supplies the image features directly to the
    text decoder, so it must preserve the same placeholder count and surrounding
    image markers as the native processor.
    """
    image_seq_len = int(getattr(processor, "image_seq_len", 0))
    image_token = getattr(processor, "image_token", "<image>")
    fake_token = getattr(processor, "fake_image_token", "<fake_token_around_image>")
    global_token = getattr(processor, "global_image_token", "<global-img>")
    end_token = getattr(processor, "end_of_utterance_token", "<end_of_utterance>")
    if image_seq_len <= 0:
        raise RuntimeError("Loaded SmolVLM processor does not expose a positive image_seq_len")

    duration_int = max(1, int(round(float(duration_seconds))))
    minutes, seconds = divmod(duration_int, 60)
    prefix = (
        f"<|im_start|>User: You are provided the following series of {num_frames} "
        f"frames from a {minutes}:{seconds:02d} video.\n"
    )
    for frame_index in range(num_frames):
        timestamp = frame_index * float(duration_seconds) / max(num_frames, 1)
        frame_minutes, frame_seconds = divmod(int(timestamp), 60)
        prefix += (
            f"\nFrame from {frame_minutes:02d}:{frame_seconds:02d}:"
            f"{fake_token}{global_token}{image_token * image_seq_len}{fake_token}"
        )

    body_text = f"\n\n{task_prompt}\nQuestion: {question}\n\n"
    body_ids = tokenizer.encode(body_text, add_special_tokens=False)[:context_max_tokens]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(f"{end_token}\nAssistant:", add_special_tokens=False)
    context_ids = torch.tensor(prefix_ids + body_ids + suffix_ids, dtype=torch.long)

    image_token_id = int(getattr(processor, "image_token_id", tokenizer.convert_tokens_to_ids(image_token)))
    actual = int((context_ids == image_token_id).sum().item())
    expected = num_frames * image_seq_len
    if actual != expected:
        raise RuntimeError(
            f"Prompt contains {actual} image placeholders, but {expected} are required "
            f"({num_frames} frames x image_seq_len={image_seq_len}). Check the tokenizer's special tokens."
        )
    return context_ids


class ChoiceVideoDataset(Dataset):
    def __init__(self, manifest_path: str | Path, video_root: str | Path, split: str,
                 num_frames: int = 8, duration_seconds: float = 3.0, cache_size: int = 8,
                 jitter_seconds: float = 0.2, min_options: int = 2, max_options: int = 8,
                 shuffle_options: bool = True, max_samples: int | None = None, seed: int = 42):
        self.rows = read_jsonl(manifest_path)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        self.video_root = Path(video_root)
        self.split = split
        self.training = split == "train"
        self.min_options = min_options
        self.max_options = max_options
        self.shuffle_options = shuffle_options and self.training
        self.reader = VideoFrameReader(
            num_frames=num_frames,
            duration_seconds=duration_seconds,
            cache_size=cache_size,
            jitter_seconds=jitter_seconds if self.training else 0.0,
        )
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        video_path = Path(row["video_path"])
        if not video_path.is_absolute():
            video_path = self.video_root / video_path
        frames = self.reader.read(video_path, row["target_frame"], row["fps"], training=self.training)
        full_options = list(row["options"])
        answer = row["answer"]
        if answer not in full_options:
            raise ValueError(f"Answer {answer!r} is not in options for {row['sample_id']}")
        if self.training and len(full_options) > self.max_options:
            target_count = self.rng.randint(self.min_options, min(self.max_options, len(full_options)))
            negatives = [option for option in full_options if option != answer]
            selected = [answer] + self.rng.sample(negatives, target_count - 1)
        else:
            selected = full_options[:]
        if self.shuffle_options:
            self.rng.shuffle(selected)
        templates = row.get("question_templates") or ["Which option best matches the video?"]
        question = self.rng.choice(templates) if self.training else templates[index % len(templates)]
        return {
            "sample_id": row["sample_id"],
            "match_id": row["match_id"],
            "frames": frames,
            "duration_seconds": self.reader.duration_seconds,
            "question": question,
            "options": selected,
            "correct_index": selected.index(answer),
            "answer": answer,
            "task_type": row["task_type"],
            "metadata": row,
        }


class DecisionCollator:
    def __init__(self, processor, context_max_tokens: int = 160, option_max_tokens: int = 24,
                 task_prompt: str = "Choose the best option.", image_longest_edge: int = 512):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.context_max_tokens = context_max_tokens
        self.option_max_tokens = option_max_tokens
        self.task_prompt = task_prompt
        self.image_longest_edge = int(image_longest_edge)
        # SmolVLM2's stock image processor may tile a 720p frame into several 512px
        # crops (default longest edge is larger). For the short-clip training recipe,
        # enforce one 512px-longest-edge view per frame to keep visual-token count and
        # activation memory bounded. This is the same processor used at inference.
        image_processor = self.processor.image_processor
        if hasattr(image_processor, "size"):
            image_processor.size = {"longest_edge": self.image_longest_edge}
        if hasattr(image_processor, "max_image_size"):
            image_processor.max_image_size = {"longest_edge": self.image_longest_edge}
        if hasattr(image_processor, "do_image_splitting"):
            image_processor.do_image_splitting = False
        self.option_open = "<option>"
        self.option_close = "</option>"
        self.answer_prompt = "The answer is:"

    def _encode(self, text: str, add_special_tokens: bool) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return torch.tensor(ids, dtype=torch.long)

    def _process_frames(self, frames):
        image_processor = self.processor.image_processor
        result = image_processor(images=frames, return_tensors="pt")
        pixel_values = result["pixel_values"]
        if pixel_values.ndim == 4:
            # [frames, channels, height, width]
            pass
        elif pixel_values.ndim == 5:
            # Flatten image crops/tiles into the image sequence.
            pixel_values = pixel_values.flatten(0, 1)
        else:
            raise ValueError(f"Unexpected image pixel_values shape: {tuple(pixel_values.shape)}")
        pixel_mask = result.get("pixel_attention_mask")
        if pixel_mask is None:
            pixel_mask = torch.ones((pixel_values.shape[0], pixel_values.shape[-2], pixel_values.shape[-1]), dtype=torch.bool)
        else:
            if pixel_mask.ndim == 4:
                pixel_mask = pixel_mask.flatten(0, 1)
            if pixel_mask.ndim != 3:
                raise ValueError(f"Unexpected pixel_attention_mask shape: {tuple(pixel_mask.shape)}")
        return pixel_values, pixel_mask

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        processed = [self._process_frames(ex["frames"]) for ex in examples]
        shapes = {tuple(pv.shape) for pv, _ in processed}
        if len(shapes) != 1:
            raise ValueError(f"All processed video frames in a batch must have equal shapes; found {shapes}. Use batch size 1.")
        pixel_values = torch.stack([pv for pv, _ in processed], dim=0)
        pixel_attention_mask = torch.stack([pm for _, pm in processed], dim=0)

        context_ids, option_ids = [], []
        for ex in examples:
            context_ids.append(build_smolvlm_context_ids(
                self.tokenizer, self.processor, num_frames=len(ex["frames"]),
                duration_seconds=float(ex.get("duration_seconds", 3.0)),
                question=ex["question"], task_prompt=self.task_prompt,
                context_max_tokens=self.context_max_tokens,
            ))
            sample_options = []
            for option in ex["options"]:
                sample_options.append(self._encode(str(option), add_special_tokens=False)[: self.option_max_tokens])
            option_ids.append(sample_options)

        return {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "context_ids": context_ids,
            "option_ids": option_ids,
            "correct_index": torch.tensor([ex["correct_index"] for ex in examples], dtype=torch.long),
            "task_type_names": [ex["task_type"] for ex in examples],
            "sample_ids": [ex["sample_id"] for ex in examples],
            "match_ids": [ex["match_id"] for ex in examples],
            "questions": [ex["question"] for ex in examples],
            "options_text": [ex["options"] for ex in examples],
            "answers": [ex["answer"] for ex in examples],
            "metadata": [ex["metadata"] for ex in examples],
        }
