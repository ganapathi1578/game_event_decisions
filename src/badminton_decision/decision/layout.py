from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ChoiceLayout:
    position_ids: torch.Tensor
    allowed_attention: torch.Tensor
    choice_read_indices: torch.Tensor
    answer_end_index: int
    option_spans: list[tuple[int, int]]


def build_choice_layout(
    context_length: int,
    option_token_ids: list[list[int]],
    option_open_ids: list[int],
    option_close_ids: list[int],
    answer_token_ids: list[int],
) -> ChoiceLayout:
    """Reproduces the BEV/Jev option layout: shared option position starts and isolated blocks.

    Context tokens are causal. Each option can attend to the complete context and its own
    causal block, but cannot attend to another option. The final answer prompt can attend to
    context plus all choices. Every option block starts at the same logical RoPE position.
    """
    if context_length < 1:
        raise ValueError("context_length must be >= 1")
    if not option_token_ids:
        raise ValueError("At least one option is required")
    if not option_open_ids or not option_close_ids or not answer_token_ids:
        raise ValueError("Option markers and answer prompt must tokenize to non-empty sequences")

    option_sequences: list[list[int]] = []
    for option in option_token_ids:
        if len(option) == 0:
            raise ValueError("An option tokenized to an empty sequence")
        option_sequences.append(option_open_ids + option + option_close_ids)

    seq_len = context_length + sum(len(x) for x in option_sequences) + len(answer_token_ids)
    position_ids = torch.empty(seq_len, dtype=torch.long)
    position_ids[:context_length] = torch.arange(context_length)

    option_spans: list[tuple[int, int]] = []
    choice_read_indices: list[int] = []
    cursor = context_length
    option_position_start = context_length
    max_option_position = context_length - 1
    for full_option, content_option in zip(option_sequences, option_token_ids):
        start, end = cursor, cursor + len(full_option)
        option_spans.append((start, end))
        # Reset every option to the same starting logical position.
        position_ids[start:end] = torch.arange(option_position_start, option_position_start + len(full_option))
        # The reference implementation stores choices_end_idx after appending the full
        # </option> marker and reads the hidden state at that final marker token.
        read_index = end - 1
        choice_read_indices.append(read_index)
        max_option_position = max(max_option_position, option_position_start + len(full_option) - 1)
        cursor = end

    answer_start = cursor
    position_ids[answer_start:] = torch.arange(max_option_position + 1, max_option_position + 1 + len(answer_token_ids))
    answer_end = seq_len - 1

    allowed = torch.zeros((seq_len, seq_len), dtype=torch.bool)
    # Context block: standard causal attention.
    allowed[:context_length, :context_length] = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
    # Options: see all context, plus their own causal block only.
    for start, end in option_spans:
        allowed[start:end, :context_length] = True
        allowed[start:end, start:end] = torch.tril(torch.ones(end - start, end - start, dtype=torch.bool))
    # Answer prompt: see all preceding context/choices and itself causally.
    if answer_start < seq_len:
        allowed[answer_start:, :answer_start] = True
        allowed[answer_start:, answer_start:] = torch.tril(torch.ones(seq_len - answer_start, seq_len - answer_start, dtype=torch.bool))

    # Prevent all-masked attention rows (notably relevant if the method is extended to padding).
    if not allowed.diagonal().all():
        raise RuntimeError("Every token must be allowed to attend to itself")
    return ChoiceLayout(
        position_ids=position_ids,
        allowed_attention=allowed,
        choice_read_indices=torch.tensor(choice_read_indices, dtype=torch.long),
        answer_end_index=answer_end,
        option_spans=option_spans,
    )


def to_additive_mask(allowed_attention: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert True/False allowed mask to [1, 1, L, L] additive mask."""
    if allowed_attention.ndim != 2 or allowed_attention.shape[0] != allowed_attention.shape[1]:
        raise ValueError("allowed_attention must be square [L, L]")
    mask = torch.zeros(allowed_attention.shape, dtype=dtype, device=allowed_attention.device)
    mask = mask.masked_fill(~allowed_attention, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)
