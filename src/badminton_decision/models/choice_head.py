from __future__ import annotations

import math

import torch
import torch.nn as nn


class HeadSelfAttention(nn.Module):
    """Single-head attention block used by the reference BEV choice head."""
    def __init__(self, hidden_dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.q = nn.Linear(hidden_dim, head_dim)
        self.k = nn.Linear(hidden_dim, head_dim)
        self.v = nn.Linear(hidden_dim, head_dim)
        self.o = nn.Linear(head_dim, hidden_dim)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~key_mask[:, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bqk,bkd->bqd", weights, v)
        return self.o(out)


class ChoiceAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = HeadSelfAttention(hidden_dim, head_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ChoiceHead(nn.Module):
    """Permutation-equivariant choice head: task token + unordered choices + answer token."""
    def __init__(self, hidden_dim: int, head_dim: int = 512, num_layers: int = 2, num_task_types: int = 4):
        super().__init__()
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([ChoiceAttentionBlock(hidden_dim, head_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.answer_proj = nn.Linear(hidden_dim, head_dim)
        self.choice_proj = nn.Linear(hidden_dim, head_dim)
        self.head_dim = head_dim

    def forward(self, choice_embeddings: torch.Tensor, answer_embedding: torch.Tensor,
                choice_mask: torch.Tensor, task_type: torch.Tensor) -> torch.Tensor:
        batch = choice_embeddings.shape[0]
        task_token = self.task_embedding(task_type).unsqueeze(1)
        x = torch.cat([task_token, choice_embeddings, answer_embedding.unsqueeze(1)], dim=1)
        x = self.input_norm(x)
        always_valid = torch.ones(batch, 1, dtype=torch.bool, device=choice_mask.device)
        key_mask = torch.cat([always_valid, choice_mask, always_valid], dim=1)
        for layer in self.layers:
            x = layer(x, key_mask)
        x = self.final_norm(x)
        answer = self.answer_proj(x[:, -1])
        choices = self.choice_proj(x[:, 1:-1])
        logits = torch.einsum("bd,bcd->bc", answer, choices) / math.sqrt(self.head_dim)
        return logits.masked_fill(~choice_mask, -torch.inf)
