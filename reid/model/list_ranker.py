"""Candidate-set reranker: one query and its static-gallery top10 only."""

import torch
from torch import nn
from torch.nn import functional as F


class RelationBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.edge = nn.Sequential(
            nn.Linear(2 * hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.value = nn.Linear(hidden, hidden)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edges):
        batch, k, d = h.shape
        left = h[:, :, None, :].expand(batch, k, k, d)
        right = h[:, None, :, :].expand(batch, k, k, d)
        affinity = self.edge(torch.cat([left, right, edges], dim=-1)).squeeze(-1)
        diagonal = torch.eye(k, dtype=torch.bool, device=h.device)[None]
        attention = affinity.masked_fill(diagonal, -1e4).softmax(dim=2)
        message = (attention[..., None] * self.value(right)).sum(dim=2)
        return self.norm(h + self.update(torch.cat([h, message], dim=-1)))


class CandidateGraphRanker(nn.Module):
    def __init__(self, features=13, hidden=64, depth=2):
        super().__init__()
        self.config = dict(features=features, hidden=hidden, depth=depth)
        self.embed = nn.Sequential(
            nn.Linear(features, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList([RelationBlock(hidden) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, nodes, edges, base):
        if (
            nodes.ndim != 3
            or nodes.shape[1] < 2
            or edges.shape != (*nodes.shape[:2], nodes.shape[1], 2)
            or base.shape != nodes.shape[:2]
        ):
            raise ValueError("Expected nodes[B,K,F],edges[B,K,K,2],base[B,K],K>=2")
        h = self.embed(nodes)
        for block in self.blocks:
            h = block(h, edges)
        residual = 3 * torch.tanh(self.head(h).squeeze(-1) / 3)
        return base + residual, residual

    @torch.inference_mode()
    def score_lists(self, nodes, edges, base):
        if self.training:
            raise RuntimeError("Use eval() before scoring")
        # Keep every neural operation single-query, regardless of API batching.
        return torch.cat(
            [
                self(nodes[i : i + 1], edges[i : i + 1], base[i : i + 1])[0]
                for i in range(len(nodes))
            ]
        )


def ranking_loss(scores, positive, residual):
    target = positive.float()
    count = target.sum(1)
    valid = count > 0
    regularizer = 0.01 * residual.square().mean()
    if not valid.any():
        return scores.sum() * 0 + regularizer
    target = target / count.clamp_min(1)[:, None]
    return (
        -(target[valid] * F.log_softmax(scores[valid], dim=1)).sum(1).mean()
        + regularizer
    )
