"""Парный re-ranker: сравнивает два кропа и решает, одна ли это машина (D-028).

Зачем: у 95,4% query нужный кадр уже в top-10, но не на первом месте. Идеальный порядок внутри
десятки дал бы mAP@10 = 0,954 против 0,758 у эмбеддера. Переранжирование не входит в замер
скорости (`extract()`), поэтому здесь можно позволить себе тяжёлую модель.

Схема: токены обоих кропов берутся из обученного эмбеддера (он заморожен), к ним добавляется
эмбеддинг «свой/чужой кадр», дальше несколько слоёв self-attention по объединённой
последовательности — то есть модель может сопоставлять части машин между кадрами. На выходе —
логит «та же машина».
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PairReranker(nn.Module):
    def __init__(self, dim: int, layers: int = 4, heads: int = 8, proj: int = 384,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, proj)
        self.segment = nn.Parameter(torch.zeros(2, 1, proj))  # какой кадр: query или кандидат
        self.cls = nn.Parameter(torch.zeros(1, 1, proj))
        nn.init.trunc_normal_(self.segment, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(proj, heads, proj * 4, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(proj)
        self.head = nn.Linear(proj, 1)

    def forward(self, tok_a: torch.Tensor, tok_b: torch.Tensor) -> torch.Tensor:
        """tok_a, tok_b: [B, N, D] — токены двух кропов. Возвращает логит [B]."""
        a = self.proj(tok_a) + self.segment[0]
        b = self.proj(tok_b) + self.segment[1]
        cls = self.cls.expand(a.size(0), -1, -1)
        x = self.encoder(torch.cat([cls, a, b], dim=1))
        return self.head(self.norm(x[:, 0])).squeeze(-1)

    @torch.inference_mode()
    def score_pairs(self, tok_q: torch.Tensor, tok_c: torch.Tensor, batch: int = 64) -> torch.Tensor:
        """Оценки для одного query против нескольких кандидатов: tok_q [N, D], tok_c [K, N, D]."""
        out = []
        for i in range(0, tok_c.size(0), batch):
            chunk = tok_c[i:i + batch]
            out.append(self(tok_q.unsqueeze(0).expand(chunk.size(0), -1, -1), chunk))
        return torch.cat(out)


def rerank(cosine: torch.Tensor, logits: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Итоговый порядок: косинус плюс взвешенная вероятность от re-ranker'а.

    Оставляем вклад косинуса, чтобы плохо обученный re-ranker не мог всё испортить: при weight=0
    порядок в точности прежний.
    """
    return cosine + weight * torch.sigmoid(logits)
