"""Функции потерь: CE со сглаживанием меток и batch-hard triplet (рецепт Bag-of-Tricks)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTriplet(nn.Module):
    """Для каждого якоря — самый далёкий позитив и самый близкий негатив в батче."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(feat, feat, p=2)
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(len(labels), dtype=torch.bool, device=feat.device)
        pos = dist.masked_fill(~same | eye, float("-inf")).max(dim=1).values
        neg = dist.masked_fill(same, float("inf")).min(dim=1).values
        valid = torch.isfinite(pos) & torch.isfinite(neg)
        if not valid.any():
            return feat.sum() * 0.0
        return F.relu(pos[valid] - neg[valid] + self.margin).mean()


class ArcFace(nn.Module):
    """Классификатор с угловым отступом: сжимает класс в конус и раздвигает классы между собой.

    В обычной кросс-энтропии нет требования, чтобы разные машины были далеко друг от друга —
    достаточно правильной стороны границы. ArcFace добавляет отступ прямо в угол между
    эмбеддингом и центром класса, поэтому признаки получаются плотнее внутри класса.
    """

    def __init__(self, dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin = scale, margin

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(feat, dim=1), F.normalize(self.weight, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        return self.scale * torch.cos(theta + self.margin * target)


class ReIDLoss(nn.Module):
    """CE по логитам классификатора плюс triplet по признакам до BNNeck."""

    def __init__(self, label_smoothing: float = 0.1, margin: float = 0.3, triplet_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet = BatchHardTriplet(margin)
        self.w = triplet_weight

    def forward(self, logits, feat, labels):
        ce, tri = self.ce(logits, labels), self.triplet(feat, labels)
        return ce + self.w * tri, {"ce": ce.item(), "triplet": tri.item()}
