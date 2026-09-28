"""Переранжировщик, который считает соответствие патчей ЯВНО.

Почему не как раньше. Прежняя голова склеивала токены двух кадров и отдавала их self-attention:
соответствие она должна была «догадаться» вычислить сама, на 7,5 тыс. обучающих кадров. Три
независимые правки обучения (пары из ошибок фузии, свой backbone, расписание трудности с
Hide-and-Seek) не сдвинули результат ни на сотую — A-058, A-059, A-060. Все работающие методы
соответствие считают явно: 4D-корреляция у CVNet, пулинг по матрице query-key у TransMatcher,
cross-attention по отобранным дескрипторам у AMES.

Что считаем здесь. Для каждого патча первого кадра берём матрицу сходств со всеми патчами второго
и извлекаем признаки, которые голове иначе неоткуда взять:
  * сила лучшего соответствия и среднее по трём лучшим — насколько деталь вообще нашлась;
  * взаимность — нашёл ли лучший кандидат этот же патч своим лучшим;
  * смещение лучшего соответствия по x и y — геометрия: у той же машины смещения согласованы,
    у разных машин одной модели они тоже согласованы, но сила соответствия ниже.
Дальше трансформер по последовательности патчей и голова.

Матрица сходств симметрична по построению, поэтому проблема прежней головы — разные правила для
двух порядков пары — здесь не возникает.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.model import build_backbone

N_FEATURES = 6  # best, top3, взаимность, dx, dy, расстояние


class CorrVerifier(nn.Module):
    def __init__(self, arch: str = "dinov2_vits14", input_size=(280, 280), layers: int = 3,
                 dim: int = 128, pretrained: bool = False, init_from: str | None = None):
        super().__init__()
        self.backbone = build_backbone(arch, input_size, pretrained=pretrained and not init_from)
        if init_from:
            from reid.model.verifier import PairVerifier
            PairVerifier._load_backbone(self, init_from)
        self.n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        self.embed = nn.Sequential(nn.Linear(N_FEATURES, dim), nn.GELU(), nn.Linear(dim, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, 0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        t = self.backbone.forward_features(x)
        return F.normalize(t[:, self.n_prefix:], dim=-1)

    def correspondence_features(self, ta: torch.Tensor, tb: torch.Tensor) -> torch.Tensor:
        """[B,N,D] и [B,N,D] → [B,N,6]: что нашёл каждый патч первого кадра во втором."""
        b, n, _ = ta.shape
        side = int(round(n ** 0.5))
        sim = torch.bmm(ta, tb.transpose(1, 2))                    # [B, N, N]
        top3 = sim.topk(min(3, n), dim=2).values
        best_val, best_idx = sim.max(dim=2)
        # Взаимность: лучший для patch_a кандидат — выбирает ли он patch_a своим лучшим
        back = sim.argmax(dim=1)                                   # [B, N] для патчей b
        mutual = (torch.gather(back, 1, best_idx) ==
                  torch.arange(n, device=sim.device).expand(b, n)).float()
        ay, ax = torch.arange(n, device=sim.device) // side, torch.arange(n, device=sim.device) % side
        by, bx = best_idx // side, best_idx % side
        dy = (by - ay.expand(b, n)).float() / side
        dx = (bx - ax.expand(b, n)).float() / side
        return torch.stack([best_val, top3.mean(dim=2), mutual, dx, dy,
                            (dx ** 2 + dy ** 2).sqrt()], dim=2)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        n = xa.size(0)
        tok = self._tokens(torch.cat([xa, xb]))
        feats = self.correspondence_features(tok[:n], tok[n:])
        x = self.encoder(torch.cat([self.cls.expand(n, -1, -1), self.embed(feats)], dim=1))
        return self.head(x[:, 0]).squeeze(-1)
