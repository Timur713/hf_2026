"""Парный верификатор: сравнивает два кропа и уточняет порядок top-10 (D-039).

Зачем: эмбеддер находит нужный кадр в десятке у 95% query, но не всегда ставит первым.
Верификатор смотрит на пару кадров целиком и поправляет порядок. Даёт +0,018 к mAP@10 (A-020).

Важно для бюджета: переранжирование **не входит** в замер `extract()` (D-028), только в общий
тайм-аут около 4 минут. Модель весит около 90 МБ.

Использование на инференсе:

    from reid.model.verifier import load_verifier, rerank_order
    ver, vcfg = load_verifier("artifacts/models/verifier_f0/verifier.pt", device)
    # crops: uint8 [3, H, W] — те же кропы, что идут в эмбеддер, но своим размером входа
    logits = ver.score_pairs(prep(query_crop, vcfg), torch.stack([prep(c, vcfg) for c in cand_crops]))
    order = rerank_order(cosine_top10, logits)          # индексы кандидатов в новом порядке
"""
from __future__ import annotations

import torch
import torch.nn as nn

from reid.data.preprocess import to_model_input
from reid.model import build_backbone

DEFAULT_WEIGHT = 0.2  # подобран на пяти фолдах (A-020): 0.1 и 0.5 хуже, 1.0 роняет метрику


class PairVerifier(nn.Module):
    """Сиамский backbone + слои кросс-внимания над токенами двух кадров → логит «та же машина»."""

    def __init__(self, arch="dinov2_vits14", input_size=(168, 168), layers=4, proj=256,
                 pretrained: bool = False, init_from: str | None = None):
        """pretrained=True нужен при обучении: замороженный случайный backbone учить не на чем.
        На инференсе остаётся False — веса приходят из чекпоинта, а интернета там нет.

        init_from — каталог нашего дообученного эмбеддера. Универсальная DINOv2 машин не видела:
        её патч-токены кодируют форму и текстуру, а не личность экземпляра, и голове приходится
        доставать из признаков то, чего в них нет. Дообученный backbone эту информацию уже несёт.
        Размер входа должен совпадать с тем, на котором эмбеддер учился, иначе не сойдутся
        позиционные эмбеддинги.
        """
        super().__init__()
        self.backbone = build_backbone(arch, input_size, pretrained=pretrained and not init_from)
        if init_from:
            self._load_backbone(init_from)
        dim = self.backbone.num_features
        self.proj = nn.Linear(dim, proj)
        self.segment = nn.Parameter(torch.zeros(2, 1, proj))
        self.cls = nn.Parameter(torch.zeros(1, 1, proj))
        nn.init.trunc_normal_(self.segment, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(proj, 8, proj * 4, 0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(proj), nn.Linear(proj, 1))

    def _load_backbone(self, model_dir: str) -> None:
        """Берёт веса backbone из артефакта эмбеддера, остальное (neck, классификатор) не нужно."""
        import os

        from safetensors.torch import load_file

        state = load_file(os.path.join(model_dir, "weights.safetensors"))
        prefix = "backbone."
        weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not weights:
            raise ValueError(f"{model_dir}: в весах нет ключей с префиксом {prefix!r}")
        missing, unexpected = self.backbone.load_state_dict(weights, strict=False)
        if unexpected:
            raise ValueError(f"{model_dir}: лишние ключи в backbone: {sorted(unexpected)[:5]}")
        # Голова классификации в backbone не входит, поэтому пропуски допустимы только там
        hard = [k for k in missing if not k.startswith(("head.", "fc."))]
        if hard:
            raise ValueError(f"{model_dir}: не хватает весов backbone: {sorted(hard)[:5]}")

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        n = xa.size(0)
        tok = self.backbone.forward_features(torch.cat([xa, xb]))
        a, b = self.proj(tok[:n]) + self.segment[0], self.proj(tok[n:]) + self.segment[1]
        x = self.encoder(torch.cat([self.cls.expand(n, -1, -1), a, b], dim=1))
        return self.head(x[:, 0]).squeeze(-1)

    @torch.inference_mode()
    def score_pairs(self, xq: torch.Tensor, xc: torch.Tensor, batch: int = 64) -> torch.Tensor:
        """Один query против K кандидатов: xq [3,H,W], xc [K,3,H,W] → логиты [K]."""
        out = []
        for i in range(0, xc.size(0), batch):
            chunk = xc[i:i + batch]
            out.append(self(xq.unsqueeze(0).expand(chunk.size(0), -1, -1, -1), chunk))
        return torch.cat(out)


def load_verifier(path: str, device: str = "cuda") -> tuple[PairVerifier, dict]:
    """Возвращает модель в режиме eval и конфиг предобработки её входа."""
    ck = torch.load(path, map_location="cpu")
    if ck.get("head") == "corr":
        from reid.model.corr_verifier import CorrVerifier
        model = CorrVerifier(ck["arch"], (ck["input"], ck["input"]))
    else:
        model = PairVerifier(ck["arch"], (ck["input"], ck["input"]))
    model.load_state_dict(ck["state"])
    cfg = {"input_size": [ck["input"], ck["input"]], "resize": "stretch",
           "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
           "weight": ck.get("weight", DEFAULT_WEIGHT)}
    return model.to(device).eval(), cfg


def prep_crop(crop: torch.Tensor, cfg: dict) -> torch.Tensor:
    """uint8 [3,h,w] → вход верификатора. Тот же кроп, что у эмбеддера, но свой размер."""
    return to_model_input(crop, cfg["input_size"], cfg["mean"], cfg["std"], cfg.get("resize", "stretch"))


def rerank_order(cosine: torch.Tensor, logits: torch.Tensor, weight: float = DEFAULT_WEIGHT):
    """Новый порядок кандидатов: cosine + weight · sigmoid(logit).

    Косинус остаётся в формуле намеренно: при weight=0 порядок в точности прежний, поэтому
    плохой верификатор не может испортить выдачу сильнее, чем на свой вес.
    """
    return torch.argsort(-(cosine + weight * torch.sigmoid(logits)))
