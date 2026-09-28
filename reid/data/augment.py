"""Аугментации обучения трека A.

Работают поверх кэша кропов: джиттер бокса имитирует неточность BBox, остальное — стандартный
для ReID набор. Цель — чтобы модель училась по самой машине, а не по фону и не по экспозиции
конкретной камеры (см. docs/findings.md §6).
"""
from __future__ import annotations

import random

import torch
from torchvision.transforms import v2

from reid.data.crop_cache import CACHE_PAD


class TrainAugment:
    """uint8-кроп [3, h, w] → uint8-кроп после геометрии и цвета. Erasing применяется позже."""

    def __init__(self, box_jitter: float = 0.05, pad_jitter: float = 0.05, flip: float = 0.5,
                 color: float = 0.2, plate_band_p: float = 0.0):
        self.box_jitter, self.pad_jitter, self.flip = box_jitter, pad_jitter, flip
        self.plate_band_p = plate_band_p
        self.color = v2.ColorJitter(brightness=color, contrast=color, saturation=color) if color else None

    def jitter_box(self, box: tuple[int, int, int, int], shape: tuple[int, int], bbox_pad: float):
        """Случайно сдвигает и расширяет бокс внутри кэшированного контекста."""
        h, w = shape
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        pad = random.uniform(0.0, self.pad_jitter) + bbox_pad
        dx, dy = random.uniform(-1, 1) * self.box_jitter * bw, random.uniform(-1, 1) * self.box_jitter * bh
        x0 = min(max(x0 - pad * bw + dx, 0), w - 2)
        y0 = min(max(y0 - pad * bh + dy, 0), h - 2)
        x1 = min(max(x1 + pad * bw + dx, x0 + 1), w)
        y1 = min(max(y1 + pad * bh + dy, y0 + 1), h)
        return int(x0), int(y0), int(x1), int(y1)

    def plate_band(self, crop: torch.Tensor) -> torch.Tensor:
        """Заливает прямоугольник номерной формы в нижне-центральной зоне (D-027).

        Точные координаты номера неизвестны и не нужны: задача — отучить модель опираться
        на эту область, а не найти её.
        """
        _, h, w = crop.shape
        bw = int(w * random.uniform(0.12, 0.28))
        bh = max(3, int(bw / random.uniform(2.2, 5.0)))
        x0 = random.randint(int(w * 0.15), max(int(w * 0.15), w - bw - int(w * 0.15)))
        y0 = random.randint(int(h * 0.35), max(int(h * 0.35), h - bh - 1))
        crop = crop.clone()
        crop[:, y0:y0 + bh, x0:x0 + bw] = random.randint(0, 255)
        return crop

    def __call__(self, crop: torch.Tensor) -> torch.Tensor:
        if random.random() < self.flip:
            crop = torch.flip(crop, dims=[-1])
        if self.color is not None:
            crop = self.color(crop)
        if random.random() < self.plate_band_p:
            crop = self.plate_band(crop)
        return crop


class RandomErasing:
    """Стирает случайный прямоугольник уже нормализованного тензора (заполнение нулём = среднее)."""

    def __init__(self, p: float = 0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)):
        self.p, self.scale, self.ratio = p, scale, ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x
        _, H, W = x.shape
        for _ in range(10):
            area = H * W * random.uniform(*self.scale)
            ar = random.uniform(*self.ratio)
            h, w = int(round((area * ar) ** 0.5)), int(round((area / ar) ** 0.5))
            if h < H and w < W:
                top, left = random.randint(0, H - h), random.randint(0, W - w)
                x[:, top:top + h, left:left + w] = 0.0
                return x
        return x


def check_pad_budget(bbox_pad: float, pad_jitter: float) -> None:
    assert bbox_pad + pad_jitter <= CACHE_PAD, (
        f"bbox_pad + pad_jitter = {bbox_pad + pad_jitter} > CACHE_PAD = {CACHE_PAD}: "
        "кэш не содержит столько контекста, пересоберите его")
