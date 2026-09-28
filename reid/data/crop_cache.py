"""Кэш кропов: вырезанная по BBox с запасом область кадра, уменьшенная до 512 px по длинной стороне.

Нужен, чтобы обучение не декодировало полные кадры 1920×1080 (около 7 ГБ) на каждой эпохе.
В кэше лежит «контекстный» кроп: BBox, расширенный на CACHE_PAD с каждой стороны. Поэтому из кэша
можно получить вход модели с любым bbox_pad ≤ CACHE_PAD.

Отличие от пути инференса (preprocess_frame): лишнее уменьшение до 512 px и перекодирование JPEG q95.
Влияние на метрику проверяет трек B при сборке пайплайна (контракт C3, допуск Δ mAP@10 ≤ 0.002).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from reid.data.preprocess import clip_bbox, expand_box, to_model_input

CACHE_PAD = 0.15
CACHE_MAX_SIDE = 512
TRAIN_SPLITS = ("train",)


def load_manifest(cache_dir: str, splits: tuple[str, ...]) -> pd.DataFrame:
    """Строки манифеста нужных частей. Для обучения — только ("train",): тест не трогаем (AGENTS.md, п. 4)."""
    df = pd.read_csv(os.path.join(cache_dir, "manifest.csv"), dtype={"image_id": str})
    return df[df.split.isin(splits)].reset_index(drop=True)


def read_crop(cache_dir: str, image_id: str) -> torch.Tensor:
    """Кроп из кэша как uint8 [3, h, w]."""
    img = Image.open(os.path.join(cache_dir, "img", f"{image_id}.jpg")).convert("RGB")
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)


def cached_box(row, bbox_pad: float) -> tuple[int, int, int, int]:
    """Бокс (с bbox_pad) в пикселях кэшированного кропа."""
    assert bbox_pad <= CACHE_PAD, f"bbox_pad {bbox_pad} > CACHE_PAD {CACHE_PAD}: пересоберите кэш"
    box = expand_box(clip_bbox(row.x, row.y, row.w, row.h, row.frame_w, row.frame_h),
                     bbox_pad, row.frame_w, row.frame_h)
    s = row.scale
    x0, y0 = (box[0] - row.ctx_x0) * s, (box[1] - row.ctx_y0) * s
    x1, y1 = (box[2] - row.ctx_x0) * s, (box[3] - row.ctx_y0) * s
    return int(round(x0)), int(round(y0)), max(int(round(x1)), int(round(x0)) + 1), \
        max(int(round(y1)), int(round(y0)) + 1)


def cached_model_input(cache_dir: str, row, cfg: dict) -> torch.Tensor:
    """Вход модели из кэша — тот же to_model_input, что и на инференсе."""
    crop = read_crop(cache_dir, row.image_id)
    x0, y0, x1, y1 = cached_box(row, cfg.get("bbox_pad", 0.0))
    return to_model_input(crop[:, y0:y1, x0:x1], cfg["input_size"], cfg["mean"], cfg["std"],
                          cfg.get("resize", "stretch"))
