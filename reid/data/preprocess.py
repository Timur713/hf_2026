"""Общая предобработка: crop по BBox и приведение кропа ко входу модели (контракт C3).

Эти функции вызывают И обучение (трек A), И инференс (трек B). Другой реализации crop/resize
быть не должно: любое расхождение с обучением бьёт по метрике. Менять — только с согласия обоих.

Все функции работают с torch-тензорами uint8 [3, H, W] на любом устройстве (CPU или CUDA).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 — пиксели, x1/y1 не включительно


def clip_bbox(x: float, y: float, w: float, h: float, frame_w: int, frame_h: int) -> Box:
    """BBox (x, y, w, h) → (x0, y0, x1, y1), обрезанный по кадру; не меньше 1×1 px."""
    x0 = min(max(int(round(x)), 0), frame_w - 1)
    y0 = min(max(int(round(y)), 0), frame_h - 1)
    x1 = min(max(int(round(x + w)), x0 + 1), frame_w)
    y1 = min(max(int(round(y + h)), y0 + 1), frame_h)
    return x0, y0, x1, y1


def expand_box(box: Box, pad: float, frame_w: int, frame_h: int) -> Box:
    """Расширяет бокс на долю pad от ширины/высоты с каждой стороны и обрезает по кадру."""
    if pad <= 0:
        return box
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (max(int(round(x0 - dx)), 0), max(int(round(y0 - dy)), 0),
            min(int(round(x1 + dx)), frame_w), min(int(round(y1 + dy)), frame_h))


def to_model_input(crop: torch.Tensor, input_size: Sequence[int], mean: Sequence[float],
                   std: Sequence[float], resize: str = "stretch") -> torch.Tensor:
    """uint8 [3, h, w] → float [3, H, W], нормализованный по mean/std.

    resize: stretch  — растянуть до H×W без сохранения пропорций;
            letterbox — вписать с сохранением пропорций, поля — нулём после нормализации.
    """
    H, W = int(input_size[0]), int(input_size[1])
    x = crop.unsqueeze(0).float().div_(255.0)
    if resize == "stretch":
        x = F.interpolate(x, size=(H, W), mode="bilinear", antialias=True, align_corners=False)
    elif resize == "letterbox":
        h, w = x.shape[-2:]
        s = min(H / h, W / w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)
        m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
        canvas = m.expand(1, 3, H, W).clone()  # после нормализации поля станут нулём
        top, left = (H - nh) // 2, (W - nw) // 2
        canvas[..., top:top + nh, left:left + nw] = x
        x = canvas
    else:
        raise ValueError(f"unknown resize policy: {resize}")
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return ((x - m) / s)[0]


def preprocess_frame(frame: torch.Tensor, bbox_xywh: Sequence[float], cfg: dict) -> torch.Tensor:
    """Полный путь инференса: кадр uint8 [3, H, W] + BBox → вход модели [3, h, w].

    cfg — поля model_config.yaml: input_size, bbox_pad, resize, mean, std.
    """
    fh, fw = frame.shape[-2:]
    box = expand_box(clip_bbox(*bbox_xywh, fw, fh), cfg.get("bbox_pad", 0.0), fw, fh)
    x0, y0, x1, y1 = box
    return to_model_input(frame[:, y0:y1, x0:x1], cfg["input_size"], cfg["mean"], cfg["std"],
                          cfg.get("resize", "stretch"))
