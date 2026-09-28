"""Внешние ReID-датасеты для предобучения (задача A4.1).

Кадры там уже вырезаны по машине, поэтому BBox и кэш кропов не нужны: читаем файл, слегка
джиттерим рамку случайным кропом и приводим тем же `to_model_input`, что и основной путь.

VeRi-776: имя файла `0001_c001_00016450_0.jpg` → машина 0001, камера c001.
"""
from __future__ import annotations

import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from reid.data.preprocess import to_model_input

VERI_NAME = re.compile(r"^(\d+)_c(\d+)_")


def parse_list_file(path: str) -> dict[str, tuple[int, int]]:
    """VehicleID: рядом с кадрами лежит `train_list.txt` вида «имя_файла номер_машины».

    Меток камер в VehicleID нет, поэтому всем кадрам ставим камеру 0: кросс-камерного
    сэмплирования там не будет, а для предобучения важно число машин, а не камер.
    """
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0] + ".jpg"] = (int(parts[1]), 0)
    return out


def parse_veri(filename: str) -> tuple[int, int] | None:
    m = VERI_NAME.match(filename)
    return (int(m.group(1)), int(m.group(2))) if m else None


class ExternalReIDDataset(Dataset):
    """Плоский каталог кадров, где личность и камера закодированы в имени файла."""

    def __init__(self, root: str, cfg: dict, parser=parse_veri, aug=None, erasing=None,
                 crop_scale: tuple[float, float] = (0.85, 1.0), list_file: str | None = None):
        self.root, self.cfg, self.aug, self.erasing = root, cfg, aug, erasing
        self.crop_scale = crop_scale
        table = parse_list_file(list_file) if list_file else None
        items = []
        for name in sorted(os.listdir(root)):
            parsed = table.get(name) if table is not None else parser(name)
            if parsed:
                items.append((name, *parsed))
        vids = sorted({v for _, v, _ in items})
        self.vid_index = {v: i for i, v in enumerate(vids)}
        self.files = [n for n, _, _ in items]
        self.labels = [self.vid_index[v] for _, v, _ in items]
        self.cams = [c for _, _, c in items]

    def __len__(self) -> int:
        return len(self.files)

    @property
    def num_classes(self) -> int:
        return len(self.vid_index)

    def __getitem__(self, i: int):
        img = Image.open(os.path.join(self.root, self.files[i])).convert("RGB")
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
        if self.aug is not None:  # случайный кроп вместо джиттера BBox: рамки здесь уже нет
            _, h, w = x.shape
            s = random.uniform(*self.crop_scale)
            ch, cw = max(8, int(h * s)), max(8, int(w * s))
            top, left = random.randint(0, h - ch), random.randint(0, w - cw)
            x = x[:, top:top + ch, left:left + cw]
            x = self.aug(x)
        x = to_model_input(x, self.cfg["input_size"], self.cfg["mean"], self.cfg["std"],
                           self.cfg.get("resize", "stretch"))
        if self.erasing is not None:
            x = self.erasing(x)
        return x, self.labels[i], self.cams[i]
