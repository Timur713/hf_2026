"""Датасеты обучения и валидации поверх кэша кропов (контракт C1, C3).

Обучение видит только split == "train" из манифеста: тестовые кадры для обучения запрещены
(AGENTS.md, жёсткое правило 4). Валидация идёт по фолду: query и gallery из splits/.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from reid.data.augment import RandomErasing, TrainAugment, check_pad_budget
from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.data.preprocess import to_model_input


def _rows(df: pd.DataFrame) -> list[SimpleNamespace]:
    # SimpleNamespace, а не itertuples: строки должны пересылаться в воркеры DataLoader (spawn).
    return [SimpleNamespace(**r) for r in df.to_dict("records")]


class TrainDataset(Dataset):
    """Кадры обучающих фолдов. Возвращает (вход модели, индекс личности, camera_id)."""

    def __init__(self, cache_dir: str, folds_path: str, val_fold: int | None, cfg: dict,
                 aug: TrainAugment | None = None, erasing: RandomErasing | None = None):
        man = load_manifest(cache_dir, ("train",))
        folds = pd.read_csv(folds_path, dtype={"image_id": str})
        if val_fold is not None:
            folds = folds[folds.fold != val_fold]
        df = man.merge(folds[["image_id", "vehicle_id", "camera_id"]], on="image_id", how="inner")
        self.vids = sorted(df.vehicle_id.unique())
        self.vid_index = {v: i for i, v in enumerate(self.vids)}
        self.rows = _rows(df)
        self.labels = [self.vid_index[r.vehicle_id] for r in self.rows]
        self.cams = [int(r.camera_id) for r in self.rows]
        # Приближение ракурса: анфас и корма узкие, борт широкий. Нужно сэмплеру, чтобы кадры
        # одной машины в батче не оказались все с одного ракурса.
        self.views = [float(np.log(max(r.w, 1) / max(r.h, 1))) for r in self.rows]
        self.cache_dir, self.cfg = cache_dir, cfg
        self.aug, self.erasing = aug, erasing
        if aug is not None:
            check_pad_budget(cfg.get("bbox_pad", 0.0), aug.pad_jitter)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        crop = read_crop(self.cache_dir, row.image_id)
        if self.aug is None:
            x0, y0, x1, y1 = cached_box(row, self.cfg.get("bbox_pad", 0.0))
        else:
            x0, y0, x1, y1 = self.aug.jitter_box(cached_box(row, 0.0), crop.shape[-2:],
                                                 self.cfg.get("bbox_pad", 0.0))
        crop = crop[:, y0:y1, x0:x1]
        if self.aug is not None:
            crop = self.aug(crop)
        x = to_model_input(crop, self.cfg["input_size"], self.cfg["mean"], self.cfg["std"],
                           self.cfg.get("resize", "stretch"))
        if self.erasing is not None:
            x = self.erasing(x)
        return x, self.labels[i], self.cams[i]

    @property
    def num_classes(self) -> int:
        return len(self.vids)


class EvalDataset(Dataset):
    """Кадры одной части фолда (query или gallery) без аугментаций, в порядке файла сплита."""

    def __init__(self, cache_dir: str, split_csv: str, cfg: dict):
        man = load_manifest(cache_dir, ("train", "query", "gallery"))
        ids = pd.read_csv(split_csv, dtype={"image_id": str}).image_id.tolist()
        df = man.set_index("image_id").loc[ids].reset_index()
        self.rows, self.ids, self.cache_dir, self.cfg = _rows(df), ids, cache_dir, cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        row = self.rows[i]
        x0, y0, x1, y1 = cached_box(row, self.cfg.get("bbox_pad", 0.0))
        return to_model_input(read_crop(self.cache_dir, row.image_id)[:, y0:y1, x0:x1],
                              self.cfg["input_size"], self.cfg["mean"], self.cfg["std"],
                              self.cfg.get("resize", "stretch"))
