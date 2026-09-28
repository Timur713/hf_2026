"""VehicleID как источник пар для предобучения верификатора.

Зачем. Наш верификатор почти случаен: один, без косинуса, он даёт mAP@10 0,13 и ставит позитив на
11-е место из 30 (A-064). Четыре правки рецепта подряд не сдвинули его ни на сотую, потому что
упор не в рецепте: 7,6 тыс. кадров и 1,2 тыс. машин — это нечего есть для задачи «тот же
экземпляр под другим ракурсом». VehicleID даёт 113 346 кадров и 13 164 машины, он публичный и
разрешён правилами (D-043 уже использовал его для эмбеддера).

Главное — не объём, а **какие негативы**. Разбор ошибок показал: мы сыпемся на машинах одной
модели и одного цвета, в основном каршеринговых, в одинаковой оклейке. В VehicleID есть разметка
модели и цвета по машинам, поэтому негатив можно брать не случайный, а из той же модели и того же
цвета — ровно та пара, которую верификатор обязан научиться различать.

Файлы (ужатый набор, `scripts/repack_vehicleid.py`):
    <root>/image/<name>.jpg        кадры, длинная сторона 320
    <root>/train_list.txt          "<name> <vehicle_id>"
    <root>/model_attr.txt          "<vehicle_id> <model_id>"   (необязательно)
    <root>/color_attr.txt          "<vehicle_id> <color_id>"   (необязательно)
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image

from reid.data.preprocess import to_model_input


def _read_pairs(path: str) -> dict[str, int]:
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    return out


class VehicleIDPairs(torch.utils.data.Dataset):
    """Пары или списки кандидатов из VehicleID. Интерфейс тот же, что у PairDataset.

    mode="bce"      → (xa, xb, label)
    mode="listwise" → (xq, [K кадров], индекс позитива)
    """

    def __init__(self, root: str, cfg: dict, aug, mode: str = "bce", k: int = 16,
                 seed: int = 2026, length: int = 100000, hard_ratio: float = 0.8,
                 curriculum: bool = False):
        """curriculum — наращивать долю трудных негативов с 0,2 до hard_ratio по ходу обучения.

        Нужно, потому что негатив «та же модель, тот же цвет» — предельный случай: две одинаковые
        машины, различимые почти только номером. Если подавать их с первого шага, голова стартует
        сразу на неразрешимом и loss стоит у ln 2 — ровно это наблюдалось на первом прогоне.
        """
        self.root, self.cfg, self.aug, self.mode, self.k = root, cfg, aug, mode, k
        self.seed, self.length, self.hard_ratio = seed, length, hard_ratio
        self.curriculum = curriculum

        img2vid = _read_pairs(os.path.join(root, "train_list.txt"))
        self.names = sorted(img2vid)
        self.vid = np.array([img2vid[n] for n in self.names])
        by_vid: dict[int, list[int]] = {}
        for i, v in enumerate(self.vid):
            by_vid.setdefault(int(v), []).append(i)
        # Кадр без второго кадра той же машины не годится ни в позитив, ни в якорь списка
        self.by_vid = {v: idx for v, idx in by_vid.items() if len(idx) > 1}
        self.anchors = [i for v in self.by_vid for i in self.by_vid[v]]

        # Группы «та же модель и тот же цвет» — отсюда берутся трудные негативы
        model = _read_pairs(os.path.join(root, "model_attr.txt")) \
            if os.path.exists(os.path.join(root, "model_attr.txt")) else {}
        color = _read_pairs(os.path.join(root, "color_attr.txt")) \
            if os.path.exists(os.path.join(root, "color_attr.txt")) else {}
        self.group: dict[tuple[int, int], list[int]] = {}
        self.vid_group: dict[int, tuple[int, int]] = {}
        for v in self.by_vid:
            key = (model.get(str(v), -1), color.get(str(v), -1))
            if key == (-1, -1):
                continue
            self.vid_group[v] = key
            self.group.setdefault(key, []).append(v)
        # Группа из одной машины трудных негативов не даёт
        self.group = {k_: v for k_, v in self.group.items() if len(v) > 1}
        self.vid_group = {v: g for v, g in self.vid_group.items() if g in self.group}

    def __len__(self):
        return self.length

    def stats(self) -> str:
        cov = len(self.vid_group) / max(len(self.by_vid), 1)
        return (f"VehicleID: {len(self.names)} кадров, {len(self.by_vid)} машин с парой, "
                f"групп «модель+цвет» {len(self.group)}, покрытие разметкой {cov:.1%}")

    def _negative(self, i: int, rng: random.Random, progress: float = 1.0) -> int:
        """Трудный негатив — другая машина той же модели и того же цвета."""
        v = int(self.vid[i])
        share = (0.2 + (self.hard_ratio - 0.2) * progress) if self.curriculum else self.hard_ratio
        group = self.group.get(self.vid_group.get(v, None), None)
        if group and rng.random() < share:
            for _ in range(8):
                other = rng.choice(group)
                if other != v:
                    return rng.choice(self.by_vid[other])
        j = rng.randrange(len(self.names))
        while self.vid[j] == self.vid[i]:
            j = rng.randrange(len(self.names))
        return j

    def image(self, i: int, rng: random.Random) -> torch.Tensor:
        img = Image.open(os.path.join(self.root, "image", f"{self.names[i]}.jpg")).convert("RGB")
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
        random.setstate(rng.getstate())        # аугментации детерминированы по индексу примера
        return to_model_input(self.aug(x), self.cfg["input_size"], self.cfg["mean"],
                              self.cfg["std"])

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed * 7919 + idx)
        # Загрузчик идёт по индексам подряд, поэтому доля пройденного обучения = idx / length
        progress = idx / max(self.length - 1, 1)
        i = self.anchors[rng.randrange(len(self.anchors))]
        pos_pool = [j for j in self.by_vid[int(self.vid[i])] if j != i]
        if self.mode == "listwise":
            cand = [rng.choice(pos_pool)] + [self._negative(i, rng, progress) for _ in range(self.k - 1)]
            target = rng.randrange(len(cand))
            cand[0], cand[target] = cand[target], cand[0]
            return self.image(i, rng), torch.stack([self.image(j, rng) for j in cand]), target
        want_pos = idx % 2 == 0
        j = rng.choice(pos_pool) if want_pos else self._negative(i, rng, progress)
        return self.image(i, rng), self.image(j, rng), float(want_pos)
