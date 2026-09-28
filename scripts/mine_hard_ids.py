"""Список путающихся машин для PK-сэмплера: кого с кем модель не различает.

Зачем. Разбор ошибок 24.09: выше правильного ответа стоят другие автомобили той же модели и того
же цвета, причём 91,5% из них сняты другими камерами — то есть это не артефакт ракурса, а обычная
неразличимость экземпляров. Триплет-лосс развёл бы их, если бы видел в одном батче, но PK-сэмплер
берёт P личностей случайно: при 16 личностях из 1232 два одинаковых каршеринговых Chery попадают
в батч примерно раз в сто эпох.

Здесь для каждой машины обучающих фолдов считается список ближайших к ней машин по центроиду
эмбеддинга. Сэмплер потом набирает батч вокруг одной машины и её соседей.

Утечки нет: берутся только кадры обучающих фолдов, val-фолд исключён так же, как в обучении.

    PYTHONPATH=. python scripts/mine_hard_ids.py --model artifacts/models/pool_cls_top16_f0 \\
        --val-fold 0 --out artifacts/hard_ids/f0.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from reid.data.crop_cache import cached_model_input, load_manifest
from reid.model import load, load_config


class Frames(Dataset):
    def __init__(self, rows, cache_dir, cfg):
        self.rows, self.cache_dir, self.cfg = rows, cache_dir, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return cached_model_input(self.cache_dir, self.rows[i], self.cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="любая обученная модель: нужен только порядок соседей")
    ap.add_argument("--val-fold", type=int, default=0, help="этот фолд исключается, как в обучении")
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--folds-path", default="splits/folds.csv")
    ap.add_argument("--neighbors", type=int, default=32, help="сколько соседей хранить на машину")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from types import SimpleNamespace

    man = load_manifest(args.cache, ("train",))
    folds = pd.read_csv(args.folds_path, dtype={"image_id": str})
    if args.val_fold is not None and args.val_fold >= 0:
        folds = folds[folds.fold != args.val_fold]
    df = man.merge(folds[["image_id", "vehicle_id"]], on="image_id", how="inner")
    rows = [SimpleNamespace(**r) for r in df.to_dict("records")]
    print(f"кадров {len(rows)}, машин {df.vehicle_id.nunique()}", flush=True)

    cfg = load_config(args.model)
    model = load(args.model, args.device)
    loader = DataLoader(Frames(rows, args.cache, cfg), batch_size=args.batch,
                        num_workers=args.workers)
    feats = []
    with torch.inference_mode():
        for x in loader:
            e = model.features(x.to(args.device))
            feats.append(torch.nn.functional.normalize(e, dim=1).float().cpu().numpy())
    emb = np.concatenate(feats)

    vids = np.array(sorted(df.vehicle_id.unique()))
    index = {v: i for i, v in enumerate(vids)}
    centroids = np.zeros((len(vids), emb.shape[1]), dtype=np.float32)
    for e, r in zip(emb, rows):
        centroids[index[r.vehicle_id]] += e
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    sim = centroids @ centroids.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)[:, : args.neighbors]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, vids=vids, neighbors=vids[order], model=np.array(args.model),
             val_fold=args.val_fold)
    near = sim[np.arange(len(vids)), order[:, 0]]
    print(f"сохранено {args.out}: сходство с ближайшей машиной в среднем {near.mean():.3f}, "
          f"у верхних 10% выше {np.quantile(near, 0.9):.3f}")


if __name__ == "__main__":
    main()
