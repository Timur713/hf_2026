"""Добывает трудные негативы OOF-моделями (D-032, шаг 1 к задаче A3.3).

Для кадров фолда j негативы ищет модель, обученная БЕЗ фолда j. Тогда это её настоящие ошибки,
а не примеры, которые она запомнила. Именно на этом провалилась первая версия верификатора (A-009).

    PYTHONPATH=. python scripts/mine_pairs.py --model-pattern 'artifacts/models/vitb14_oof_f{k}' --neg-k 20
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from reid.data.crop_cache import cached_model_input, load_manifest
from reid.model import load, load_config


class FoldImages(Dataset):
    def __init__(self, cache, rows, cfg):
        from types import SimpleNamespace
        self.rows = [SimpleNamespace(**r) for r in rows.to_dict("records")]
        self.cache, self.cfg = cache, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return cached_model_input(self.cache, self.rows[i], self.cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pattern", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--neg-k", type=int, default=20)
    ap.add_argument("--out", default="artifacts/pairs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    folds = pd.read_csv("splits/folds.csv", dtype={"image_id": str})
    man = load_manifest(args.cache, ("train",))
    for k in map(int, args.folds.split(",")):
        model_dir = args.model_pattern.format(k=k)
        cfg, model = load_config(model_dir), load(model_dir, device)
        part = folds[folds.fold == k]
        rows = man.merge(part[["image_id", "vehicle_id", "camera_id"]], on="image_id")
        loader = DataLoader(FoldImages(args.cache, rows, cfg), batch_size=64, num_workers=6)
        with torch.inference_mode():
            emb = torch.cat([model(x.to(device)).float().cpu() for x in loader])
        vid = rows.vehicle_id.to_numpy()
        sim = (emb @ emb.T).numpy()
        np.fill_diagonal(sim, -2)
        order = np.argsort(-sim, axis=1)[:, : args.neg_k * 4]
        negatives = np.full((len(rows), args.neg_k), -1, dtype=np.int32)
        for i, row in enumerate(order):
            hard = [j for j in row if vid[j] != vid[i]][: args.neg_k]
            negatives[i, : len(hard)] = hard
        np.savez(os.path.join(args.out, f"fold{k}.npz"),
                 image_ids=rows.image_id.to_numpy().astype("U40"),  # не object: иначе нужен allow_pickle
                 vehicle_id=vid, camera_id=rows.camera_id.to_numpy(), negatives=negatives)
        share = float((sim[np.arange(len(rows)), order[:, 0]] > 0.5).mean())
        print(f"фолд {k}: {len(rows)} кадров, негативы от модели без этого фолда; "
              f"доля с очень близким чужим кадром {share:.2f}", flush=True)


if __name__ == "__main__":
    main()
