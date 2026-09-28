"""Запоминает верификатор личности или учится сравнивать (A-094).

Контроль обучаемости показал loss 0.0010 на 96 фиксированных парах: ёмкости хватает.
Тогда слабость на валидации — это не «архитектура не считает соответствие» (вывод A-061),
а разрыв обобщения. Здесь он измеряется прямо: AUC на парах фолдов, которые верификатор
видел при обучении, против AUC на отложенном фолде.

Большой разрыв означает, что модель выучила конкретные машины, и лечится это числом
личностей, а не сменой головы.

    PYTHONPATH=. python scripts/analysis/verifier_gap.py \
        --verifier artifacts/models/v_fus_f1only/verifier.pt --train-folds 1 --val-fold 0
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from reid.data.augment import TrainAugment
from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.data.preprocess import to_model_input
from reid.model.verifier import PairVerifier as VerifierNetwork


def sample_pairs(pairs_dir, fold, n, seed=2026):
    """n позитивов и n негативов из добытых пар одного фолда."""
    d = np.load(os.path.join(pairs_dir, f"fold{fold}.npz"))
    ids, vids, cams = d["image_ids"].tolist(), d["vehicle_id"].tolist(), d["camera_id"].tolist()
    neg = [[j for j in row if j >= 0] for row in d["negatives"]]
    by = {}
    for i, v in enumerate(vids):
        by.setdefault(v, []).append(i)
    rng = random.Random(seed)
    out = []
    for want_pos in (True, False):
        got = 0
        while got < n:
            i = rng.randrange(len(ids))
            pool = [j for j in by[vids[i]] if j != i and cams[j] != cams[i]] if want_pos else neg[i]
            if not pool:
                continue
            out.append((ids[i], ids[rng.choice(pool)], 1.0 if want_pos else 0.0))
            got += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--train-folds", default="1")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--pairs", default="artifacts/pairs")
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--n", type=int, default=600, help="пар каждого знака на фолд")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    payload = torch.load(args.verifier, map_location="cpu", weights_only=True)
    side = int(payload["input"])
    model = VerifierNetwork(str(payload["arch"]), (side, side), pretrained=False)
    model.load_state_dict(payload["state"], strict=True)
    model.to(args.device).eval()
    print(f"верификатор: {payload['arch']} вход {side}, обучен на {payload.get('trained_on','?')}")
    cfg = {"input_size": (side, side), "mean": [0.485, 0.456, 0.406],
           "std": [0.229, 0.224, 0.225]}
    man = load_manifest(args.cache, ("train",)).set_index("image_id")
    aug = TrainAugment(0.0, 0.0, flip=0.0, color=0.0)      # без аугментации: меряем модель, не шум

    def prep(image_id):
        row = man.loc[image_id]
        crop = read_crop(args.cache, image_id)
        x0, y0, x1, y1 = cached_box(row, 0.0)
        return to_model_input(crop[:, y0:y1, x0:x1], cfg["input_size"], cfg["mean"], cfg["std"])

    def auc_of(fold):
        pairs = sample_pairs(args.pairs, fold, args.n)
        scores, labels = [], []
        for s in range(0, len(pairs), args.batch):
            chunk = pairs[s:s + args.batch]
            a = torch.stack([prep(p[0]) for p in chunk]).to(args.device)
            b = torch.stack([prep(p[1]) for p in chunk]).to(args.device)
            with torch.inference_mode():
                scores += model(a, b).float().cpu().tolist()
            labels += [p[2] for p in chunk]
        s, y = np.array(scores), np.array(labels)
        P, N = s[y == 1], s[y == 0]
        return float((P[:, None] > N[None, :]).mean()), len(P), len(N)

    for fold in [int(k) for k in args.train_folds.split(",")]:
        a, np_, nn = auc_of(fold)
        print(f"fold{fold} (ВИДЕЛ при обучении): AUC {a:.3f}  позитивов {np_}, негативов {nn}")
    a, np_, nn = auc_of(args.val_fold)
    print(f"fold{args.val_fold} (НЕ видел):          AUC {a:.3f}  позитивов {np_}, негативов {nn}")


if __name__ == "__main__":
    main()
