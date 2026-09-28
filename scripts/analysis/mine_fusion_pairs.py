"""Трудные пары из ошибок ФУЗИИ, а не одиночной модели.

Прежний верификатор обучался на негативах, добытых одиночной OOF-моделью (D-032). Поверх фузии он
почти не помогал (+0.003, A-054) — но он и не видел её ошибок: фузия путается на других парах.
Здесь негативы берутся прямо из фузионного ранжирования того же фолда.

Модель гонять не нужно: сходства уже лежат в дампах. Формат выхода совпадает с `mine_pairs.py`,
чтобы `train_verifier.py` читал его без изменений.

    PYTHONPATH=. python scripts/analysis/mine_fusion_pairs.py --fold 1 --out artifacts/pairs_fusion <дампы>
"""
import argparse
import os

import numpy as np

from reid.eval.eval_local import FoldTruth


def zrows(s):
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--neg-k", type=int, default=20)
    ap.add_argument("--out", default="artifacts/pairs_fusion")
    args = ap.parse_args()

    truth = FoldTruth.load(args.fold)
    # Негативы ищем среди ВСЕХ кадров фолда, а не только gallery: верификатор учится на парах
    # кадров, и ему всё равно, какую роль кадр играл в оценке.
    Z, ids = None, None
    for p in args.dumps:
        d = np.load(f"{p}/fold{args.fold}.npz")
        n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        emb = np.concatenate([n(d["query_emb"]), n(d["gallery_emb"])])
        z = zrows(emb @ emb.T)
        Z = z if Z is None else Z + z
        ids = d["query_ids"].tolist() + d["gallery_ids"].tolist()

    vid = np.array([truth.vid[i] for i in ids])
    cam = np.array([truth.cam[i] for i in ids])
    np.fill_diagonal(Z, -1e9)

    negatives = np.full((len(ids), args.neg_k), -1, dtype=np.int32)
    same = vid[None, :] == vid[:, None]
    for i in range(len(ids)):
        cand = np.argsort(-Z[i])
        hard = [j for j in cand if not same[i, j]][: args.neg_k]
        negatives[i, : len(hard)] = hard

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, f"fold{args.fold}.npz"),
             image_ids=np.array(ids), vehicle_id=vid, camera_id=cam, negatives=negatives)

    # Насколько эти негативы отличаются от прежних — если пересечение велико, смысла мало
    old = os.path.join("artifacts/pairs", f"fold{args.fold}.npz")
    msg = ""
    if os.path.exists(old):
        o = np.load(old)
        pos = {i: set(r[r >= 0].tolist()) for i, r in enumerate(o["negatives"])}
        idx = {x: k for k, x in enumerate(o["image_ids"].tolist())}
        share = []
        for i, image_id in enumerate(ids):
            if image_id in idx:
                new = set(negatives[i][negatives[i] >= 0].tolist())
                old_ids = {idx.get(ids[j], -1) for j in new}
                share.append(len(pos[idx[image_id]] & old_ids) / max(len(new), 1))
        msg = f", пересечение с прежними негативами {np.mean(share) * 100:.0f}%"
    print(f"фолд {args.fold}: {len(ids)} кадров, по {args.neg_k} трудных негативов{msg}")


if __name__ == "__main__":
    main()
