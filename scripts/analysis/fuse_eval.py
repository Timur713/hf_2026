"""Фузия произвольного числа моделей через стандартизованные сходства (A-052).

Складывать косинусы напрямую нельзя: у разных моделей они в разных масштабах, и порядок определяет
модель с наибольшим разбросом. Поэтому сходства каждой модели стандартизуются по строке — по одному
query относительно всей gallery, — и только потом складываются. Используются только сам query и
статическая gallery, независимость запросов соблюдена (правило 1).

    PYTHONPATH=. python scripts/analysis/fuse_eval.py --fold 0 \
        artifacts/emb/final_oof_f0_cls_avg artifacts/emb/siglip2_f0_full_avg
"""
import argparse
import itertools

import numpy as np

from reid.eval.eval_local import evaluate


def sims(path: str, fold: int):
    d = np.load(f"{path}/fold{fold}.npz")
    n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    s = n(d["query_emb"]) @ n(d["gallery_emb"]).T
    return s, d["query_ids"].tolist(), d["gallery_ids"].tolist()


def zrows(s):
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)


def score(s, g_ids, q_ids, fold):
    order = np.argsort(-s, axis=1)[:, :60]
    return evaluate(fold, {q: [g_ids[j] for j in order[i]] for i, q in enumerate(q_ids)})["map10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max-combo", type=int, default=0, help="перебрать все сочетания до такого размера")
    args = ap.parse_args()

    z, names = [], []
    q_ids = g_ids = None
    for p in args.dumps:
        s, q, g = sims(p, args.fold)
        if q_ids is None:
            q_ids, g_ids = q, g
        assert q == q_ids and g == g_ids, f"{p}: другой порядок query/gallery"
        z.append(zrows(s))
        names.append(p.rstrip("/").split("/")[-1])
        print(f"{names[-1]:38s} одна модель {score(s, g_ids, q_ids, args.fold):.4f}", flush=True)

    print()
    if args.max_combo:
        best = []
        for r in range(2, min(args.max_combo, len(z)) + 1):
            for combo in itertools.combinations(range(len(z)), r):
                v = score(sum(z[i] for i in combo), g_ids, q_ids, args.fold)
                best.append((v, combo))
        best.sort(reverse=True)
        for v, combo in best[:8]:
            print(f"{v:.4f}  " + " + ".join(names[i] for i in combo))
    else:
        v = score(sum(z), g_ids, q_ids, args.fold)
        print(f"фузия всех {len(z)}: {v:.4f}")


if __name__ == "__main__":
    main()
