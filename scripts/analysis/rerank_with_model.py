"""Вторая ступень: модель считает эмбеддинги только для окна кандидатов и меняет порядок.

По ТЗ §7 переранжирование не входит в замер скорости, поэтому на этой ступени можно позволить
дорогую модель — высокое разрешение или большой backbone (D-056). Оценивать её надо ровно так,
как она будет работать в сдаче:

  * окно кандидатов даёт дешёвая фузия ступени 1 вместе со штрафом за хабность;
  * дорогая модель считает сходства **только внутри окна**;
  * обе оценки стандартизуются **по окну** и складываются (D-053) — не по всей галерее, иначе
    число окажется не тем, что получится на инференсе.

Отличие от `fusion_add.py`: там кандидат становится полноправным участником поиска и влияет на
состав top-K, здесь он только переставляет уже найденное.

    PYTHONPATH=. python scripts/analysis/rerank_with_model.py --fold 0 \\
        --base artifacts/emb/pool_cls_top16_f0_cls_top16 artifacts/emb/div_clip_f0_cls_top16 \\
               artifacts/emb/div_siglip_f0_top16 \\
        --rerank artifacts/emb/hires392_f0_cls_top16
"""
from __future__ import annotations

import argparse

import numpy as np

from reid.eval.eval_local import FoldTruth, ranking_metrics
from scripts.analysis.hub_penalty import parts


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--rerank", nargs="+", required=True, help="дампы дорогой модели (или нескольких)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.3, help="штраф за хабность на ступени 1")
    ap.add_argument("--window", nargs="+", type=int, default=[10, 20, 30])
    ap.add_argument("--weights", nargs="+", type=float, default=[0.2, 0.3, 0.5, 0.8, 1.0, 1.5])
    args = ap.parse_args()

    truth = FoldTruth.load(args.fold)
    S, H, q_ids, g_ids = parts(args.base, args.fold, 3)
    base = S - args.lam * H[None, :]

    # Сходства моделей второй ступени складываются ТОЛЬКО после выравнивания шкал (D-052):
    # у разных моделей разброс косинусов отличается в разы, и сумма сырых величин — это
    # переодетая одна модель, а не фузия.
    R = None
    for path in args.rerank:
        d = np.load(f"{path}/fold{args.fold}.npz")
        assert d["query_ids"].tolist() == q_ids and d["gallery_ids"].tolist() == g_ids
        s = unit(d["query_emb"]) @ unit(d["gallery_emb"]).T
        s = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)
        R = s if R is None else R + s

    def score(ranking):
        return ranking_metrics(truth, ranking)["map10"]

    order = np.argsort(-base, axis=1)
    print(f"фолд {args.fold}, ступень 1 из {len(args.base)} модел. со штрафом λ={args.lam}, "
          f"ступень 2 из {len(args.rerank)} модел.")
    b = score({q: [g_ids[j] for j in order[i][:10]] for i, q in enumerate(q_ids)})
    print(f"без второй ступени: {b:.4f}\n")
    print("     w " + "".join(f"{w:>9}" for w in args.weights))
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    for k in args.window:
        row = []
        for w in args.weights:
            ranked = {}
            for i, q in enumerate(q_ids):
                c = order[i][:k]
                mix = z(base[i, c]) + w * z(R[i, c])
                ranked[q] = [g_ids[c[j]] for j in np.argsort(-mix)][:10]
            row.append(score(ranked))
        best = max(row)
        print(f"окно {k:<3}" + "".join(f"{v:>9.4f}" for v in row) + f"   лучшее {best - b:+.4f}")


if __name__ == "__main__":
    main()
