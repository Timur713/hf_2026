"""Стоит ли новая модель места в фузии: одна команда, четыре числа.

Новый участник оценивается не одним числом, а четырьмя, потому что сильная одиночная модель может
не добавить фузии ничего (A-055), а слабая — испортить её (A-056):

  * сама по себе;
  * добавленная к текущему составу;
  * заменяющая каждого из нынешних участников;
  * всё то же со штрафом за хабность галереи (D-055), потому что в сдачу идёт именно он.

    PYTHONPATH=. python scripts/analysis/fusion_add.py --fold 0 \\
        --base artifacts/emb/pool_cls_top16_f0_cls_top16 artifacts/emb/div_clip_f0_cls_top16 \\
               artifacts/emb/div_siglip_f0_top16 \\
        --new artifacts/emb/hires392_f0_cls_top16
"""
from __future__ import annotations

import argparse

import numpy as np

from reid.eval.eval_local import FoldTruth, ranking_metrics
from scripts.analysis.hub_penalty import parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True, help="нынешний состав фузии")
    ap.add_argument("--new", required=True, help="дамп кандидата")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.3, help="вес штрафа за хабность")
    ap.add_argument("--k", type=int, default=3, help="соседей в хабности")
    args = ap.parse_args()

    truth = FoldTruth.load(args.fold)
    short = lambda p: p.rstrip("/").split("/")[-1][:30]

    def score(dumps):
        S, H, q_ids, g_ids = parts(dumps, args.fold, args.k)
        out = []
        for lam in (0.0, args.lam):
            order = np.argsort(-(S - lam * H[None, :]), axis=1)[:, :10]
            out.append(ranking_metrics(truth, {q: [g_ids[j] for j in order[i]]
                                               for i, q in enumerate(q_ids)})["map10"])
        return out

    base = score(args.base)
    print(f"фолд {args.fold}, штраф λ={args.lam}\n")
    print(f"{'состав':46s} {'без штрафа':>11s} {'со штрафом':>11s}")
    print(f"{'нынешняя фузия':46s} {base[0]:11.4f} {base[1]:11.4f}")
    for p in args.base:
        v = score([p])
        print(f"  {'из неё ' + short(p):44s} {v[0]:11.4f} {v[1]:11.4f}")
    print()
    v = score([args.new])
    print(f"{'кандидат ' + short(args.new) + ' один':46s} {v[0]:11.4f} {v[1]:11.4f}")
    v = score(list(args.base) + [args.new])
    print(f"{'фузия + кандидат':46s} {v[0]:11.4f} {v[1]:11.4f}   "
          f"({v[1] - base[1]:+.4f})")
    for i, p in enumerate(args.base):
        dumps = [d for j, d in enumerate(args.base) if j != i] + [args.new]
        v = score(dumps)
        print(f"{'кандидат вместо ' + short(p):46s} {v[0]:11.4f} {v[1]:11.4f}   "
              f"({v[1] - base[1]:+.4f})")


if __name__ == "__main__":
    main()
