"""Итоговое число решения: фузия по всем фолдам, со штрафом за хабность и без.

Каждый фолд оценивается дампами моделей, обученных **без этого фолда**, — иначе получится утечка.
Имена дампов задаются шаблоном с `{k}`; фолд, для которого дампа нет, пропускается и об этом
говорится явно, чтобы среднее не считалось молча по неполному набору.

    PYTHONPATH=. python scripts/analysis/fusion_oof.py \\
        'artifacts/emb/div_dinov2_f{k}_cls_top16' 'artifacts/emb/div_clip_f{k}_cls_top16' \\
        'artifacts/emb/div_siglip_f{k}_top16'
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from reid.eval.eval_local import FoldTruth, ranking_metrics
from scripts.analysis.hub_penalty import parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help="шаблоны дампов с {k} вместо номера фолда")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for fold in [int(x) for x in args.folds.split(",")]:
        dumps = [p.format(k=fold) for p in args.patterns]
        missing = [d for d in dumps if not os.path.exists(f"{d}/fold{fold}.npz")]
        if missing:
            print(f"фолд {fold}: пропущен, нет дампов {', '.join(missing)}")
            continue
        truth = FoldTruth.load(fold)
        S, H, q_ids, g_ids = parts(dumps, fold, args.k)
        out = []
        for lam in (0.0, args.lam):
            order = np.argsort(-(S - lam * H[None, :]), axis=1)[:, :10]
            m = ranking_metrics(truth, {q: [g_ids[j] for j in order[i]]
                                        for i, q in enumerate(q_ids)})
            out.append(m)
        rows.append((fold, out[0], out[1]))
        print(f"фолд {fold}: без штрафа {out[0]['map10']:.4f}  со штрафом {out[1]['map10']:.4f}  "
              f"(R1 {out[1]['rank1']:.4f}, {out[1]['n_query_eval']} query)")

    if not rows:
        return
    plain = np.array([r[1]["map10"] for r in rows])
    hub = np.array([r[2]["map10"] for r in rows])
    print(f"\nфолдов в среднем: {len(rows)}")
    print(f"  без штрафа:  {plain.mean():.4f} ± {plain.std():.4f}")
    print(f"  со штрафом:  {hub.mean():.4f} ± {hub.std():.4f}   ({hub.mean() - plain.mean():+.4f})")
    if len(rows) < 5:
        print("  внимание: это не пятифолдовое число, в отчёт его подавать нельзя")


if __name__ == "__main__":
    main()
