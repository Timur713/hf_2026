"""Калибровка порога отказа по пяти фолдам (вход для задачи B4.1).

Решение в режиме кандидатов сводится к одному вопросу: есть ли машина query в gallery. Двойника
с той же камеры прятать не нужно — эталонный evaluate.py засчитывает его как верный ответ (D-023).

Порог берём не там, где максимум на одном фолде, а в середине устойчивой области по всем пяти:
одиночный максимум почти всегда оказывается случайным.

    PYTHONPATH=. python scripts/calibrate_threshold.py --emb-pattern 'artifacts/emb/vitb14_oof_f{k}_cls_avg'
"""
import argparse

import numpy as np

from reid.eval.eval_local import evaluate, ranking_from_embeddings


def fold_curve(pattern: str, fold: int, grid: np.ndarray):
    d = np.load(pattern.format(k=fold) + f"/fold{fold}.npz")
    q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
    ranking = ranking_from_embeddings(d["query_emb"], d["gallery_emb"], g_ids, q_ids)
    top = (d["query_emb"] @ d["gallery_emb"].T).max(axis=1)
    out = []
    for thr in grid:
        cands = [(q, ranking[q][0], float(top[i])) for i, q in enumerate(q_ids) if top[i] >= thr]
        m = evaluate(fold, ranking, cands)["official"]
        out.append((0.7 * m["f1"] + 0.3 * m["tnr"], m["f1"], m["tnr"]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-pattern", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    grid = np.linspace(args.lo, args.hi, args.steps)
    curves = np.stack([fold_curve(args.emb_pattern, k, grid) for k in map(int, args.folds.split(","))])
    mean, std = curves[..., 0].mean(0), curves[..., 0].std(0)
    best = int(np.argmax(mean))
    # Устойчивая область: пороги, где среднее не хуже максимума на 0.005
    plateau = np.flatnonzero(mean >= mean[best] - 0.005)
    chosen = int(np.median(plateau))
    print(f"{'порог':>7} {'среднее':>9} {'разброс':>9} {'F1':>7} {'TNR':>7}")
    for i in range(0, len(grid), max(1, len(grid) // 20)):
        mark = " <- максимум" if i == best else (" <- выбран" if i == chosen else "")
        print(f"{grid[i]:7.3f} {mean[i]:9.4f} {std[i]:9.4f} "
              f"{curves[:, i, 1].mean():7.4f} {curves[:, i, 2].mean():7.4f}{mark}")
    print(f"\nвыбранный порог {grid[chosen]:.4f}: балл {mean[chosen]:.4f} ± {std[chosen]:.4f} "
          f"(F1 {curves[:, chosen, 1].mean():.4f}, TNR {curves[:, chosen, 2].mean():.4f})")
    print(f"устойчивая область: {grid[plateau[0]]:.3f}–{grid[plateau[-1]]:.3f}")
    print(f"если отвечать всегда: {mean[0]:.4f}")


if __name__ == "__main__":
    main()
