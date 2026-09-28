"""Порог отказа для фузии со штрафом за хабность.

Прежние 0,600 откалиброваны на шкале косинуса одиночной модели и к фузии не относятся. Режим
кандидатов весит 10% итога, поэтому порог надо пересчитать до сдачи.

**Чем меряется уверенность.** Косинус склейки для выбранного кандидата, приведённый к статистике
галереи: `(cos − μ_gg) / σ_gg`, где μ и σ — среднее и разброс сходств gallery–gallery. Так надо,
потому что у каждого прогона свой средний уровень сходств: на фолде 0 галерея-галерея даёт 0,9456,
на фолде 1 — 0,8790, и абсолютный порог, снятый на одном фолде, на другом рушит балл с 0,963 до
0,872. Нормировка на статистику самой галереи этот сдвиг снимает, сохраняя лучший балл: перенос
порога между фолдами стоит 0,008 вместо 0,091. Величина считается по статической галерее, то есть
разрешена правилом 1 и на инференсе стоит двух констант.

Порог берём в середине устойчивой области по всем доступным фолдам, а не в точке максимума: одиночный
максимум почти всегда случаен (так же выбирался прежний порог, A-028/A-036).

    PYTHONPATH=. python scripts/analysis/calibrate_fusion_threshold.py --folds 0,1 \\
        'artifacts/emb/div_dinov2_f{k}_cls_top16' 'artifacts/emb/div_clip_f{k}_cls_top16' \\
        'artifacts/emb/div_siglip_f{k}_top16'
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from reid.eval.eval_local import FoldTruth, candidate_metrics


def fused(dumps: list[str], fold: int, k: int):
    """Возвращает (оценка ранжирования со штрафом, сырой косинус склейки, ids)."""
    Q = G = None
    q_ids = g_ids = None
    for path in dumps:
        d = np.load(f"{path}/fold{fold}.npz")
        ids = (d["query_ids"].tolist(), d["gallery_ids"].tolist())
        assert q_ids is None or ids == (q_ids, g_ids), f"{path}: другой порядок query/gallery"
        q_ids, g_ids = ids
        unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        qe, ge = unit(d["query_emb"]), unit(d["gallery_emb"])
        scale = np.sqrt((ge @ ge.T).std())          # склейка по D-052
        Q = qe / scale if Q is None else np.concatenate([Q, qe / scale], axis=1)
        G = ge / scale if G is None else np.concatenate([G, ge / scale], axis=1)
    Q, G = (Q / np.linalg.norm(Q, axis=1, keepdims=True),
            G / np.linalg.norm(G, axis=1, keepdims=True))
    raw = Q @ G.T
    z = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-9)
    gg = G @ G.T
    off = ~np.eye(len(G), dtype=bool)
    conf = (raw - gg[off].mean()) / (gg[off].std() + 1e-12)   # уверенность в единицах галереи
    np.fill_diagonal(gg, -np.inf)
    hub = np.sort(gg, axis=1)[:, -k:].mean(axis=1)
    hub = (hub - hub.mean()) / (hub.std() + 1e-9)
    return z - 0.5 * hub, conf, q_ids, g_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--grid", type=int, default=81)
    args = ap.parse_args()

    curves, lo, hi = {}, [], []
    for fold in [int(x) for x in args.folds.split(",")]:
        dumps = [p.format(k=fold) for p in args.patterns]
        if any(not os.path.exists(f"{d}/fold{fold}.npz") for d in dumps):
            print(f"фолд {fold}: пропущен, нет дампов")
            continue
        truth = FoldTruth.load(fold)
        score, raw, q_ids, g_ids = fused(dumps, fold, args.k)
        top = np.argmax(score, axis=1)
        conf = raw[np.arange(len(q_ids)), top]
        lo.append(conf.min())
        hi.append(conf.max())
        curves[fold] = (truth, [(q, g_ids[top[i]], float(conf[i])) for i, q in enumerate(q_ids)])
        print(f"фолд {fold}: уверенность от {conf.min():.3f} до {conf.max():.3f}, "
              f"медиана {np.median(conf):.3f}")

    if not curves:
        return
    grid = np.linspace(min(lo), max(hi), args.grid)
    table = {}
    for fold, (truth, cands) in curves.items():
        row = []
        for t in grid:
            kept = [(q, g, c) for q, g, c in cands if c >= t]
            m = candidate_metrics(truth, kept, "official")
            row.append((m["candidate_score"], m["f1"], m["tnr"]))
        table[fold] = row

    mean = np.array([[table[f][i][0] for f in table] for i in range(len(grid))]).mean(axis=1)
    best = mean.max()
    good = np.where(mean >= best - 0.002)[0]          # устойчивая область, а не точка максимума
    pick = int(np.median(good))
    print(f"\n{'порог':>8s}" + "".join(f"{f'ф{f}':>9s}" for f in table) + f"{'среднее':>10s}")
    for i in range(0, len(grid), max(1, len(grid) // 20)):
        mark = " <-" if i == pick else ""
        print(f"{grid[i]:8.3f}" + "".join(f"{table[f][i][0]:9.4f}" for f in table)
              + f"{mean[i]:10.4f}{mark}")
    print(f"\nустойчивая область: {grid[good[0]]:.3f}…{grid[good[-1]]:.3f}, "
          f"берём середину **{grid[pick]:.3f}**, средний балл {mean[pick]:.4f}")
    for f in table:
        s, f1, tnr = table[f][pick]
        print(f"  фолд {f}: балл {s:.4f}  F1 {f1:.4f}  TNR {tnr:.4f}")


if __name__ == "__main__":
    main()
