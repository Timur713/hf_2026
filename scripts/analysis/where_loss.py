"""Где именно теряется mAP@10: на каком ранге стоит первый настоящий позитив.

    PYTHONPATH=. python scripts/analysis/where_loss.py [шаблон каталога дампов]
"""
import sys

import numpy as np, pandas as pd

# Шаблон каталога дампов; по умолчанию финальная модель
EMB = sys.argv[1] if len(sys.argv) > 1 else "artifacts/emb_final/final_oof_f{k}_cls_avg"
meta = pd.read_csv("splits/folds.csv", dtype={"image_id": str})
vid = dict(zip(meta.image_id, meta.vehicle_id)); cam = dict(zip(meta.image_id, meta.camera_id))

buckets = np.zeros(6); tot = 0
same_cam_above = 0; above_tot = 0
for k in range(5):
    d = np.load(f"{EMB.format(k=k)}/fold{k}.npz")
    q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
    g_vid = np.array([vid[g] for g in g_ids]); g_cam = np.array([cam[g] for g in g_ids])
    sim = d["query_emb"] @ d["gallery_emb"].T
    order = np.argsort(-sim, axis=1)
    for i, q in enumerate(q_ids):
        qv, qc = vid[q], cam[q]
        junk = (g_vid == qv) & (g_cam == qc)
        keep = order[i][~junk[order[i]]]
        pos = np.where(g_vid[keep] == qv)[0]
        if len(pos) == 0:
            continue
        tot += 1
        r = pos[0]  # ранг первого позитива, 0-based
        buckets[0 if r == 0 else 1 if r < 3 else 2 if r < 10 else 3 if r < 30 else 4 if r < 100 else 5] += 1
        if r > 0:  # что стоит выше настоящего позитива
            above = keep[:r]
            same_cam_above += np.sum(g_cam[above] == qc); above_tot += r

names = ["ранг 1 (идеал)", "ранги 2-3", "ранги 4-10", "ранги 11-30", "ранги 31-100", "ниже 100"]
for n, b in zip(names, buckets):
    print(f"{n:14s} {b/tot*100:5.1f}%")
print(f"\nсреди объектов, ошибочно стоящих ВЫШЕ настоящего позитива, "
      f"{same_cam_above/above_tot*100:.1f}% — с камеры запроса (в галерее таких {3.2:.1f}%)")
