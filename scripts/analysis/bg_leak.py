"""Насколько выдача перекошена в сторону камеры запроса.

Считаем долю чужих машин с ТОЙ ЖЕ камеры среди top-10 (после junk-фильтра) и сравниваем
с базовой долей таких объектов в gallery. Отношение >1 означает, что модель сравнивает сцену.
camera_id используется только для локального анализа на train-фолдах (правило 2).

    PYTHONPATH=. python scripts/analysis/bg_leak.py "artifacts/emb_final/final_oof_f{k}_cls_avg"
"""
import sys
import numpy as np
import pandas as pd

meta = pd.read_csv("splits/folds.csv")
vid = dict(zip(meta.image_id, meta.vehicle_id))
cam = dict(zip(meta.image_id, meta.camera_id))

for name in sys.argv[1:]:
    ratios, obs_all, base_all = [], [], []
    for k in range(5):
        path = f"{name.format(k=k)}/fold{k}.npz"
        try:
            d = np.load(path)
        except FileNotFoundError:
            continue
        q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
        g_vid = np.array([vid[g] for g in g_ids])
        g_cam = np.array([cam[g] for g in g_ids])
        sim = d["query_emb"] @ d["gallery_emb"].T
        order = np.argsort(-sim, axis=1)
        obs, base = [], []
        for i, q in enumerate(q_ids):
            qv, qc = vid[q], cam[q]
            junk = (g_vid == qv) & (g_cam == qc)
            keep = order[i][~junk[order[i]]][:10]
            # чужая машина с камеры запроса
            obs.append(np.mean((g_vid[keep] != qv) & (g_cam[keep] == qc)))
            pool = ~junk
            base.append(np.mean(((g_vid != qv) & (g_cam == qc))[pool]))
        obs_all.append(np.mean(obs)); base_all.append(np.mean(base))
        ratios.append(np.mean(obs) / np.mean(base))
    if ratios:
        print(f"{name:42s} в top-10 {np.mean(obs_all)*100:5.2f}%  фон-база {np.mean(base_all)*100:4.2f}%"
              f"  перекос x{np.mean(ratios):.2f}  (по фолдам: {', '.join(f'{r:.1f}' for r in ratios)})")
