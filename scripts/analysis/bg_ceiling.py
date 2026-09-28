"""Потолок выигрыша от полного устранения фоновой подсказки.

Оракул: запрещаем модели ставить наверх чужие машины с камеры запроса (опускаем их в конец).
Это не решение — camera_id на инференсе использовать нельзя, — а верхняя граница того, сколько
mAP@10 стоит фоновый ярлык. Реальная сегментация даст только часть этого.
"""
import sys

import numpy as np
import pandas as pd

# Шаблон каталога дампов; по умолчанию финальная модель
EMB = sys.argv[1] if len(sys.argv) > 1 else "artifacts/emb_final/final_oof_f{k}_cls_avg"
from reid.eval.eval_local import FoldTruth, ranking_from_embeddings, evaluate

meta = pd.read_csv("splits/folds.csv", dtype={"image_id": str})
vid = dict(zip(meta.image_id, meta.vehicle_id))
cam = dict(zip(meta.image_id, meta.camera_id))

base_m, orac_m, half_m = [], [], []
for k in range(5):
    d = np.load(f"{EMB.format(k=k)}/fold{k}.npz")
    q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
    g_vid = np.array([vid[g] for g in g_ids]); g_cam = np.array([cam[g] for g in g_ids])
    q_vid = np.array([vid[q] for q in q_ids]); q_cam = np.array([cam[q] for q in q_ids])
    sim = d["query_emb"] @ d["gallery_emb"].T

    same_cam_other = (g_cam[None, :] == q_cam[:, None]) & (g_vid[None, :] != q_vid[:, None])
    for tag, penalty, box in (("base", 0.0, base_m), ("half", 0.02, half_m), ("oracle", 9.0, orac_m)):
        s = sim - penalty * same_cam_other
        rank = {q: [g_ids[j] for j in np.argsort(-s[i])[:60]] for i, q in enumerate(q_ids)}
        r = evaluate(k, rank, [(q, rank[q][0], 1.0) for q in q_ids],
                     np.concatenate([d["query_emb"], d["gallery_emb"]]))
        box.append(r["map10"])

print(f"как есть                     mAP@10 {np.mean(base_m):.4f}")
print(f"штраф 0.02 чужой-с-камеры    mAP@10 {np.mean(half_m):.4f}  ({np.mean(half_m)-np.mean(base_m):+.4f})")
print(f"оракул без фоновой подсказки mAP@10 {np.mean(orac_m):.4f}  ({np.mean(orac_m)-np.mean(base_m):+.4f})")
