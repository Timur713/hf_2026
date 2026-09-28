"""Сверяет reid/eval/eval_local.py с эталонным third_party/official/evaluate.py (задача A1.5).

Берёт дампы эмбеддингов фолда, строит из них три файла сдачи и ground truth фолда, затем считает
метрики обоими способами и сравнивает. Любое расхождение больше 1e-6 — ошибка: эталон главнее.

    PYTHONPATH=. python scripts/crosscheck_official.py --emb artifacts/emb/<name> --fold 0
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from reid.eval.eval_local import evaluate, ranking_from_embeddings

TOL = 1e-6


def write_submission(path, ranking, query_ids):
    with open(path, "w", newline="") as f:  # БЕЗ заголовка, как читает эталонный скрипт
        csv.writer(f).writerows([[q] + ranking[q] for q in query_ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="artifacts/crosscheck")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.emb, f"fold{args.fold}.npz"))
    q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
    q_emb, g_emb = d["query_emb"], d["gallery_emb"]
    ranking = ranking_from_embeddings(q_emb, g_emb, g_ids, q_ids)
    sim = q_emb @ g_emb.T
    # Кандидаты: верхний по сходству, но только если оно выше медианы — чтобы в сверке были
    # и ответы, и отказы, и TP, и FP, и TN.
    top = sim.max(axis=1)
    thr = float(np.median(top))
    candidates = [(q, ranking[q][0], float(top[i])) for i, q in enumerate(q_ids) if top[i] >= thr]

    sub = os.path.join(args.out, "submission.csv")
    cand = os.path.join(args.out, "candidates.csv")
    emb_path = os.path.join(args.out, "embeddings.npy")
    gt_path = os.path.join(args.out, "gt.csv")
    q_csv = os.path.join(args.out, "query.csv")
    g_csv = os.path.join(args.out, "gallery.csv")

    write_submission(sub, ranking, q_ids)
    pd.DataFrame(candidates, columns=["query_id", "gallery_id", "confidence"]).to_csv(cand, index=False)
    np.save(emb_path, np.concatenate([q_emb, g_emb]))
    folds = pd.read_csv("splits/folds.csv", dtype={"image_id": str})
    folds = folds[folds.fold == args.fold]
    gt = folds[folds.role.isin(["query", "gallery"])].copy()
    gt["split"] = gt.role
    gt[["image_id", "vehicle_id", "camera_id", "split"]].to_csv(gt_path, index=False)
    for path, ids in ((q_csv, q_ids), (g_csv, g_ids)):
        pd.DataFrame({"image_id": ids, "x": 0, "y": 0, "w": 1, "h": 1}).to_csv(path, index=False)

    report = os.path.join(args.out, "official.json")
    subprocess.run([sys.executable, "third_party/official/evaluate.py", "--gt", gt_path,
                    "--submission", sub, "--candidates", cand, "--embeddings", emb_path,
                    "--query", q_csv, "--gallery", g_csv, "--json", report], check=True)
    off = json.load(open(report))
    ours = evaluate(args.fold, ranking, candidates, np.concatenate([q_emb, g_emb]))

    pairs = [
        ("mAP@10", off["ranking"]["mAP@10"], ours["map10"]),
        ("Rank-1", off["ranking"]["Rank-1"], ours["rank1"]),
        ("Rank-5", off["ranking"]["Rank-5"], ours["rank5"]),
        ("mAP full", off["full_ranking"]["mAP_full"], ours["full_map"]),
        ("mINP", off["full_ranking"]["mINP"], ours["minp"]),
        ("F1", off["candidates"]["F1"], ours["official"]["f1"]),
        ("TNR", off["candidates"]["TNR"], ours["official"]["tnr"]),
        ("Precision", off["candidates"]["Precision"], ours["official"]["precision"]),
        ("Recall", off["candidates"]["Recall"], ours["official"]["recall"]),
        ("PR-AUC", off["candidates"]["PR-AUC"], ours["official"]["pr_auc"]),
        ("TP", off["candidates"]["TP"], ours["official"]["tp"]),
        ("FP", off["candidates"]["FP"], ours["official"]["fp"]),
        ("FN", off["candidates"]["FN"], ours["official"]["fn"]),
        ("TN", off["candidates"]["TN"], ours["official"]["tn"]),
        ("query в зачёте", off["ranking"]["n_scored"], ours["n_query_eval"]),
        ("open-set вне mAP", off["ranking"]["n_openset_excluded"], ours["n_query_openset"]),
    ]
    print(f"\n{'метрика':<18}{'эталон':>12}{'наша':>12}   ")
    bad = 0
    for name, a, b in pairs:
        ok = abs(float(a) - float(b)) <= TOL
        bad += not ok
        print(f"{name:<18}{float(a):>12.6f}{float(b):>12.6f}   {'ok' if ok else 'РАСХОЖДЕНИЕ'}")
    print(f"\nдиагностика: F1 только по кросс-камерным ответам {ours['cross_cam']['f1']:.4f} "
          f"(в официальном зачёте {ours['official']['f1']:.4f})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
