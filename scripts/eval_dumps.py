"""Оценка дампов эмбеддингов на фолдах: косинусный top-10 без обработки двойников.

Кандидаты — наивная политика «всегда отвечать top-1» с confidence = cosine. Она показывает, как
junk-двойники влияют на режим кандидатов (strip vs junk_fp). Настоящая политика — задача трека B.

    PYTHONPATH=. python scripts/eval_dumps.py --emb artifacts/emb/<name>
"""
import argparse
import json
import os

import numpy as np

from reid.eval.eval_local import evaluate, ranking_from_embeddings

KEYS = ["map10", "rank1", "rank5", "full_map", "minp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()

    per_fold = []
    for k in map(int, args.folds.split(",")):
        d = np.load(os.path.join(args.emb, f"fold{k}.npz"))
        q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
        ranking = ranking_from_embeddings(d["query_emb"], d["gallery_emb"], g_ids, q_ids)
        sim = d["query_emb"] @ d["gallery_emb"].T
        cands = [(q, ranking[q][0], float(sim[i].max())) for i, q in enumerate(q_ids)]
        res = evaluate(k, ranking, cands, np.concatenate([d["query_emb"], d["gallery_emb"]]))
        per_fold.append(res)
        print(f"fold {k}: " + "  ".join(f"{m} {res[m]:.4f}" for m in KEYS)
              + f"  | naive top-1: F1 {res['official']['f1']:.3f} (только кросс-камерные {res['cross_cam']['f1']:.3f})")

    summary = {m: (float(np.mean([r[m] for r in per_fold])), float(np.std([r[m] for r in per_fold])))
               for m in KEYS}
    for mode in ("official", "cross_cam"):
        summary[f"naive_{mode}_f1"] = (float(np.mean([r[mode]["f1"] for r in per_fold])),
                                       float(np.std([r[mode]["f1"] for r in per_fold])))
    print("mean ± std: " + "  ".join(f"{m} {v[0]:.4f}±{v[1]:.4f}" for m, v in summary.items()))
    with open(os.path.join(args.emb, "metrics.json"), "w") as f:
        json.dump({"per_fold": per_fold, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
