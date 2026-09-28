"""Ансамбль двух моделей: оценка на фолдах.

**Главное наблюдение (A-052).** Простая склейка нормированных эмбеддингов НЕ даёт равного
голосования. Косинусы разных моделей живут в разных масштабах: у DINOv2 разброс 0,085, у SigLIP-2 —
0,0062, то есть в 13,8 раза меньше. При склейке порядок определяет модель с большим разбросом, и
вторая почти не влияет. Именно поэтому ансамбли считались бесполезными (A-017, A-042).

Правильный способ — стандартизовать сходства каждой модели, а потом складывать. Стандартизация
идёт по строке, то есть по одному query относительно всей gallery: используются только сам query и
статическая gallery, независимость запросов соблюдена (AGENTS.md, правило 1).

    PYTHONPATH=. python scripts/analysis/ensemble_eval.py \
        --a artifacts/emb/final_oof_f{k}_cls_avg --b artifacts/emb/siglip2_f{k}_full_avg --folds 0,1,2
"""
import argparse

import numpy as np

from reid.eval.eval_local import evaluate, ranking_from_embeddings


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def zscore_rows(sim):
    """Стандартизация сходств по каждому query отдельно: (s - среднее) / разброс по gallery."""
    return (sim - sim.mean(axis=1, keepdims=True)) / (sim.std(axis=1, keepdims=True) + 1e-9)


def rank_from_sim(sim, gallery_ids, query_ids, k=10):
    order = np.argsort(-sim, axis=1)[:, :max(k, 60)]
    return {q: [gallery_ids[j] for j in order[i]] for i, q in enumerate(query_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--weights", default="0.3,0.5,0.7,1.0")
    ap.add_argument("--standardize", action="store_true",
                    help="складывать стандартизованные сходства вместо склейки векторов")
    args = ap.parse_args()
    folds = [int(k) for k in args.folds.split(",")]
    ws = [float(w) for w in args.weights.split(",")]

    res = {"A": [], "B": [], **{f"w={w}": [] for w in ws}}
    for k in folds:
        da, db = np.load(f"{args.a.format(k=k)}/fold{k}.npz"), np.load(f"{args.b.format(k=k)}/fold{k}.npz")
        q_ids, g_ids = da["query_ids"].tolist(), da["gallery_ids"].tolist()
        assert db["query_ids"].tolist() == q_ids, "порядок query в дампах не совпадает"
        qa, ga = norm(da["query_emb"]), norm(da["gallery_emb"])
        qb, gb = norm(db["query_emb"]), norm(db["gallery_emb"])
        res["A"].append(evaluate(k, ranking_from_embeddings(qa, ga, g_ids, q_ids))["map10"])
        res["B"].append(evaluate(k, ranking_from_embeddings(qb, gb, g_ids, q_ids))["map10"])
        if args.standardize:
            za, zb = zscore_rows(qa @ ga.T), zscore_rows(qb @ gb.T)
            for w in ws:
                res[f"w={w}"].append(evaluate(k, rank_from_sim(za + w * zb, g_ids, q_ids))["map10"])
        else:
            for w in ws:
                q = np.concatenate([qa, w * qb], axis=1)
                g = np.concatenate([ga, w * gb], axis=1)
                res[f"w={w}"].append(
                    evaluate(k, ranking_from_embeddings(norm(q), norm(g), g_ids, q_ids))["map10"])

    print(f"фолды {folds}\n")
    for name, vals in res.items():
        tag = {"A": "только A", "B": "только B"}.get(name, f"ансамбль {name}")
        print(f"{tag:16s} {np.mean(vals):.4f} ± {np.std(vals):.4f}   ({', '.join(f'{v:.4f}' for v in vals)})")
    best = max((n for n in res if n.startswith("w=")), key=lambda n: np.mean(res[n]))
    print(f"\nлучший {best}: {np.mean(res[best]) - np.mean(res['A']):+.4f} к одиночной A")


if __name__ == "__main__":
    main()
