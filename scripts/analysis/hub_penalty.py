"""Штраф за «хабы» галереи: кадр, близкий ко всему подряд, не должен выигрывать спор.

Идея. В галерее есть кадры-хабы — ракурсы и условия съёмки, к которым косинус благоволит
независимо от того, та же это машина или нет. Они вытесняют настоящий позитив из верхних строк.
Хабность кадра считается **только по сходствам gallery–gallery**: средняя близость кадра к его k
ближайшим соседям по галерее. Это константа на кадр, она предвычисляется один раз и на инференсе
не стоит ничего.

Правила соблюдены. Ответ на запрос зависит только от него самого и от статической галереи:
предвычисленные сходства gallery–gallery разрешены прямо (AGENTS.md, правило 1). Никакой
статистики по другим query и никакого query expansion здесь нет.

    PYTHONPATH=. python scripts/analysis/hub_penalty.py --fold 0 \\
        artifacts/emb/pool_cls_top16_f0_cls_top16 artifacts/emb/div_clip_f0_cls_top16 \\
        artifacts/emb/div_siglip_f0_top16
"""
from __future__ import annotations

import argparse

import numpy as np

from reid.eval.eval_local import FoldTruth, ranking_metrics


def concat_parts(dumps: list[str], fold: int, k: int):
    """То же, но по склейке — ровно так, как считает сдача (D-052).

    Вектор каждой модели делится на √σ её сходств по галерее, склейка нормируется, дальше обычный
    косинус. Хабность считается по той же склейке, поэтому треку B не нужна помодельная
    бухгалтерия: достаточно эмбеддингов галереи.
    """
    Q = G = None
    q_ids = g_ids = None
    for path in dumps:
        d = np.load(f"{path}/fold{fold}.npz")
        ids = (d["query_ids"].tolist(), d["gallery_ids"].tolist())
        assert q_ids is None or ids == (q_ids, g_ids), f"{path}: другой порядок query/gallery"
        q_ids, g_ids = ids
        unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        qe, ge = unit(d["query_emb"]), unit(d["gallery_emb"])
        scale = np.sqrt((ge @ ge.T).std())
        Q = qe / scale if Q is None else np.concatenate([Q, qe / scale], axis=1)
        G = ge / scale if G is None else np.concatenate([G, ge / scale], axis=1)
    Q, G = (Q / np.linalg.norm(Q, axis=1, keepdims=True),
            G / np.linalg.norm(G, axis=1, keepdims=True))
    S = Q @ G.T
    S = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)
    gg = G @ G.T
    np.fill_diagonal(gg, -np.inf)
    hub = np.sort(gg, axis=1)[:, -k:].mean(axis=1)
    hub = (hub - hub.mean()) / (hub.std() + 1e-9)
    return S, hub, q_ids, g_ids


def parts(dumps: list[str], fold: int, k: int):
    """Сумма стандартизованных сходств и сумма стандартизованных хабностей галереи."""
    S = H = None
    q_ids = g_ids = None
    for path in dumps:
        d = np.load(f"{path}/fold{fold}.npz")
        ids = (d["query_ids"].tolist(), d["gallery_ids"].tolist())
        assert q_ids is None or ids == (q_ids, g_ids), f"{path}: другой порядок query/gallery"
        q_ids, g_ids = ids
        unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        qe, ge = unit(d["query_emb"]), unit(d["gallery_emb"])
        s = qe @ ge.T
        s = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)
        S = s if S is None else S + s

        gg = ge @ ge.T
        np.fill_diagonal(gg, -np.inf)                       # сам себе не сосед
        hub = np.sort(gg, axis=1)[:, -k:].mean(axis=1)      # близость к k ближайшим по галерее
        hub = (hub - hub.mean()) / (hub.std() + 1e-9)       # в те же единицы, что и сходства
        H = hub if H is None else H + hub
    return S, H, q_ids, g_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10, 20, 50])
    ap.add_argument("--lams", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
    ap.add_argument("--concat", action="store_true",
                    help="считать по склейке, как в сдаче: тогда оптимум λ около 0,5")
    args = ap.parse_args()
    build = concat_parts if args.concat else parts

    truth = FoldTruth.load(args.fold)

    def score(M, g_ids, q_ids):
        order = np.argsort(-M, axis=1)[:, :10]
        return ranking_metrics(truth, {q: [g_ids[j] for j in order[i]]
                                       for i, q in enumerate(q_ids)})["map10"]

    S, _, q_ids, g_ids = build(args.dumps, args.fold, args.ks[0])
    base = score(S, g_ids, q_ids)
    print(f"фолд {args.fold}, {len(args.dumps)} модел., база {base:.4f}\n")
    print("     λ " + "".join(f"{lam:>9}" for lam in args.lams))
    best = (base, None, None)
    for k in args.ks:
        S, H, q_ids, g_ids = build(args.dumps, args.fold, k)
        row = []
        for lam in args.lams:
            v = score(S - lam * H[None, :], g_ids, q_ids)
            best = max(best, (v, k, lam))
            row.append(v)
        print(f"k={k:<4}" + "".join(f"{v:>9.4f}" for v in row))
    print(f"\nлучшее: k={best[1]} λ={best[2]}  {best[0]:.4f}  ({best[0] - base:+.4f})")


if __name__ == "__main__":
    main()
