"""Окно кандидатов production-порядка для снятия логитов верификатора (A-101).

Аудит трека B (D-060) показал, что сохранённые логиты сняты на окне 30, выбранном
`fused_similarity` БЕЗ штрафа за хабность, тогда как сдача переранжирует top-10 уже
ПОСЛЕ него. Обрезать 30 до 10 нельзя: набор кандидатов другой.

Здесь окно считается ровно так, как в сдаче: склейка с масштабом по √σ галереи (D-052),
построчная стандартизация, минус λ·z(хабность) (D-055), затем top-K. Двойник не снимается —
на инференсе `camera_id` недоступен, и evaluator убирает его сам.

    PYTHONPATH=. python scripts/analysis/export_window.py --fold 0 --out artifacts/rerank/window_f0.npz \
        artifacts/emb/hires392_f0_cls_top16 artifacts/emb/div_clip_f0_cls_top16 \
        artifacts/emb/div_siglip_f0_top16
"""
from __future__ import annotations

import argparse

import numpy as np

from scripts.analysis.hub_penalty import concat_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3, help="соседей для хабности (D-055)")
    ap.add_argument("--lam", type=float, default=0.5, help="вес штрафа за хабность (D-055)")
    ap.add_argument("--topk", type=int, default=10, help="размер окна; сдача использует 10")
    args = ap.parse_args()

    S, hub, q_ids, g_ids = concat_parts(args.dumps, args.fold, args.k)
    M = S - args.lam * hub[None, :]
    order = np.argsort(-M, axis=1)[:, : args.topk]
    g_arr = np.array(g_ids)
    np.savez(
        args.out,
        fold=np.int64(args.fold),
        query_ids=np.array(q_ids),
        candidates=g_arr[order],
        sim=np.take_along_axis(M, order, axis=1).astype(np.float32),
        emb=np.array(args.dumps),
        hub_k=np.int64(args.k),
        hub_lambda=np.float32(args.lam),
    )
    print(f"фолд {args.fold}: {len(q_ids)} query x {args.topk} кандидатов -> {args.out}")
    print("порядок: склейка D-052 -> построчный z -> минус lambda*z(хабность) D-055 -> top-K")


if __name__ == "__main__":
    main()
