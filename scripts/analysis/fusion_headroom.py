"""Сколько ещё можно взять переранжированием поверх фузии — и откуда именно.

Отвечает на три вопроса, от которых зависит, куда вкладывать оставшиеся дни:

1. **Где мы на самом деле.** Протокол сдачи строгий: в файл уходят ровно 10 кандидатов, и
   `ranking_metrics` выбрасывает из них двойника, не добирая хвост. Если подать в оценку длинный
   список, метрика окажется завышенной — ровно так посчитаны прежние числа фузии.
2. **Сколько лежит в окне.** Идеальный порядок внутри top-K при K = 10, 20, 30, 50: это потолок
   любого переранжировщика с таким окном. Разница между K = 10 и K = 30 показывает, окупается ли
   расширение окна.
3. **Сколько стоит двойник.** У каждого query в gallery ровно один кадр той же машины с той же
   камеры, и он почти всегда первый. В сдаче он съедает место из десяти, а оценщик его выбрасывает.
   Считаем, что дало бы его удаление из выдачи — это чистая прибавка без всякого обучения.

    PYTHONPATH=. python scripts/analysis/fusion_headroom.py --fold 0 \\
        artifacts/emb/pool_cls_top16_f0_cls_top16 artifacts/emb/div_clip_f0_cls_top16 \\
        artifacts/emb/div_siglip_f0_top16
"""
from __future__ import annotations

import argparse

import numpy as np

from reid.eval.eval_local import FoldTruth, ranking_metrics


def fused_similarity(dumps: list[str], fold: int):
    """Сумма стандартизованных по строке сходств — фузия из D-052."""
    S, q_ids, g_ids = None, None, None
    for path in dumps:
        d = np.load(f"{path}/fold{fold}.npz")
        ids = (d["query_ids"].tolist(), d["gallery_ids"].tolist())
        assert q_ids is None or ids == (q_ids, g_ids), f"{path}: другой порядок query/gallery"
        q_ids, g_ids = ids
        unit = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        s = unit(d["query_emb"]) @ unit(d["gallery_emb"]).T
        s = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)
        S = s if S is None else S + s
    return S, q_ids, g_ids


def oracle_order(truth: FoldTruth, q: str, cand: list[str]) -> list[str]:
    """Лучший возможный порядок окна: позитивы вперёд, остальное как было.

    Двойник остаётся на своём месте среди «остального»: переранжировщик не знает camera_id и
    убрать двойника по разметке не может — это было бы утечкой.
    """
    pos = [g for g in cand if truth.is_pos(q, g)]
    rest = [g for g in cand if not truth.is_pos(q, g)]
    return pos + rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    S, q_ids, g_ids = fused_similarity(args.dumps, args.fold)
    truth = FoldTruth.load(args.fold)
    order = np.argsort(-S, axis=1)
    ranked = {q: [g_ids[j] for j in order[i]] for i, q in enumerate(q_ids)}
    n_pos = truth.n_pos()
    closed = [q for q in q_ids if n_pos[q] > 0]

    strict = ranking_metrics(truth, {q: v[:10] for q, v in ranked.items()})
    lenient = ranking_metrics(truth, {q: v[:60] for q, v in ranked.items()})
    print(f"фузия из {len(args.dumps)} модел., фолд {args.fold}, {len(closed)} query с парой")
    print(f"  протокол сдачи (ровно 10 кандидатов)      mAP@10 {strict['map10']:.4f}  "
          f"R1 {strict['rank1']:.4f}")
    print(f"  мягкий протокол (длинный список с добором) mAP@10 {lenient['map10']:.4f}  "
          f"— так посчитаны прежние числа, они завышены на {lenient['map10'] - strict['map10']:.4f}")

    # Где стоит первый валидный позитив и сколько их вообще попало в окно
    first, twin_top1 = [], 0
    for q in closed:
        pos_ranks = [r for r, g in enumerate(ranked[q], start=1) if truth.is_pos(q, g)]
        first.append(pos_ranks[0])
        twin_top1 += truth.is_junk(q, ranked[q][0])
    first = np.array(first)
    print(f"\nпервый валидный позитив: ранг 1 у {np.mean(first == 1):.1%}, "
          f"2 у {np.mean(first == 2):.1%}, 3–10 у {np.mean((first >= 3) & (first <= 10)):.1%}, "
          f"11–30 у {np.mean((first >= 11) & (first <= 30)):.1%}, "
          f"дальше 30 у {np.mean(first > 30):.1%}")
    print(f"двойник стоит первым у {twin_top1 / len(closed):.1%} query")

    print("\nидеальный порядок внутри окна (потолок переранжировщика, протокол сдачи):")
    for k in (10, 20, 30, 50):
        best = {q: oracle_order(truth, q, ranked[q][:k])[:10] for q in q_ids}
        m = ranking_metrics(truth, best)["map10"]
        print(f"  окно {k:3d}:  mAP@10 {m:.4f}   запас {m - strict['map10']:+.4f}")

    print("\nчто даёт снятие двойника (место в десятке освобождается):")
    for k in (10, 30):
        # Двойника «угадывает» оракул — это верхняя оценка, а не готовый приём
        no_twin = {q: [g for g in ranked[q][:k] if not truth.is_junk(q, g)][:10] for q in q_ids}
        m = ranking_metrics(truth, no_twin)["map10"]
        print(f"  окно {k:3d}:  mAP@10 {m:.4f}   запас {m - strict['map10']:+.4f}")
        both = {q: oracle_order(truth, q, [g for g in ranked[q][:k]
                                           if not truth.is_junk(q, g)])[:10] for q in q_ids}
        mb = ranking_metrics(truth, both)["map10"]
        print(f"            вместе с идеальным порядком {mb:.4f}   запас {mb - strict['map10']:+.4f}")


if __name__ == "__main__":
    main()
