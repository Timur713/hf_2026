"""Потолок переранжирования: сколько mAP@10 даёт идеальный порядок внутри окна top-K.

Переранжирование не входит в замер скорости (D-0xx), поэтому окно можно брать любое разумное.
Вопрос в том, где оно перестаёт окупаться: если позитив и так редко бывает за пределами top-10,
расширять окно незачем.

    PYTHONPATH=. python scripts/analysis/rerank_ceiling.py [шаблон каталога дампов]
"""
import sys

import numpy as np

from reid.eval.eval_local import FoldTruth, evaluate

EMB = sys.argv[1] if len(sys.argv) > 1 else "artifacts/emb_final/final_oof_f{k}_cls_avg"
WINDOWS = (10, 20, 30, 50, 100, 200)


def main():
    base, ceil = [], {w: [] for w in WINDOWS}
    for k in range(5):
        d = np.load(f"{EMB.format(k=k)}/fold{k}.npz")
        q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
        truth = FoldTruth.load(k)
        sim = d["query_emb"] @ d["gallery_emb"].T
        order = np.argsort(-sim, axis=1)

        as_is = {q: [g_ids[j] for j in order[i][:60]] for i, q in enumerate(q_ids)}
        base.append(evaluate(k, as_is)["map10"])

        for w in WINDOWS:
            ranked = {}
            for i, q in enumerate(q_ids):
                win = [g_ids[j] for j in order[i][:w]]
                # идеальный переранжировщик: позитивы окна наверх, порядок остальных не важен
                pos = [g for g in win if truth.is_pos(q, g)]
                junk = [g for g in win if truth.is_junk(q, g)]
                rest = [g for g in win if g not in set(pos) | set(junk)]
                tail = [g_ids[j] for j in order[i][w:60]]
                ranked[q] = junk + pos + rest + tail
            ceil[w].append(evaluate(k, ranked)["map10"])

    b = np.mean(base)
    print(f"как есть (косинус)            mAP@10 {b:.4f}\n")
    print("идеальный порядок внутри окна:")
    for w in WINDOWS:
        c = np.mean(ceil[w])
        print(f"  top-{w:<4} {c:.4f}   (+{c - b:.4f})")
    print("\nСтрока показывает, сколько ещё можно взять, не трогая само представление.")


if __name__ == "__main__":
    main()
