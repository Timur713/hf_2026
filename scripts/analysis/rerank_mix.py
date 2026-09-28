"""Способы соединить сходство фузии с логитом верификатора — на снятых заранее логитах.

Входной файл делает `rerank_dump.py`. Здесь только numpy, поэтому перебор идёт секундами и на
ноутбуке.

Проверяется гипотеза, ради которой всё и затевалось. Карта запаса (`fusion_headroom.py`) говорит,
что половина query уже идеальна: единственный позитив стоит сразу за двойником, AP = 1,0. Весь
доступный запас лежит в 16% query, где позитив на 3–10 месте. Равномерный вес применяет
верификатор и к идеальной половине, где он способен только испортить. Отсюда гейт: пускать
верификатор туда, где база не уверена, и не трогать остальное.

Уверенность базы меряется **отрывом** внутри списка кандидатов: насколько первый непохожий на
двойника кандидат оторвался от следующего. Двойника мы по разметке не знаем и знать не можем
(camera_id на инференсе запрещён), поэтому отрыв считается по позициям 2 и 3 списка — там, где у
79,5% query стоит настоящий позитив, а за ним первый конкурент.

Подгонка под фолд отсекается делением query пополам: параметры выбираются на одной половине,
число докладывается на другой.

    PYTHONPATH=. python scripts/analysis/rerank_mix.py --dump artifacts/rerank/v_fus_f1only_fuse3_f0.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from reid.eval.eval_local import FoldTruth


def z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean(axis=-1, keepdims=True)) / (v.std(axis=-1, keepdims=True) + 1e-9)


class Bench:
    """Оценка порядка кандидатов по протоколу сдачи: ровно 10 в файл, junk выбрасывает оценщик."""

    def __init__(self, dump: str):
        d = np.load(dump, allow_pickle=True)
        self.fold = int(d["fold"])
        self.q_ids = [str(q) for q in d["query_ids"]]
        self.cand = d["candidates"].astype(str)
        self.sim, self.logits = d["sim"].astype(np.float64), d["logits"].astype(np.float64)
        self.truth = FoldTruth.load(self.fold)
        n_pos = self.truth.n_pos()
        self.keep = np.array([n_pos[q] > 0 for q in self.q_ids])
        self.norm = np.array([min(n_pos[q], 10) for q in self.q_ids], dtype=np.float64)
        # Булевы маски позитивов и двойников считаем один раз: перебор весов их трогает сотни раз
        self.is_pos = np.array([[self.truth.is_pos(q, g) for g in row]
                                for q, row in zip(self.q_ids, self.cand)])
        self.is_junk = np.array([[self.truth.is_junk(q, g) for g in row]
                                 for q, row in zip(self.q_ids, self.cand)])

    def ap(self, score: np.ndarray) -> np.ndarray:
        """AP@10 каждого query при заданных оценках кандидатов (больше — выше)."""
        order = np.argsort(-score, axis=1, kind="stable")
        pos = np.take_along_axis(self.is_pos, order, axis=1)
        junk = np.take_along_axis(self.is_junk, order, axis=1)
        out = np.zeros(len(self.q_ids))
        for i in range(len(self.q_ids)):
            sub = pos[i][:10][~junk[i][:10]]        # в файл уходят 10, оценщик выбрасывает двойника
            hits = np.cumsum(sub)
            out[i] = np.sum(np.where(sub, hits / np.arange(1, len(sub) + 1), 0.0)) / max(self.norm[i], 1.0)
        return out

    def map10(self, score: np.ndarray, mask: np.ndarray | None = None) -> float:
        m = self.keep if mask is None else (self.keep & mask)
        return float(self.ap(score)[m].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    ap.add_argument("--gates", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0])
    args = ap.parse_args()

    b = Bench(args.dump)
    zs, zl = z(b.sim), z(b.logits)
    base = b.map10(b.sim)
    print(f"фолд {b.fold}, {int(b.keep.sum())} query с парой, окно {b.cand.shape[1]}")
    print(f"база (фузия, без переранжирования): {base:.4f}\n")

    print("равномерный вес:")
    best_raw = best_z = (0.0, None)
    for w in args.weights:
        r = b.map10(b.sim + w / (1 + np.exp(-b.logits)))
        zz = b.map10(zs + w * zl)
        best_raw = max(best_raw, (r, w))
        best_z = max(best_z, (zz, w))
        print(f"  w={w:<4}  сырое {r:.4f} ({r - base:+.4f})   выровненное {zz:.4f} ({zz - base:+.4f})")

    # Отрыв: насколько кандидат на 2-м месте оторвался от 3-го. Первое место почти всегда занимает
    # двойник, поэтому спор за верхнюю строку идёт между вторым и третьим.
    margin = zs[:, 1] - zs[:, 2]
    w_star = best_z[1]
    print(f"\nгейт по отрыву 2-го от 3-го (вес {w_star}, вмешательство только при отрыве < порога):")
    for g in args.gates:
        act = margin < g
        score = np.where(act[:, None], zs + w_star * zl, zs)
        v = b.map10(score)
        print(f"  порог {g:<4}  {v:.4f} ({v - base:+.4f})   вмешались в {act.mean():.1%} query")

    print(f"\nмягкий гейт: вес падает с ростом отрыва, w·exp(-отрыв/масштаб):")
    for g in args.gates:
        score = zs + (w_star * np.exp(-margin / g))[:, None] * zl
        v = b.map10(score)
        print(f"  масштаб {g:<4}  {v:.4f} ({v - base:+.4f})")

    # Разложение по тем же группам, что и в fusion_headroom: где верификатор берёт, где отдаёт
    print(f"\nразложение равномерного веса {w_star} по группам:")
    ap_base, ap_mix = b.ap(zs), b.ap(zs + w_star * zl)
    n_pos = b.truth.n_pos()
    first = np.array([next((r for r, p in enumerate(row, 1) if p), 999) for row in b.is_pos])
    npos = np.array([n_pos[q] for q in b.q_ids])
    groups = {
        "n_pos=1, позитив 1–2": (npos == 1) & (first <= 2),
        "n_pos=1, позитив 3–10": (npos == 1) & (first >= 3) & (first <= 10),
        "n_pos>1, позитив 1–2": (npos > 1) & (first <= 2),
        "n_pos>1, позитив 3–10": (npos > 1) & (first >= 3) & (first <= 10),
        "позитив дальше 10": first > 10,
    }
    for name, m in groups.items():
        m = m & b.keep
        if not m.any():
            continue
        d = (ap_mix[m] - ap_base[m]).sum() / b.keep.sum()
        print(f"  {name:24s} доля {m.sum() / b.keep.sum():5.1%}   вклад в mAP {d:+.4f}")

    # Половина query для подбора, половина для проверки: гейт не должен быть подгонкой
    half = np.zeros(len(b.q_ids), dtype=bool)
    half[::2] = True
    print("\nпроверка на половине query (подбор на чётных, число на нечётных):")
    cands = [("равномерный", lambda w, g: zs + w * zl)]
    cands += [("жёсткий гейт", lambda w, g: np.where((margin < g)[:, None], zs + w * zl, zs))]
    cands += [("мягкий гейт", lambda w, g: zs + (w * np.exp(-margin / g))[:, None] * zl)]
    for name, fn in cands:
        grid = [(b.map10(fn(w, g), half), w, g) for w in args.weights for g in args.gates]
        _, w, g = max(grid)
        v = b.map10(fn(w, g), ~half)
        print(f"  {name:14s} w={w} порог={g}   на отложенной половине {v:.4f} "
              f"({v - b.map10(zs, ~half):+.4f})")


if __name__ == "__main__":
    main()
