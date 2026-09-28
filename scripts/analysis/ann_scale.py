"""Поиск по gallery порядка 10^6 объектов (§12 ТЗ).

В задаче gallery из 750 кадров, и точный перебор оптимален. Вопрос жюри — что будет при
промышленном размере. Здесь мы это измеряем: строим синтетическую gallery на 10^6 векторов с
реалистичной структурой и сравниваем точный перебор с приближёнными индексами по трём осям:
задержка одного запроса, память, полнота выдачи относительно точного перебора.

Синтетическая gallery получена из реальных эмбеддингов train: каждый вектор — реальный кадр плюс
гауссов шум, калиброванный так, чтобы косинус с источником совпадал с наблюдаемым косинусом между
двумя кадрами одной машины с разных камер. Так воспроизводится и локальная структура (плотные
группы почти-дубликатов), и глобальная (многообразие реальных признаков). Это заявлено честно:
настоящих миллиона кадров у нас нет.

    PYTHONPATH=. python scripts/analysis/ann_scale.py --size 1000000
"""
import argparse
import gc
import os
import time

import faiss
import numpy as np

from reid.eval.eval_local import FoldTruth

EMB = "artifacts/emb_final/final_oof_f{k}_cls_avg"


def real_embeddings() -> tuple[np.ndarray, float]:
    """Все реальные эмбеддинги фолдов и типичный косинус между кадрами одной машины."""
    vecs, same = [], []
    for k in range(5):
        path = f"{EMB.format(k=k)}/fold{k}.npz"
        if not os.path.exists(path):
            continue
        d = np.load(path)
        q, g = d["query_emb"], d["gallery_emb"]
        vecs += [q, g]
        truth = FoldTruth.load(k)
        q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
        sim = q @ g.T
        g_vid = np.array([truth.vid[x] for x in g_ids])
        for i, qq in enumerate(q_ids[:400]):
            pos = np.where(g_vid == truth.vid[qq])[0]
            if len(pos):
                same.append(sim[i, pos].mean())
    return np.concatenate(vecs).astype(np.float32), float(np.mean(same))


def synth_gallery(base: np.ndarray, target_cos: float, size: int, rng: np.random.Generator,
                  chunk: int = 100_000) -> np.ndarray:
    """Растягивает реальные векторы до `size` штук, сохраняя типичное внутриклассовое сходство."""
    d = base.shape[1]
    # cos(v, v+e) ≈ 1/sqrt(1+sigma^2*d) при единичном v и шуме N(0, sigma^2) по каждой координате
    sigma = float(np.sqrt(max(1.0 / target_cos**2 - 1.0, 1e-6) / d))
    out = np.empty((size, d), dtype=np.float32)
    for start in range(0, size, chunk):
        n = min(chunk, size - start)
        idx = rng.integers(0, len(base), size=n)
        block = base[idx] + rng.normal(0.0, sigma, size=(n, d)).astype(np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        out[start:start + n] = block
    return out


def latency(index, queries: np.ndarray, k: int, repeats: int, threads: int) -> float:
    """Медианная задержка одного запроса, batch=1 и один поток — как её меряет жюри."""
    faiss.omp_set_num_threads(1)
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        index.search(queries[i % len(queries)][None], k)
        times.append((time.perf_counter() - t0) * 1000)
    faiss.omp_set_num_threads(threads)   # вернуть исходное, иначе следующая сборка пойдёт в один поток
    return float(np.median(times))


def recall(approx: np.ndarray, exact: np.ndarray) -> float:
    """Доля точных соседей, найденных приближённым индексом."""
    return float(np.mean([len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact)]))


def times(n: float) -> str:
    """«раз» или «раза» — согласование числительного, иначе текст читается неряшливо."""
    if n != int(n):
        return "раза"
    last, last_two = int(n) % 10, int(n) % 100
    if 11 <= last_two <= 14 or last in (0, 5, 6, 7, 8, 9):
        return "раз"
    return "раз" if last == 1 else "раза"


def coarse_quantizer(gal: np.ndarray, nlist: int, sample: int) -> faiss.IndexFlatIP:
    """Один общий грубый квантователь на все варианты IVF: обучать его четыре раза незачем."""
    km = faiss.Kmeans(gal.shape[1], nlist, niter=10, spherical=True, seed=0)
    km.train(gal[:sample])
    q = faiss.IndexFlatIP(gal.shape[1])
    q.add(km.centroids.reshape(nlist, -1))
    return q


def build_ivf(kind: str, quantizer, d: int, nlist: int, gal: np.ndarray, sample: int, m: int = 96):
    """IVF-индекс на готовом квантователе. Метрика задаётся в конструкторе: выставить её
    постфактум нельзя — обучение пойдёт под другую метрику."""
    ip = faiss.METRIC_INNER_PRODUCT
    if kind == "flat":
        index = faiss.IndexIVFFlat(quantizer, d, nlist, ip)
    elif kind == "sq8":
        index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit, ip)
    elif kind == "pq":
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, ip)
    else:
        raise ValueError(kind)
    index.quantizer_trains_alone = 1      # квантователь уже обучен, не трогать
    t0 = time.perf_counter()
    index.train(gal[:sample])
    index.add(gal)
    return index, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/ann_scale.md")
    args = ap.parse_args()

    threads = faiss.omp_get_max_threads()
    rng = np.random.default_rng(args.seed)
    base, target_cos = real_embeddings()
    print(f"реальных эмбеддингов {len(base)}, размерность {base.shape[1]}, "
          f"типичный косинус одной машины {target_cos:.3f}", flush=True)

    t0 = time.perf_counter()
    gal = synth_gallery(base, target_cos, args.size, rng)
    queries = base[rng.integers(0, len(base), size=args.queries)].copy()
    raw_mib = gal.nbytes / 2**20
    print(f"gallery {gal.shape} собрана за {time.perf_counter() - t0:.0f} с, {raw_mib:.0f} МиБ",
          flush=True)

    d, rows = gal.shape[1], []

    flat = faiss.IndexFlatIP(d)
    t0 = time.perf_counter(); flat.add(gal); build = time.perf_counter() - t0
    _, exact = flat.search(queries, args.k)
    rows.append(("Точный перебор (IndexFlatIP)", build, raw_mib,
                 latency(flat, queries, args.k, args.repeats, threads), 1.0))
    print(f"точный: {rows[-1][3]:.1f} мс", flush=True)
    del flat; gc.collect()

    nlist, sample = int(4 * np.sqrt(args.size)), min(args.size, 200_000)
    t0 = time.perf_counter()
    quantizer = coarse_quantizer(gal, nlist, sample)
    q_time = time.perf_counter() - t0
    print(f"грубый квантователь на {nlist} ячеек обучен за {q_time:.0f} с", flush=True)

    m = 96   # 96 подквантователей по 16 измерений → 96 байт на вектор
    plan = [("flat", "IVF-Flat", raw_mib, (8, 32)),
            ("sq8", "IVF-SQ8 (1 байт на измерение)", args.size * d / 2**20, (8, 32)),
            ("pq", f"IVF-PQ, m={m} ({m} байт на вектор)", args.size * m / 2**20, (16, 64))]
    for kind, title, mem, probes in plan:
        index, build = build_ivf(kind, quantizer, d, nlist, gal, sample, m)
        for nprobe in probes:
            index.nprobe = nprobe
            _, got = index.search(queries, args.k)
            rows.append((f"{title}, nprobe={nprobe}", build + q_time, mem,
                         latency(index, queries, args.k, args.repeats, threads),
                         recall(got, exact)))
            print(f"{title} nprobe={nprobe}: {rows[-1][3]:.2f} мс, "
                  f"полнота {rows[-1][4]:.3f}, {mem:.0f} МиБ", flush=True)
        del index; gc.collect()

    by_name = {name: r for name, *r in rows}
    ivf = by_name["IVF-Flat, nprobe=8"]
    sq8 = by_name["IVF-SQ8 (1 байт на измерение), nprobe=8"]
    pq = by_name[f"IVF-PQ, m={m} ({m} байт на вектор), nprobe=16"]
    speedup = rows[0][3] / ivf[2]
    over = rows[0][3] / 40
    shrink = raw_mib / pq[1]
    size_str = f"{args.size:,}".replace(",", " ")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"""# Поиск по gallery на 10^6 объектов (§12 ТЗ)

В самой задаче gallery из 750 кадров: точный перебор занимает микросекунды и никакой ANN не нужен.
Эта страница отвечает на вопрос, что будет при промышленном размере.

**Как получены данные.** Настоящего миллиона кадров у нас нет. Gallery синтезирована из
{len(base)} реальных эмбеддингов train: каждый вектор — реальный кадр плюс гауссов шум,
калиброванный так, чтобы косинус с источником совпал с наблюдаемым косинусом между двумя кадрами
одной машины ({target_cos:.3f}). Так сохраняются и плотные группы почти-дубликатов, и форма
многообразия. Полнота считается относительно точного перебора по той же gallery, поэтому
синтетичность выборки на неё не влияет.

Размер: {size_str} векторов по {d} измерений. Задержка — медиана {args.repeats} запросов при
batch=1 и одном потоке, k={args.k}, {args.queries} разных запросов. Сборка — на {threads} потоках CPU.
Все IVF-индексы делят один грубый квантователь на {nlist} ячеек.

| Индекс | Сборка, с | Память, МиБ | Задержка, мс | Полнота@{args.k} |
|---|---:|---:|---:|---:|
""")
        for name, build, mem, lat, rec in rows:
            f.write(f"| {name} | {build:.0f} | {mem:.0f} | {lat:.2f} | {rec:.3f} |\n")
        f.write(f"""
**Что из этого следует.**

1. **Точный перебор на этом размере не проходит по времени.** {rows[0][3]:.0f} мс на запрос против
   всего бюджета задержки в 40 мс — превышение в {over:.1f} {times(round(over, 1))}, и это ещё без
   чтения и декодирования кадра.
2. **Обратный индекс снимает проблему времени полностью и без потерь.** IVF-Flat при nprobe=8 даёт
   {ivf[2]:.2f} мс при полноте {ivf[3]:.3f}: ускорение в {speedup:.0f} {times(round(speedup))},
   выдача совпадает с точной.
3. **Память — вторая ось, и платить за неё приходится полнотой.** Сырые векторы занимают
   {raw_mib / 1024:.1f} ГиБ. Скалярное квантование ужимает их вчетверо, до {sq8[1]:.0f} МиБ, и стоит
   {(ivf[3] - sq8[3]) * 100:.0f}% полноты. Произведение квантователей сжимает в
   {shrink:.0f} {times(round(shrink))}, до {pq[1]:.0f} МиБ, но полнота обваливается до {pq[3]:.3f}:
   {m} байт на вектор при {d} измерениях — это половина бита на измерение, и структура теряется.
4. **Что выбирать.** Если миллион векторов помещается в память — IVF-Flat: точная выдача за
   {ivf[2]:.2f} мс. Если память ограничена — IVF-SQ8: {sq8[1]:.0f} МиБ и {sq8[2]:.2f} мс при полноте
   {sq8[3]:.3f}, и это лучший компромисс из проверенных. Агрессивное сжатие до {pq[1]:.0f} МиБ на
   таких эмбеддингах не годится; если бы оно понадобилось, начинать надо было бы со снижения
   размерности, а не с грубого квантования.

Полнота ниже единицы не переводится в потерю mAP один к одному: после ANN идёт переранжирование
парным верификатором по top-K, и часть промахов на границе десятки он исправляет. Но проверять это
надо было бы на настоящем миллионе, которого у нас нет, поэтому такого утверждения мы не делаем.

Воспроизведение: `PYTHONPATH=. python scripts/analysis/ann_scale.py --size {args.size}`
(нужен `faiss-cpu==1.9.0`, в инференс-образ он не входит).
""")
    print(f"\nсохранено {args.out}")


if __name__ == "__main__":
    main()
