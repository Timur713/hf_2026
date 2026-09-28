"""Почему правильная машина стоит в десятке, но не первой (A-091).

Узкое место не поиск: позитив попадает в top-10 у 98,7% запросов, а оракул порядка внутри
десятки даёт 0,977 против базы 0,863. Здесь провалы top-1 сравниваются с успешными запросами
по четырём признакам, и каждый контраст считается против контрольной группы — иначе получается
описание датасета, а не диагноз.

Признаки: ракурс (логарифм отношения сторон BBox), масштаб, обрезка краем кадра, хабность
галереи и симметрия (находит ли сам позитив свою машину по галерее).

`camera_id` используется только для junk-фильтра и диагностики на train-фолдах, в модель и на
инференс не попадает (правило 2). BBox — входные данные задачи.

    PYTHONPATH=. python scripts/analysis/rank_causes.py --fold 0 \
        artifacts/emb/hires392_f0_cls_top16 artifacts/emb/div_clip_f0_cls_top16 \
        artifacts/emb/div_siglip_f0_top16
"""
from __future__ import annotations

import argparse
import collections
import csv

import numpy as np

from scripts.analysis.hub_penalty import concat_parts


def gallery_similarity(dumps: list[str], fold: int, g_ids: list[str]) -> np.ndarray:
    """Косинусы галерея-галерея той же склейки, что и в поиске (D-052)."""
    G = None
    for path in dumps:
        d = np.load(f"{path}/fold{fold}.npz")
        assert d["gallery_ids"].tolist() == g_ids, f"{path}: другой порядок галереи"
        ge = d["gallery_emb"] / (np.linalg.norm(d["gallery_emb"], axis=1, keepdims=True) + 1e-12)
        ge = ge / np.sqrt((ge @ ge.T).std())
        G = ge if G is None else np.concatenate([G, ge], axis=1)
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    return G @ G.T


def load_meta(fold: int, folds_path: str = "splits/folds.csv"):
    rows = [r for r in csv.DictReader(open(folds_path)) if int(r["fold"]) == fold]
    vid = {r["image_id"]: r["vehicle_id"] for r in rows}
    cam = {r["image_id"]: r["camera_id"] for r in rows}
    box = {}
    for role in ("query", "gallery"):
        for r in csv.DictReader(open(f"splits/fold{fold}_{role}.csv")):
            box[r["image_id"]] = tuple(float(r[k]) for k in ("x", "y", "w", "h"))
    return vid, cam, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=3, help="соседей для хабности")
    ap.add_argument("--lam", type=float, default=0.5, help="вес штрафа за хабность")
    ap.add_argument("--frame", nargs=2, type=int, default=[1920, 1080])
    args = ap.parse_args()

    vid, cam, box = load_meta(args.fold)
    S, hub, q_ids, g_ids = concat_parts(args.dumps, args.fold, args.k)
    M = S - args.lam * hub[None, :]
    g_arr = np.array(g_ids)
    order = np.argsort(-M, axis=1)
    hub_of = {g: hub[j] for j, g in enumerate(g_ids)}
    gidx = {g: j for j, g in enumerate(g_ids)}
    n_pos = {q: sum(1 for g in g_ids if vid[g] == vid[q] and cam[g] != cam[q]) for q in q_ids}

    W, H = args.frame
    lar = lambda i: np.log(box[i][2] / box[i][3])            # ракурс: анфас узкий, борт широкий
    area = lambda i: box[i][2] * box[i][3]
    trunc = lambda i: (box[i][0] <= 2 or box[i][1] <= 2
                       or box[i][0] + box[i][2] >= W - 2 or box[i][3] + box[i][1] >= H - 2)

    succ, fail = [], []
    for i, q in enumerate(q_ids):
        if n_pos[q] == 0:                                     # такие query метрика исключает
            continue
        full = [(g_arr[j], j) for j in order[i]]
        win = [(g, j) for g, j in full if not (vid[g] == vid[q] and cam[g] == cam[q])][:10]
        pos = [(g, j) for g, j in win if vid[g] == vid[q]]
        neg = [(g, j) for g, j in win if vid[g] != vid[q]]
        rec = dict(q=q, i=i, lead=win[0][0], pos=pos[0][0] if pos else None,
                   topneg=neg[0][0] if neg else None)
        if vid[win[0][0]] == vid[q]:
            succ.append(rec)
        elif pos:
            fail.append(rec)

    print(f"успех top-1: {len(succ)}   провал с позитивом в top-10: {len(fail)}\n")

    d = lambda a, b: abs(lar(a) - lar(b))
    print("МЕДИАНЫ, успех против провала:")
    for name, f in (("|Δ log(w/h)| запрос ↔ правильный ответ", lambda r: d(r["q"], r["pos"])),
                    ("|Δ log(w/h)| запрос ↔ конкурент", lambda r: d(r["q"], r["lead"])),
                    ("area(правильный)/area(запрос)", lambda r: area(r["pos"]) / area(r["q"])),
                    ("hub правильного ответа, z", lambda r: hub_of[r["pos"]]),
                    ("hub конкурента, z", lambda r: hub_of[r["lead"]])):
        a = [f(r) for r in succ if r["pos"]]
        b = [f(r) for r in fail]
        print(f"  {name:40s} успех {np.median(a):7.3f}   провал {np.median(b):7.3f}")

    base = np.mean([np.mean([cam[g] == cam[r["q"]] for g in g_ids]) for r in fail])
    lead_same = np.mean([cam[r["lead"]] == cam[r["q"]] for r in fail])
    print(f"\nсцена: кадров с камеры запроса в галерее {base:.1%}, "
          f"среди конкурентов при провале {lead_same:.1%} (x{lead_same / base:.1f})")
    for name, f in (("запрос обрезан краем кадра", lambda r: trunc(r["q"])),
                    ("правильный ответ обрезан", lambda r: trunc(r["pos"]))):
        a, b = [f(r) for r in succ if r["pos"]], [f(r) for r in fail]
        print(f"  {name:40s} успех {np.mean(a):5.1%}   провал {np.mean(b):5.1%}")

    gg = gallery_similarity(args.dumps, args.fold, g_ids)    # только галерея, правило 1 не затронуто
    np.fill_diagonal(gg, -np.inf)
    print("\nсимметрия: позитив как запрос по галерее, его машина в top-5")
    for name, lst in (("успех", succ), ("провал", fail)):
        acc = []
        for r in lst:
            if not r["pos"]:
                continue
            j = gidx[r["pos"]]
            top = np.argsort(-gg[j])[:5]
            acc.append(any(vid[g_ids[t]] == vid[r["q"]] and cam[g_ids[t]] != cam[r["pos"]]
                           for t in top))
        print(f"  {name}: {np.mean(acc):.0%}")

    cause = collections.Counter()
    for r in fail:
        v = d(r["q"], r["pos"]) - d(r["q"], r["lead"])
        h = hub_of[r["lead"]] - hub_of[r["pos"]]
        if v > 0.05 and h > 0.2:
            cause["ракурс + хабность"] += 1
        elif v > 0.05:
            cause["ракурс: позитив дальше по ракурсу"] += 1
        elif h > 0.2:
            cause["хабность: конкурент в плотной зоне"] += 1
        else:
            cause["ни то ни другое: двойник по виду"] += 1
    print(f"\nразложение {len(fail)} провалов (пороги 0.05 и 0.2 условны, контрасты выше от них не зависят):")
    for k, v in cause.most_common():
        print(f"  {k:38s} {v:3d} ({v / len(fail):4.0%})")


if __name__ == "__main__":
    main()
