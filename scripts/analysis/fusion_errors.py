"""Почему фузия не ставит верный ответ первым: разбор оставшихся ошибок.

После фузии верный ответ лежит в top-10 у 98,4% запросов, но первым — только у 81%. Весь
оставшийся запас (+0,12 до потолка 0,97) — это порядок внутри десятки. Скрипт ищет, что объединяет
случаи, где порядок неверен, и рисует примеры.

Считаемые признаки не требуют разметки и доступны на инференсе, поэтому найденная зависимость
сразу подсказывает, чем чинить:
  * размер BBox — мелкая машина, мало деталей;
  * яркость кропа — ночь;
  * отношение сторон — грубая замена ракурсу: анфас узкий, борт широкий;
  * согласие моделей — насколько пять мнений расходятся.

    PYTHONPATH=. python scripts/analysis/fusion_errors.py --fold 0 <дампы>
"""
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd

from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.eval.eval_local import FoldTruth


def zrows(s):
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--out", default="docs/errors/fusion_errors.jpg")
    ap.add_argument("--rows", type=int, default=6)
    args = ap.parse_args()

    per_model, q_ids, g_ids = [], None, None
    for p in args.dumps:
        d = np.load(f"{p}/fold{args.fold}.npz")
        n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        per_model.append(zrows(n(d["query_emb"]) @ n(d["gallery_emb"]).T))
        q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
    A = np.stack(per_model)
    Z = A.sum(0)
    truth = FoldTruth.load(args.fold)

    man = load_manifest(args.cache, ("train",)).set_index("image_id")
    meta = {}
    for image_id in set(q_ids) | set(g_ids):
        row = SimpleNamespace(image_id=image_id, **man.loc[image_id].to_dict())
        x0, y0, x1, y1 = cached_box(row, 0.0)
        crop = read_crop(args.cache, image_id)[:, y0:y1, x0:x1]
        meta[image_id] = dict(long_side=max(row.w, row.h), aspect=row.w / max(row.h, 1),
                              bright=float(crop.float().mean()))

    gv = np.array([truth.vid[g] for g in g_ids])
    gc = np.array([truth.cam[g] for g in g_ids])
    order = np.argsort(-Z, axis=1)

    rows = []
    for i, q in enumerate(q_ids):
        junk = (gv == truth.vid[q]) & (gc == truth.cam[q])
        keep = order[i][~junk[order[i]]]
        pos = np.where(gv[keep] == truth.vid[q])[0]
        if not len(pos):
            continue
        rank = int(pos[0])
        top, right = g_ids[keep[0]], g_ids[keep[rank]]
        # согласие моделей: среднее попарное совпадение их личных top-1
        own_top = [g_ids[int(np.argmax(np.where(junk, -1e9, A[m, i])))] for m in range(len(A))]
        agree = max(own_top.count(x) for x in set(own_top)) / len(A)
        rows.append(dict(q=q, rank=rank, top=top, right=right, agree=agree,
                         gap=float(Z[i, keep[0]] - Z[i, keep[rank]]),
                         long_side=meta[q]["long_side"], bright=meta[q]["bright"],
                         aspect_diff=abs(np.log(meta[q]["aspect"] / meta[right]["aspect"]))))
    df = pd.DataFrame(rows)
    ok, bad = df[df["rank"] == 0], df[df["rank"] > 0]
    print(f"фолд {args.fold}: верных первых {len(ok)}, неверных {len(bad)} "
          f"(из них в top-10: {(bad['rank'] < 10).sum()})\n")
    print(f"{'признак':28s} {'верные':>9} {'ошибки':>9}  вывод")
    for name, col, hint in (("длинная сторона BBox, px", "long_side", "мелкие хуже"),
                            ("яркость кропа 0-255", "bright", "тёмные хуже"),
                            ("расхождение ракурсов", "aspect_diff", "смена ракурса хуже"),
                            ("согласие пяти моделей", "agree", "разнобой = ошибка")):
        a, b = ok[col].mean(), bad[col].mean()
        mark = hint if abs(a - b) / (abs(a) + 1e-9) > 0.05 else "разницы нет"
        print(f"{name:28s} {a:9.3f} {b:9.3f}  {mark}")

    # доля ошибок по децилям согласия — самый действенный признак, если он работает
    print()
    for lo, hi in ((0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        sub = df[(df.agree >= lo) & (df.agree < hi)]
        if len(sub):
            print(f"согласие {lo:.1f}–{hi:.1f}: запросов {len(sub):4d}, ошибок первого места "
                  f"{(sub['rank'] > 0).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
