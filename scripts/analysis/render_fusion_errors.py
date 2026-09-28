"""Картинки ошибок фузии: что именно мы ставим первым вместо верного ответа.

Статистика (fusion_errors.py) говорит, что ошибки связаны с расхождением ракурсов и разногласием
моделей. Но она не показывает, ЧТО модель считает похожим. Здесь три колонки: запрос, наш первый
кандидат и верный ответ с его рангом.

    PYTHONPATH=. python scripts/analysis/render_fusion_errors.py --fold 0 <дампы>
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.eval.eval_local import FoldTruth

FONTS = ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/dejavu/DejaVuSans.ttf")


def font(size):
    for p in FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--out", default="docs/errors/fusion_errors.jpg")
    ap.add_argument("--rows", type=int, default=8)
    args = ap.parse_args()

    Z = None
    for p in args.dumps:
        d = np.load(f"{p}/fold{args.fold}.npz")
        n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        s = n(d["query_emb"]) @ n(d["gallery_emb"]).T
        z = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
        Z = z if Z is None else Z + z
        q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()

    truth = FoldTruth.load(args.fold)
    gv = np.array([truth.vid[g] for g in g_ids])
    gc = np.array([truth.cam[g] for g in g_ids])
    order = np.argsort(-Z, axis=1)

    cases = []
    for i, q in enumerate(q_ids):
        junk = (gv == truth.vid[q]) & (gc == truth.cam[q])
        keep = order[i][~junk[order[i]]]
        pos = np.where(gv[keep] == truth.vid[q])[0]
        if not len(pos) or pos[0] == 0:
            continue
        rank = int(pos[0])
        cases.append((q, g_ids[keep[0]], g_ids[keep[rank]], rank + 1,
                      float(Z[i, keep[0]] - Z[i, keep[rank]])))
    # Самые обидные: верный ответ на втором месте, отрыв минимальный
    cases.sort(key=lambda c: (c[3], c[4]))
    cases = cases[: args.rows]

    man = load_manifest(args.cache, ("train",)).set_index("image_id")
    C, PAD, LAB = 200, 8, 28

    def crop_of(image_id):
        row = SimpleNamespace(image_id=image_id, **man.loc[image_id].to_dict())
        x0, y0, x1, y1 = cached_box(row, 0.0)
        arr = read_crop(args.cache, image_id)[:, y0:y1, x0:x1]
        return Image.fromarray(arr.permute(1, 2, 0).numpy()).resize((C, C))

    canvas = Image.new("RGB", (C * 3 + PAD * 4, (C + LAB) * len(cases) + PAD), "white")
    draw, f = ImageDraw.Draw(canvas), font(15)
    for r, (q, wrong, right, rank, gap) in enumerate(cases):
        y = r * (C + LAB) + PAD
        for c, image_id in enumerate((q, wrong, right)):
            canvas.paste(crop_of(image_id), (PAD + c * (C + PAD), y + LAB))
        draw.text((PAD, y + 8), f"запрос | наш ответ (неверный) | верный, ранг {rank}, "
                               f"отставание {gap:.2f}", fill="black", font=f)
        draw.rectangle([PAD + C + PAD, y + LAB, PAD + 2 * C + PAD - 1, y + LAB + C - 1],
                       outline="red", width=3)
        draw.rectangle([PAD + 2 * (C + PAD), y + LAB, PAD + 2 * (C + PAD) + C - 1, y + LAB + C - 1],
                       outline="green", width=3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out, quality=88)
    print(f"сохранено {args.out}: {len(cases)} случаев, всего ошибок первого места "
          f"{sum(1 for _ in cases)}+")


if __name__ == "__main__":
    main()
