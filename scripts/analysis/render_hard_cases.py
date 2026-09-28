"""Трудные query глазами: что именно должен различить переранжировщик.

Берём группу, в которой лежит две трети всего запаса (`fusion_headroom.py`): единственный позитив
стоит на 3–10 месте выдачи фузии. Рисуем ряд «query — кандидаты по порядку», позитив обводим
зелёным, двойника (та же машина, та же камера) — синим. Кропы режутся из **исходных кадров**, а не
из кэша: вопрос ровно в том, видна ли разница в принципе и при каком разрешении.

Подпись над каждым кропом — размер объекта в пикселях исходного кадра. Если различие человеку
видно на 600 px, а верификатор смотрит на 168, то дело не в лоссе.

    PYTHONPATH=. python scripts/analysis/render_hard_cases.py --fold 0 --out docs/errors/hard_f0.jpg \\
        --emb artifacts/emb/pool_cls_top16_f0_cls_top16 artifacts/emb/div_clip_f0_cls_top16 \\
              artifacts/emb/div_siglip_f0_top16
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from reid.eval.eval_local import FoldTruth
from scripts.analysis.fusion_headroom import fused_similarity

TILE_H = 260          # высота кропа в итоговой картинке
PAD = 8
LABEL_H = 22


def crop(images_dir: str, image_id: str, box: pd.Series, pad: float = 0.08) -> Image.Image:
    img = Image.open(os.path.join(images_dir, f"{image_id}.jpg")).convert("RGB")
    x, y, w, h = float(box.x), float(box.y), float(box.w), float(box.h)
    dx, dy = w * pad, h * pad
    return img.crop((max(0, int(x - dx)), max(0, int(y - dy)),
                     min(img.width, int(x + w + dx)), min(img.height, int(y + h + dy))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "dataset"))
    ap.add_argument("--rows", type=int, default=6, help="сколько query нарисовать")
    ap.add_argument("--skip", type=int, default=0, help="пропустить первые N трудных query")
    ap.add_argument("--max-cand", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    S, q_ids, g_ids = fused_similarity(args.emb, args.fold)
    truth = FoldTruth.load(args.fold)
    n_pos = truth.n_pos()
    order = np.argsort(-S, axis=1)
    boxes = pd.read_csv(os.path.join(args.data_root, "train.csv"), dtype={"image_id": str}) \
        .set_index("image_id")
    images_dir = os.path.join(args.data_root, "images")

    hard = []
    for i, q in enumerate(q_ids):
        if n_pos[q] != 1:
            continue
        ranked = [g_ids[j] for j in order[i]]
        rank = next((r for r, g in enumerate(ranked, 1) if truth.is_pos(q, g)), None)
        if rank is not None and 3 <= rank <= 10:
            hard.append((q, ranked, rank))
    print(f"трудных query (n_pos=1, позитив на 3–10 месте): {len(hard)}")

    rows = hard[args.skip:args.skip + args.rows]
    tiles = []
    for q, ranked, rank in rows:
        shown = ranked[: max(args.max_cand, rank)][: max(rank, 1)] if rank > args.max_cand else ranked[: args.max_cand]
        row = [("query", q, "black")]
        for r, g in enumerate(shown, 1):
            color = "#16a34a" if truth.is_pos(q, g) else ("#2563eb" if truth.is_junk(q, g) else "#dc2626")
            row.append((f"#{r}", g, color))
        tiles.append(row)

    sizes = {g: crop(images_dir, g, boxes.loc[g]).size for row in tiles for _, g, _ in row}
    width = max(sum(int(TILE_H * sizes[g][0] / sizes[g][1]) + PAD for _, g, _ in row)
                for row in tiles)
    height = len(tiles) * (TILE_H + LABEL_H + PAD)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for row in tiles:
        x = 0
        for label, image_id, color in row:
            c = crop(images_dir, image_id, boxes.loc[image_id])
            w_px, h_px = int(boxes.loc[image_id].w), int(boxes.loc[image_id].h)
            tile = c.resize((int(TILE_H * c.width / c.height), TILE_H), Image.LANCZOS)
            canvas.paste(tile, (x, y + LABEL_H))
            draw.rectangle([x, y + LABEL_H, x + tile.width - 1, y + LABEL_H + TILE_H - 1],
                           outline=color, width=4)
            draw.text((x + 4, y + 4), f"{label}  {w_px}x{h_px}px", fill=color)
            x += tile.width + PAD
        y += TILE_H + LABEL_H + PAD

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    canvas.save(args.out, quality=92)
    print(f"сохранено {args.out}  ({canvas.width}x{canvas.height})")
    print("зелёный — настоящий позитив, синий — двойник с той же камеры, красный — чужая машина")


if __name__ == "__main__":
    main()
