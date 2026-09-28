"""Собирает кэш кропов для train и открытого теста (задача A1.3).

Тестовые кропы нужны только для прогона модели (эмбеддинги test.npz). Обучение берёт из манифеста
только split == "train" (reid.data.crop_cache.load_manifest).

Разрешение кэша задаётся `--max-side`. По умолчанию 512 — этого хватает эмбеддеру при входе 280.
Переранжировщику нужен отдельный кэш покрупнее: медианная машина в кадре занимает 672 px, и при 512
половина деталей теряется ещё до модели (A-049).

    PYTHONPATH=. DATA_ROOT=./dataset python scripts/build_crop_cache.py --out artifacts/crops --workers 12
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image

from reid.data.crop_cache import CACHE_MAX_SIDE, CACHE_PAD
from reid.data.preprocess import clip_bbox, expand_box


def work(task):
    image_id, x, y, w, h, split, data_root, out_dir, max_side = task
    img = Image.open(os.path.join(data_root, "images", f"{image_id}.jpg"))
    fw, fh = img.size
    img = img.convert("RGB")
    cx0, cy0, cx1, cy1 = expand_box(clip_bbox(x, y, w, h, fw, fh), CACHE_PAD, fw, fh)
    crop = img.crop((cx0, cy0, cx1, cy1))
    scale = min(1.0, max_side / max(crop.size))
    if scale < 1.0:
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                           Image.BILINEAR)
    scale_x = crop.width / (cx1 - cx0)  # фактический масштаб после округления размеров
    crop.save(os.path.join(out_dir, "img", f"{image_id}.jpg"), quality=95)
    return dict(image_id=image_id, split=split, x=x, y=y, w=w, h=h, frame_w=fw, frame_h=fh,
                ctx_x0=cx0, ctx_y0=cy0, ctx_x1=cx1, ctx_y1=cy1, scale=scale_x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "dataset"))
    ap.add_argument("--out", default="artifacts/crops")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max-side", type=int, default=CACHE_MAX_SIDE,
                    help="длинная сторона кэшированного кропа в пикселях")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "img"), exist_ok=True)

    parts = []
    for name, split in (("train.csv", "train"), ("test_query.csv", "query"), ("test_gallery.csv", "gallery")):
        path = os.path.join(args.data_root, name)
        if not os.path.exists(path):
            # На сервере обучения тестовых кадров может не быть намеренно: так нарушить правило 4
            # невозможно по построению. Кэш собирается только из того, что есть.
            print(f"пропускаю {name}: нет файла")
            continue
        df = pd.read_csv(path, dtype={"image_id": str})
        parts.append(df[["image_id", "x", "y", "w", "h"]].assign(split=split))
    if not parts:
        raise SystemExit(f"в {args.data_root} нет ни одного CSV с разметкой")
    df = pd.concat(parts, ignore_index=True)
    tasks = [(*r, args.data_root, args.out, args.max_side) for r in df.itertuples(index=False)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(work, tasks, chunksize=32))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    print(f"cached {len(rows)} crops → {args.out} (длинная сторона ≤ {args.max_side} px)")


if __name__ == "__main__":
    main()
