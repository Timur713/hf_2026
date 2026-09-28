"""Сравнивает эмбеддинги, полученные из кэша кропов и из полного кадра (путь инференса).

Показывает, насколько кэш (JPEG q95, уменьшение до 512 px) меняет вектор. Итоговая проверка по
метрике (контракт C3) — за треком B.

    PYTHONPATH=. python scripts/check_cache_consistency.py --model artifacts/models/<name> --n 64
"""
import argparse

import numpy as np
import torch
from PIL import Image

from reid.data.crop_cache import cached_model_input, load_manifest
from reid.data.preprocess import preprocess_frame
from reid.model import load, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--data-root", default="dataset")
    args = ap.parse_args()
    cfg, model = load_config(args.model), load(args.model)
    rows = load_manifest(args.cache, ("train", "query", "gallery")).sample(args.n, random_state=0)
    a, b = [], []
    for r in rows.itertuples():
        frame = np.asarray(Image.open(f"{args.data_root}/images/{r.image_id}.jpg").convert("RGB"))
        a.append(preprocess_frame(torch.from_numpy(frame.copy()).permute(2, 0, 1), (r.x, r.y, r.w, r.h), cfg))
        b.append(cached_model_input(args.cache, r, cfg))
    with torch.inference_mode():
        cos = (model(torch.stack(a)) * model(torch.stack(b))).sum(1).numpy()
    print(f"cosine(frame path, cache path) over {args.n} crops: "
          f"mean {cos.mean():.4f}  min {cos.min():.4f}  p5 {np.percentile(cos, 5):.4f}")


if __name__ == "__main__":
    main()
