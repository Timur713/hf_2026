"""Эмбеддинги из промежуточных слоёв backbone — разнородность без единого лишнего forward.

Мы всегда берём признаки последнего слоя. Слои середины кодируют более локальные и текстурные
вещи, то есть дают **другое представление того же кадра**, а не ту же модель под другим углом —
в отличие от смены пулинга, которая просто дублирует вес backbone в фузии и вредит.

Главное: все слои получаются из одного и того же прохода, поэтому в замер скорости (§7 ТЗ)
добавляется только конкатенация.

    PYTHONPATH=. python scripts/extract_layers.py --model artifacts/models/pool_cls_top16_f0 \\
        --layers 5,7,9 --cache artifacts/crops --folds 0 --no-test
"""
from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from reid.data.crop_cache import cached_model_input, load_manifest
from reid.model import load, load_config, pool_tokens


class CacheDataset(Dataset):
    def __init__(self, cache_dir, manifest, cfg):
        self.rows = [SimpleNamespace(**r) for r in manifest.to_dict("records")]
        self.cache_dir, self.cfg = cache_dir, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return cached_model_input(self.cache_dir, self.rows[i], self.cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="5,7,9", help="номера блоков через запятую, от нуля")
    ap.add_argument("--pool", default="", help="по умолчанию пулинг модели")
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.model)
    pool = args.pool or cfg["pool"]
    layers = [int(x) for x in args.layers.split(",")]
    model = load(args.model, args.device)
    man = load_manifest(args.cache, ("train", "query", "gallery"))
    loader = DataLoader(CacheDataset(args.cache, man, cfg), batch_size=args.batch,
                        num_workers=args.workers)

    feats = {l: [] for l in layers}
    t0 = time.time()
    with torch.inference_mode():
        for x in loader:
            outs = model.backbone.get_intermediate_layers(
                x.to(args.device), n=layers, reshape=False, return_prefix_tokens=True, norm=True)
            for l, (patches, prefix) in zip(layers, outs):
                tok = torch.cat([prefix, patches], dim=1)
                e = torch.nn.functional.normalize(pool_tokens(tok, prefix.shape[1], pool), dim=1)
                feats[l].append(e.float().cpu().numpy())

    folds = pd.read_csv("splits/folds.csv", dtype={"image_id": str})
    ids = man.image_id.tolist()
    pos = {i: k for k, i in enumerate(ids)}
    for l in layers:
        emb = np.concatenate(feats[l]).astype(np.float32)
        out = os.path.join("artifacts/emb", f"{cfg['name']}_L{l}_{pool}")
        os.makedirs(out, exist_ok=True)
        for k in [int(x) for x in args.folds.split(",")]:
            f = folds[folds.fold == k]
            q = f[f.role == "query"].image_id.tolist()
            g = f[f.role == "gallery"].image_id.tolist()
            np.savez(os.path.join(out, f"fold{k}.npz"), query_ids=np.array(q),
                     gallery_ids=np.array(g),
                     query_emb=emb[[pos[i] for i in q]], gallery_emb=emb[[pos[i] for i in g]])
        print(f"saved {out}  (D={emb.shape[1]}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
