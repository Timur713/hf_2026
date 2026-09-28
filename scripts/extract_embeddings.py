"""Эмбеддинги из кэша кропов → дампы по контракту C4 (fold{k}.npz и test.npz).

Для модели без дообучения один проход даёт эмбеддинги всех кадров. Для дообученной модели дамп
фолда k допустим только от модели, обученной без фолда k (--folds задаёт, какие фолды писать).
Если указать несколько --pools, один проход сохраняет дампы для каждого варианта пулинга.

    PYTHONPATH=. python scripts/extract_embeddings.py --model artifacts/models/<name> --pools cls,cls_avg
"""
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
        # SimpleNamespace, а не itertuples: строки должны пересылаться в воркеры DataLoader (spawn)
        self.rows = [SimpleNamespace(**r) for r in manifest.to_dict("records")]
        self.cache_dir, self.cfg = cache_dir, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return cached_model_input(self.cache_dir, self.rows[i], self.cfg)


def dump_name(cfg, pool):
    base = cfg["name"]
    suffix = f"_{cfg['pool']}"
    return (base[: -len(suffix)] if base.endswith(suffix) else base) + f"_{pool}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pools", help="через запятую; по умолчанию пулинг из конфига модели")
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "dataset"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta-flip", action="store_true",
                    help="усреднять эмбеддинг кадра и его зеркала; стоит второго forward")
    ap.add_argument("--global-only", action="store_true",
                    help="у модели с полосами взять только глобальную ветку: полосы тогда были "
                         "вспомогательным сигналом при обучении, а не частью эмбеддинга")
    ap.add_argument("--input", type=int, default=0,
                    help="прогнать на другом входе, чем обучалась модель (вторая ступень, D-056)")
    ap.add_argument("--suffix", default="",
                    help="приписка к имени дампа: вариант не должен затирать основной дамп")
    args = ap.parse_args()

    cfg = load_config(args.model)
    pools = args.pools.split(",") if args.pools else [cfg["pool"]]
    if args.input:
        cfg = dict(cfg, input_size=[args.input, args.input])
    model = load(args.model, args.device, input_size=cfg["input_size"])
    man = load_manifest(args.cache, ("train", "query", "gallery"))
    loader = DataLoader(CacheDataset(args.cache, man, cfg), batch_size=args.batch, num_workers=args.workers)

    feats = {p: [] for p in pools}
    t0 = time.time()
    with torch.inference_mode():
        for i, x in enumerate(loader):
            x = x.to(args.device)
            if getattr(model, "parts", 0) and not args.global_only:
                # У модели с полосами эмбеддинг — склейка веток; пулинг применяется к глобальной
                e = torch.nn.functional.normalize(model.features(x), dim=1)
                for p in pools:
                    feats[p].append(e.float().cpu().numpy())
                continue
            tok = model.tokens(x)
            tok_flip = model.tokens(torch.flip(x, dims=[-1])) if args.tta_flip else None
            for p in pools:
                e = torch.nn.functional.normalize(pool_tokens(tok, model.n_prefix, p), dim=1)
                if tok_flip is not None:
                    e = torch.nn.functional.normalize(
                        e + torch.nn.functional.normalize(pool_tokens(tok_flip, model.n_prefix, p), dim=1), dim=1)
                feats[p].append(e.float().cpu().numpy())
            if i % 50 == 0:
                done = (i + 1) * args.batch
                print(f"{min(done, len(man))}/{len(man)}  {done / (time.time() - t0):.1f} img/s", flush=True)
    index = {iid: j for j, iid in enumerate(man.image_id)}

    def rows(ids):
        return [index[i] for i in ids]

    for p in pools:
        emb = np.concatenate(feats[p]).astype(np.float32)
        out = os.path.join("artifacts/emb", dump_name(cfg, p) + args.suffix)
        os.makedirs(out, exist_ok=True)
        for k in map(int, args.folds.split(",")) if args.folds else []:
            q = pd.read_csv(f"splits/fold{k}_query.csv", dtype={"image_id": str}).image_id.tolist()
            g = pd.read_csv(f"splits/fold{k}_gallery.csv", dtype={"image_id": str}).image_id.tolist()
            np.savez(os.path.join(out, f"fold{k}.npz"), query_ids=np.array(q), gallery_ids=np.array(g),
                     query_emb=emb[rows(q)], gallery_emb=emb[rows(g)])
        if not args.no_test:
            q = pd.read_csv(os.path.join(args.data_root, "test_query.csv"), dtype={"image_id": str}).image_id.tolist()
            g = pd.read_csv(os.path.join(args.data_root, "test_gallery.csv"), dtype={"image_id": str}).image_id.tolist()
            np.savez(os.path.join(out, "test.npz"), query_ids=np.array(q), gallery_ids=np.array(g),
                     query_emb=emb[rows(q)], gallery_emb=emb[rows(g)])
        print(f"saved dumps → {out}  (D={emb.shape[1]}, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
