"""Расхождение между тем, что мы меряем, и тем, что отправляем.

`scripts/extract_embeddings.py` берёт пулинг токенов и нормирует — **без BNNeck**. Все наши метрики
посчитаны этим путём. А `reid/infer/extractor.py` зовёт `model(batch)`, то есть `Embedder.forward`,
который применяет BNNeck. Значит вектор в сдаче не тот, по которому мы отчитываемся.

Скрипт меряет обе ветки на одном фолде одной моделью.

    PYTHONPATH=. python scripts/analysis/bnneck_gap.py --models final_oof_f0 pool_cls_top16_f0
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from reid.data.dataset import EvalDataset
from reid.eval.eval_local import FoldTruth, evaluate, ranking_from_embeddings
from reid.model import load, load_config, pool_tokens


@torch.inference_mode()
def both_branches(model, cache, split_csv, cfg, device, batch, workers):
    ds = EvalDataset(cache, split_csv, cfg)
    loader = DataLoader(ds, batch_size=batch, num_workers=workers)
    raw, necked = [], []
    for x in loader:
        p = pool_tokens(model.tokens(x.to(device)), model.n_prefix, cfg["pool"])
        raw.append(torch.nn.functional.normalize(p, dim=1).float().cpu().numpy())
        necked.append(torch.nn.functional.normalize(model.neck(p), dim=1).float().cpu().numpy())
    return np.concatenate(raw), np.concatenate(necked), ds.ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    truth = FoldTruth.load(args.fold)
    for name in args.models:
        path = f"artifacts/models/{name}"
        model, cfg = load(path, args.device).eval(), load_config(path)
        qr, qn, _ = both_branches(model, args.cache, f"splits/fold{args.fold}_query.csv",
                                  cfg, args.device, args.batch, args.workers)
        gr, gn, g_ids = both_branches(model, args.cache, f"splits/fold{args.fold}_gallery.csv",
                                      cfg, args.device, args.batch, args.workers)
        out = {}
        for tag, q, g in (("без BNNeck", qr, gr), ("с BNNeck", qn, gn)):
            r = ranking_from_embeddings(q, g, g_ids, truth.query_ids)
            out[tag] = evaluate(args.fold, r)["map10"]
        print(f"{name:22s} без BNNeck {out['без BNNeck']:.4f}   с BNNeck {out['с BNNeck']:.4f}   "
              f"разница {out['с BNNeck'] - out['без BNNeck']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
