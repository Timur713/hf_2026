"""Сравнение переранжировщиков на одном дампе и одной базе (A/B).

Числа из разных прогонов обучения несравнимы: у каждого своя база и свой кэш. Здесь все
верификаторы гоняются по одному дампу эмбеддера, одному фолду и одному набору весов смешивания,
поэтому разницу можно относить на сам переранжировщик.

    PYTHONPATH=. python scripts/analysis/verifier_ab.py --fold 0 \
        --emb artifacts/emb/final_oof_f0_cls_avg \
        artifacts/models/verifier_final_f0/verifier.pt:artifacts/crops \
        artifacts/models/v_hires_s392_f0/verifier.pt:artifacts/crops1024
"""
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.eval.eval_local import evaluate, ranking_from_embeddings
from reid.model.verifier import load_verifier, prep_crop

WEIGHTS = (0.05, 0.1, 0.2, 0.3, 0.5)


@torch.inference_mode()
def score(path: str, cache: str, emb_dir: str, fold: int, topk: int, device: str) -> dict:
    d = np.load(f"{emb_dir}/fold{fold}.npz")
    q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
    base = ranking_from_embeddings(d["query_emb"], d["gallery_emb"], g_ids, q_ids, k=topk)
    sim = d["query_emb"] @ d["gallery_emb"].T
    gi = {g: i for i, g in enumerate(g_ids)}

    model, cfg = load_verifier(path, device)
    man = load_manifest(cache, ("train",)).set_index("image_id")

    def prep(image_id):
        row = SimpleNamespace(image_id=image_id, **man.loc[image_id].to_dict())
        x0, y0, x1, y1 = cached_box(row, 0.0)
        return prep_crop(read_crop(cache, image_id)[:, y0:y1, x0:x1], cfg)

    need = sorted({q for q in q_ids} | {g for q in q_ids for g in base[q]})
    prepared = {i: prep(i) for i in need}

    out = {w: {} for w in WEIGHTS}
    for i, q in enumerate(q_ids):
        cand = base[q]
        xa = prepared[q].unsqueeze(0).expand(len(cand), -1, -1, -1).to(device)
        xb = torch.stack([prepared[g] for g in cand]).to(device)
        lg = model(xa, xb).float().cpu().numpy()
        idx = np.array([gi[g] for g in cand])
        for w in WEIGHTS:
            s = sim[i, idx] + w * (1 / (1 + np.exp(-lg)))
            out[w][q] = [cand[j] for j in np.argsort(-s)][:10]
    return {"база": evaluate(fold, base)["map10"],
            **{f"w={w}": evaluate(fold, out[w])["map10"] for w in WEIGHTS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="путь_к_verifier.pt:каталог_кэша")
    ap.add_argument("--emb", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    for spec in args.models:
        path, _, cache = spec.partition(":")
        res = score(path, cache or "artifacts/crops", args.emb, args.fold, args.topk, args.device)
        best = max((k for k in res if k != "база"), key=lambda k: res[k])
        print(f"\n{path}  (кэш {cache or 'artifacts/crops'})")
        print("  " + "  ".join(f"{k} {v:.4f}" for k, v in res.items()))
        print(f"  лучший {best}: {res[best]:.4f}  ({res[best] - res['база']:+.4f} к базе)", flush=True)


if __name__ == "__main__":
    main()
