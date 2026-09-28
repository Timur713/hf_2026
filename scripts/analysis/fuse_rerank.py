"""Складываются ли фузия моделей и парное переранжирование.

Фузия поднимает базовый порядок, переранжировщик правит верх выдачи. Вопрос в том, не
«съедают» ли они один и тот же запас: если переранжировщик чинил ровно те ошибки, которые фузия
убрала сама, суммарный прирост будет меньше суммы слагаемых.

    PYTHONPATH=. python scripts/analysis/fuse_rerank.py --fold 0 \
        --dumps artifacts/emb/siglip2_f0_full_avg artifacts/emb/pool_cls_top16_f0_cls_top16 \
        --verifier artifacts/models/v_hires_s392_f0/verifier.pt --cache artifacts/crops1024
"""
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.eval.eval_local import evaluate
from reid.model.verifier import load_verifier, prep_crop

WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)


def zrows(s):
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--cache", default="artifacts/crops1024")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    z = None
    for p in args.dumps:
        d = np.load(f"{p}/fold{args.fold}.npz")
        n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        s = zrows(n(d["query_emb"]) @ n(d["gallery_emb"]).T)
        z = s if z is None else z + s
        q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()

    order = np.argsort(-z, axis=1)
    base = {q: [g_ids[j] for j in order[i][:60]] for i, q in enumerate(q_ids)}
    print(f"фузия {len(args.dumps)} моделей, без переранжирования: {evaluate(args.fold, base)['map10']:.4f}",
          flush=True)

    model, cfg = load_verifier(args.verifier, args.device)
    man = load_manifest(args.cache, ("train",)).set_index("image_id")

    def prep(image_id):
        row = SimpleNamespace(image_id=image_id, **man.loc[image_id].to_dict())
        x0, y0, x1, y1 = cached_box(row, 0.0)
        return prep_crop(read_crop(args.cache, image_id)[:, y0:y1, x0:x1], cfg)

    cand = {q: base[q][: args.topk] for q in q_ids}
    need = sorted(set(q_ids) | {g for q in q_ids for g in cand[q]})
    prepared = {i: prep(i) for i in need}
    gi = {g: i for i, g in enumerate(g_ids)}

    out = {w: {} for w in WEIGHTS}
    with torch.inference_mode():
        for i, q in enumerate(q_ids):
            c = cand[q]
            xa = prepared[q].unsqueeze(0).expand(len(c), -1, -1, -1).to(args.device)
            xb = torch.stack([prepared[g] for g in c]).to(args.device)
            lg = model(xa, xb).float().cpu().numpy()
            idx = np.array([gi[g] for g in c])
            for w in WEIGHTS:
                s = z[i, idx] + w * (1 / (1 + np.exp(-lg)))
                out[w][q] = [c[j] for j in np.argsort(-s)][:10]

    res = {w: evaluate(args.fold, out[w])["map10"] for w in WEIGHTS}
    for w, v in res.items():
        print(f"  вес {w}: {v:.4f}")
    best = max(res, key=res.get)
    print(f"\nлучший вес {best}: {res[best]:.4f}  ({res[best] - res[0.0]:+.4f} к фузии без переранжирования)")


if __name__ == "__main__":
    main()
