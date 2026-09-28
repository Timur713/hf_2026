"""Смешивание косинуса и логита верификатора: сырое против выровненного по шкале.

Текущая формула `cos + 0.2·sigmoid(logit)` (reid/model/verifier.py) складывает две величины из
разных шкал с подобранным коэффициентом. Ровно этот дефект D-052 нашёл в ансамблях: там косинусы
двух моделей отличались разбросом в 13.8 раза, и «ансамбль» оказывался одной моделью.

Здесь сравниваются способы соединения внутри списка кандидатов одного query:
  raw    — как сейчас;
  z      — обе величины стандартизуются по списку кандидатов, затем складываются;
  rank   — обратная сумма рангов (RRF), шкалы не важны вовсе.

    PYTHONPATH=. python scripts/analysis/mix_scales.py --emb <дамп> --verifier <путь> --cache <кэш>
"""
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from reid.data.crop_cache import cached_box, load_manifest, read_crop
from reid.eval.eval_local import FoldTruth, evaluate
from reid.model.verifier import load_verifier, prep_crop


def z(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", required=True, help="один дамп или несколько для фузии")
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--cache", default="artifacts/crops1024")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    S = None
    for p in args.emb:
        d = np.load(f"{p}/fold{args.fold}.npz")
        n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        s = n(d["query_emb"]) @ n(d["gallery_emb"]).T
        s = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
        S = s if S is None else S + s
        q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()

    order = np.argsort(-S, axis=1)
    base = {q: [g_ids[j] for j in order[i][:args.topk]] for i, q in enumerate(q_ids)}
    # Ровно 10 кандидатов, как в сдаче: evaluate() выбрасывает двойника из поданного списка и
    # добирает хвост, поэтому длинный список даёт завышенное число. Для связи со старыми записями
    # печатаем и мягкий вариант.
    strict = {q: v[:10] for q, v in base.items()}
    lenient = {q: [g_ids[j] for j in order[i][:60]] for i, q in enumerate(q_ids)}
    print(f"база ({len(args.emb)} модел.): {evaluate(args.fold, strict)['map10']:.4f}"
          f"  (мягкий протокол с добором: {evaluate(args.fold, lenient)['map10']:.4f})", flush=True)

    model, cfg = load_verifier(args.verifier, args.device)
    man = load_manifest(args.cache, ("train",)).set_index("image_id")

    def prep(image_id):
        row = SimpleNamespace(image_id=image_id, **man.loc[image_id].to_dict())
        x0, y0, x1, y1 = cached_box(row, 0.0)
        return prep_crop(read_crop(args.cache, image_id)[:, y0:y1, x0:x1], cfg)

    cand = {q: base[q][: args.topk] for q in q_ids}
    prepared = {i: prep(i) for i in sorted(set(q_ids) | {g for q in q_ids for g in cand[q]})}
    gi = {g: i for i, g in enumerate(g_ids)}

    logits, sims = {}, {}
    with torch.inference_mode():
        for i, q in enumerate(q_ids):
            c = cand[q]
            xa = prepared[q].unsqueeze(0).expand(len(c), -1, -1, -1).to(args.device)
            xb = torch.stack([prepared[g] for g in c]).to(args.device)
            logits[q] = model(xa, xb).float().cpu().numpy()
            sims[q] = S[i, [gi[g] for g in c]]

    def rank_of(v):
        return np.argsort(np.argsort(-v)).astype(np.float64)

    results = {}
    for w in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
        raw, zz = {}, {}
        for q in q_ids:
            c = cand[q]
            r = sims[q] + w * (1 / (1 + np.exp(-logits[q])))
            raw[q] = [c[j] for j in np.argsort(-r)][:10]
            r2 = z(sims[q]) + w * z(logits[q])
            zz[q] = [c[j] for j in np.argsort(-r2)][:10]
        results[f"raw w={w}"] = evaluate(args.fold, raw)["map10"]
        results[f"z   w={w}"] = evaluate(args.fold, zz)["map10"]
    for k in (10, 30, 60):
        rr = {}
        for q in q_ids:
            c = cand[q]
            r = 1 / (k + rank_of(sims[q])) + 1 / (k + rank_of(logits[q]))
            rr[q] = [c[j] for j in np.argsort(-r)][:10]
        results[f"RRF k={k}"] = evaluate(args.fold, rr)["map10"]

    for name, v in sorted(results.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {name:12s} {v:.4f}")


if __name__ == "__main__":
    main()
