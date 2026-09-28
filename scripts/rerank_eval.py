"""Сравнивает способы переранжирования top-K на фолде: база, k-reciprocal, парный re-ranker.

k-reciprocal считается строго по одному query: используются сходства этого query с gallery и
предвычисленные сходства gallery–gallery. Другие query не участвуют — правило независимости
запросов соблюдено (AGENTS.md, правило 1).

    PYTHONPATH=. python scripts/rerank_eval.py --emb artifacts/emb/<name> --fold 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from reid.eval.eval_local import evaluate, ranking_from_embeddings


def k_reciprocal_order(sim_q: np.ndarray, sim_gg: np.ndarray, topk: int, k1: int = 20, k2: int = 6,
                       lam: float = 0.3) -> np.ndarray:
    """Порядок кандидатов для ОДНОГО query по расстоянию Жаккара k-reciprocal.

    sim_q  — сходства query со всей gallery [G];
    sim_gg — сходства gallery–gallery [G, G] (статическая gallery, считается заранее).
    """
    cand = np.argsort(-sim_q)[:topk]
    nn_q = set(np.argsort(-sim_q)[:k1])

    def recip_set(g: int) -> set:
        nn_g = np.argsort(-sim_gg[g])[:k1]
        out = {int(j) for j in nn_g if g in np.argsort(-sim_gg[j])[:k1]}
        out.add(int(g))
        # расширение: добавляем соседей соседей, если они сильно пересекаются
        for j in list(out):
            nn_j = {int(x) for x in np.argsort(-sim_gg[j])[:k1 // 2]}
            if len(nn_j & out) > 2 / 3 * len(nn_j):
                out |= nn_j
        return out

    jacc = np.empty(len(cand), dtype=np.float32)
    for i, g in enumerate(cand):
        rg = recip_set(int(g))
        inter, union = len(nn_q & rg), len(nn_q | rg)
        jacc[i] = 1.0 - inter / max(union, 1)
    final = lam * (1.0 - sim_q[cand]) + (1.0 - lam) * jacc
    return cand[np.argsort(final)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--reranker", help="путь к artifacts/models/<name>/reranker.pt")
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--correspondence", metavar="MODEL",
                    help="переранжирование по соответствию патчей эмбеддера (без обучения)")
    ap.add_argument("--top-share", type=float, default=0.5,
                    help="какую долю лучших патчей усреднять в оценке соответствия")
    args = ap.parse_args()

    d = np.load(f"{args.emb}/fold{args.fold}.npz")
    q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
    q, g = d["query_emb"], d["gallery_emb"]
    sim, sim_gg = q @ g.T, g @ g.T

    base = ranking_from_embeddings(q, g, g_ids, q_ids, k=10)
    print(f"база (косинус)            : mAP@10 {evaluate(args.fold, base)['map10']:.4f}")

    for k1, lam in ((20, 0.3), (20, 0.5), (10, 0.3), (30, 0.3)):
        ranked = {qid: [g_ids[j] for j in k_reciprocal_order(sim[i], sim_gg, args.topk, k1=k1, lam=lam)][:10]
                  for i, qid in enumerate(q_ids)}
        print(f"k-reciprocal k1={k1:<3} lam={lam}: mAP@10 {evaluate(args.fold, ranked)['map10']:.4f}")

    if args.correspondence:
        import torch
        from reid.model import load, load_config
        from reid.model.interpret import pair_scores, tokens_for
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg = load(args.correspondence, dev), load_config(args.correspondence)
        cand10 = ranking_from_embeddings(q, g, g_ids, q_ids, k=10)
        gi = {x: i for i, x in enumerate(g_ids)}
        tok_q = tokens_for(model, args.cache, q_ids, cfg, dev)
        tok_g = tokens_for(model, args.cache, g_ids, cfg, dev)
        print(f"патч-токены посчитаны: query {tuple(tok_q.shape)}, gallery {tuple(tok_g.shape)}",
              flush=True)
        corr = {}
        for i, qid in enumerate(q_ids):
            idx = [gi[x] for x in cand10[qid]]
            corr[qid] = pair_scores(tok_q[i].float().to(dev), tok_g[idx].float().to(dev),
                                    args.top_share).cpu().numpy()
        for w in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            ranked = {}
            for i, qid in enumerate(q_ids):
                idx = np.array([gi[x] for x in cand10[qid]])
                score = sim[i, idx] + w * corr[qid]
                ranked[qid] = [cand10[qid][j] for j in np.argsort(-score)]
            print(f"соответствие патчей, вес {w:<4}: mAP@10 {evaluate(args.fold, ranked)['map10']:.4f}")

    if args.reranker:
        import torch.nn.functional as F
        from reid.data.dataset import EvalDataset
        from reid.model import load, load_config
        from reid.model.reranker import PairReranker
        from scripts.train_reranker import compute_tokens
        ck = torch.load(args.reranker, map_location="cpu")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        emb_model = load(ck["embedder"], dev)
        cfg = load_config(ck["embedder"])
        rr = PairReranker(ck["dim"], layers=ck["layers"]).to(dev).eval()
        rr.load_state_dict(ck["state"])
        tok_q, _ = compute_tokens(emb_model, args.cache, EvalDataset(args.cache, f"splits/fold{args.fold}_query.csv", cfg), cfg, dev)
        tok_g, _ = compute_tokens(emb_model, args.cache, EvalDataset(args.cache, f"splits/fold{args.fold}_gallery.csv", cfg), cfg, dev)
        gi = {x: i for i, x in enumerate(g_ids)}
        cand10 = ranking_from_embeddings(q, g, g_ids, q_ids, k=10)
        logits = {}
        for i, qid in enumerate(q_ids):
            idx = torch.tensor([gi[x] for x in cand10[qid]])
            logits[qid] = (idx, rr.score_pairs(tok_q[i].float().to(dev), tok_g[idx].float().to(dev)).cpu())
        for w in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            ranked = {}
            for i, qid in enumerate(q_ids):
                idx, lg = logits[qid]
                score = torch.tensor(sim[i, idx.numpy()]) + w * torch.sigmoid(lg)
                ranked[qid] = [cand10[qid][j] for j in torch.argsort(-score).tolist()]
            print(f"парный re-ranker, вес {w:<4}: mAP@10 {evaluate(args.fold, ranked)['map10']:.4f}")


if __name__ == "__main__":
    main()
