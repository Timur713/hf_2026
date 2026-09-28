"""Обучение парного re-ranker'а поверх обученного эмбеддера (задача A3.3, решение D-028).

Эмбеддер заморожен. Из него один раз берутся токены всех кропов (с усреднением 2×2, чтобы
последовательность была короче), дальше учится маленький трансформер, сравнивающий два кадра.

Негативы — не случайные: для каждого кадра берутся ближайшие по косинусу кадры ДРУГИХ машин,
то есть ровно те пары, которые эмбеддер путает.

    PYTHONPATH=. python scripts/train_reranker.py --model artifacts/models/<name> --val-fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reid.data.dataset import EvalDataset, TrainDataset
from reid.eval.eval_local import FoldTruth, evaluate, ranking_from_embeddings
from reid.model import load, load_config
from reid.model.reranker import PairReranker, rerank


@torch.inference_mode()
def compute_tokens(model, cache, rows_csv_or_ds, cfg, device, batch=64, workers=6, pool=2):
    """Токены [N, T, D] в fp16 плюс эмбеддинги [N, D]. Патчи усредняются окном pool×pool."""
    loader = DataLoader(rows_csv_or_ds, batch_size=batch, num_workers=workers)
    toks, embs = [], []
    for x in loader:
        x = (x[0] if isinstance(x, (list, tuple)) else x).to(device)  # TrainDataset отдаёт (x, label, cam)
        t = model.tokens(x)
        prefix, patches = t[:, :model.n_prefix], t[:, model.n_prefix:]
        g = int(patches.shape[1] ** 0.5)
        p = patches.transpose(1, 2).reshape(patches.size(0), -1, g, g)
        p = F.avg_pool2d(p, pool).flatten(2).transpose(1, 2)
        toks.append(torch.cat([prefix, p], dim=1).half().cpu())
        embs.append(model(x).float().cpu())
    return torch.cat(toks), torch.cat(embs)


class PairSampler:
    """Готовит пары (якорь, кандидат, метка) с трудными негативами из top-K эмбеддера."""

    def __init__(self, labels, cams, emb, neg_k=15, seed=2026):
        self.labels, self.cams, self.rng = np.array(labels), np.array(cams), random.Random(seed)
        sim = (emb @ emb.T).numpy()
        np.fill_diagonal(sim, -2)
        order = np.argsort(-sim, axis=1)[:, :neg_k * 3]
        self.negatives = [[j for j in row if self.labels[j] != self.labels[i]][:neg_k]
                          for i, row in enumerate(order)]
        by_label = {}
        for i, lab in enumerate(self.labels):
            by_label.setdefault(lab, []).append(i)
        # позитивы предпочитаем кросс-камерные: именно такие пары оценивает жюри
        self.positives = []
        for i, lab in enumerate(self.labels):
            cross = [j for j in by_label[lab] if j != i and self.cams[j] != self.cams[i]]
            same = [j for j in by_label[lab] if j != i and self.cams[j] == self.cams[i]]
            self.positives.append(cross or same)

    def batch(self, size):
        a, b, y = [], [], []
        for _ in range(size // 2):
            i = self.rng.randrange(len(self.labels))
            if self.positives[i]:
                a.append(i); b.append(self.rng.choice(self.positives[i])); y.append(1.0)
            if self.negatives[i]:
                a.append(i); b.append(self.rng.choice(self.negatives[i])); y.append(0.0)
        return torch.tensor(a), torch.tensor(b), torch.tensor(y)


def evaluate_rerank(model_rr, tok_q, tok_g, q_emb, g_emb, q_ids, g_ids, fold, device, topk, weight):
    base = ranking_from_embeddings(q_emb.numpy(), g_emb.numpy(), g_ids, q_ids, k=topk)
    sim = (q_emb @ g_emb.T)
    g_index = {g: i for i, g in enumerate(g_ids)}
    ranked = {}
    model_rr.eval()
    for i, q in enumerate(q_ids):
        cand = base[q]
        idx = torch.tensor([g_index[g] for g in cand])
        logits = model_rr.score_pairs(tok_q[i].float().to(device), tok_g[idx].float().to(device)).cpu()
        score = rerank(sim[i, idx], logits, weight)
        ranked[q] = [cand[j] for j in torch.argsort(-score).tolist()][:10]
    model_rr.train()
    return evaluate(fold, ranked)["map10"], evaluate(fold, {q: v[:10] for q, v in base.items()})["map10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--cache", default="artifacts/crops")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--topk", type=int, default=10, help="сколько кандидатов переранжируем")
    ap.add_argument("--weight", type=float, default=1.0, help="вес re-ranker'а поверх косинуса")
    ap.add_argument("--name")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.model)
    emb_model = load(args.model, device)
    for p in emb_model.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    train_ds = TrainDataset(args.cache, "splits/folds.csv", args.val_fold, cfg)
    tok_tr, emb_tr = compute_tokens(emb_model, args.cache, train_ds, cfg, device)
    q_ds = EvalDataset(args.cache, f"splits/fold{args.val_fold}_query.csv", cfg)
    g_ds = EvalDataset(args.cache, f"splits/fold{args.val_fold}_gallery.csv", cfg)
    tok_q, emb_q = compute_tokens(emb_model, args.cache, q_ds, cfg, device)
    tok_g, emb_g = compute_tokens(emb_model, args.cache, g_ds, cfg, device)
    print(f"токены готовы за {time.time() - t0:.0f} с: train {tuple(tok_tr.shape)}, "
          f"query {tuple(tok_q.shape)}, gallery {tuple(tok_g.shape)}", flush=True)

    sampler = PairSampler(train_ds.labels, train_ds.cams, emb_tr)
    rr = PairReranker(tok_tr.shape[-1], layers=args.layers).to(device)
    opt = torch.optim.AdamW(rr.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.1)

    rr.train()
    for step in range(1, args.steps + 1):
        a, b, y = sampler.batch(args.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = rr(tok_tr[a].float().to(device), tok_tr[b].float().to(device))
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            m_rr, m_base = evaluate_rerank(rr, tok_q, tok_g, emb_q, emb_g, q_ds.ids, g_ds.ids,
                                           args.val_fold, device, args.topk, args.weight)
            print(f"шаг {step}/{args.steps}  loss {loss.item():.4f}  "
                  f"mAP@10 после re-ranking {m_rr:.4f} (без него {m_base:.4f})", flush=True)

    name = args.name or f"rerank_{os.path.basename(args.model)}_f{args.val_fold}"
    out = f"artifacts/models/{name}"
    os.makedirs(out, exist_ok=True)
    torch.save({"state": rr.state_dict(), "layers": args.layers, "dim": tok_tr.shape[-1],
                "embedder": args.model, "topk": args.topk, "weight": args.weight}, f"{out}/reranker.pt")
    print(f"saved {out}  ({(time.time() - t0) / 60:.1f} мин)")


if __name__ == "__main__":
    main()
