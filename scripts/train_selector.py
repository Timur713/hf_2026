"""Выбиратель: кому верить на этом запросе.

Измерено 24.09: если для каждого запроса безошибочно выбрать лучший из шести ранжировщиков
(фузия и пять одиночных моделей), mAP@10 вырастает с 0,8563 до 0,9124 на фолде 0 и с 0,8382 до
0,8888 на фолде 1. У 19% запросов какая-то одиночная модель лучше фузии. Усреднение это умение
гасит: модель при 392 вытаскивает 18,8% трудных случаев в top-2, а внутри фузии — только 9,8%.

Поэтому вместо постоянных весов фузии веса предсказываются **под запрос** по профилю уверенности
каждого ранжировщика: отрывы между верхними кандидатами, энтропия распределения сходств, согласие
с остальными. Модель крошечная (несколько тысяч параметров) — обучающих запросов всего около
тысячи, и большая переобучится.

Правила соблюдены: признаки считаются только по самому запросу и статической галерее, другие
query не участвуют.

    PYTHONPATH=. python scripts/train_selector.py --train-fold 1 --eval-fold 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from reid.eval.eval_local import FoldTruth, ranking_metrics
from scripts.analysis.hub_penalty import parts

SETS = {
    0: {"DINOv2": "artifacts/emb/pool_cls_top16_f0_cls_top16",
        "CLIP": "artifacts/emb/div_clip_f0_cls_top16",
        "SigLIP": "artifacts/emb/div_siglip_f0_top16",
        "DINOv2-reg": "artifacts/emb/div_dinoreg_f0_cls_top16",
        "EVA-02": "artifacts/emb/div_eva02_f0_cls_top16"},
    1: {"DINOv2": "artifacts/emb/div_dinov2_f1_cls_top16",
        "CLIP": "artifacts/emb/div_clip_f1_cls_top16",
        "SigLIP": "artifacts/emb/siglip2_f1_full_avg",
        "DINOv2-reg": "artifacts/emb/div_dinoreg_f1_cls_top16",
        "EVA-02": "artifacts/emb/div_eva02_f1_cls_top16"},
}


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def build(fold: int, window: int):
    """Окно кандидатов от базы и признаки каждого ранжировщика внутри него."""
    names = list(SETS[fold])
    base_dumps = [SETS[fold][k] for k in ("DINOv2", "CLIP", "SigLIP")]
    S, H, q_ids, g_ids = parts(base_dumps, fold, 3)
    base = S - 0.3 * H[None, :]
    sims = [base]
    for k in names:
        d = np.load(f"{SETS[fold][k]}/fold{fold}.npz")
        sims.append(unit(d["query_emb"]) @ unit(d["gallery_emb"]).T)
    order = np.argsort(-base, axis=1)[:, :window]

    Z = []            # [R, Q, K] — стандартизованные по окну оценки каждого ранжировщика
    prof = []         # [R, Q, F] — профиль уверенности ранжировщика на этом запросе
    for s in sims:
        row = np.take_along_axis(s, order, axis=1)
        zr = (row - row.mean(1, keepdims=True)) / (row.std(1, keepdims=True) + 1e-9)
        Z.append(zr)
        full = np.sort(s, axis=1)[:, ::-1]
        srt = np.sort(zr, axis=1)[:, ::-1]
        p = np.exp(zr - zr.max(1, keepdims=True))
        p = p / p.sum(1, keepdims=True)
        ent = -(p * np.log(p + 1e-9)).sum(1)
        prof.append(np.stack([srt[:, 0] - srt[:, 1], srt[:, 1] - srt[:, 2], srt[:, 2] - srt[:, 4],
                              ent, full[:, 0] - full[:, 1], full[:, :10].std(1)], axis=1))
    Z = np.stack(Z).astype(np.float32)
    prof = np.stack(prof).astype(np.float32)
    prof = (prof - prof.mean(axis=(0, 1), keepdims=True)) / (prof.std(axis=(0, 1), keepdims=True) + 1e-9)

    truth = FoldTruth.load(fold)
    cand = np.array([[g_ids[j] for j in row] for row in order], dtype=object)
    pos = np.array([[truth.is_pos(q, g) for g in row] for q, row in zip(q_ids, cand)])
    junk = np.array([[truth.is_junk(q, g) for g in row] for q, row in zip(q_ids, cand)])
    return dict(Z=Z, prof=prof, pos=pos, junk=junk, cand=cand, q_ids=q_ids, truth=truth,
                names=["база"] + names)


class Selector(nn.Module):
    """Профиль уверенности каждого ранжировщика → его вес на этом запросе."""

    def __init__(self, n_rankers: int, n_feat: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Старт с равных весов: модель должна улучшать фузию, а не изобретать её заново
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.bias = nn.Parameter(torch.zeros(n_rankers))
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, prof, Z):
        # prof [R, Q, F], Z [R, Q, K]
        logits = self.net(prof).squeeze(-1) + self.bias[:, None]     # [R, Q]
        w = torch.softmax(logits * self.temp, dim=0)
        return (w[:, :, None] * Z).sum(0), w


def evaluate(data, score):
    order = np.argsort(-score, axis=1)
    ranking = {q: [data["cand"][i][j] for j in order[i]][:10]
               for i, q in enumerate(data["q_ids"])}
    return ranking_metrics(data["truth"], ranking)["map10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-fold", type=int, default=1)
    ap.add_argument("--eval-fold", type=int, default=0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    tr = build(args.train_fold, args.window)
    ev = build(args.eval_fold, args.window)
    print(f"ранжировщики: {', '.join(tr['names'])}")
    print(f"обучение на фолде {args.train_fold}: база {evaluate(tr, tr['Z'][0]):.4f}; "
          f"оценка на фолде {args.eval_fold}: база {evaluate(ev, ev['Z'][0]):.4f}")

    # Цель — мягкая: вероятность распределяется по всем валидным позитивам окна
    def target(d):
        t = d["pos"].astype(np.float32)
        return t / np.maximum(t.sum(1, keepdims=True), 1e-9), d["pos"].any(1)

    t_tr, ok_tr = target(tr)
    Zt = torch.tensor(tr["Z"]); Pt = torch.tensor(tr["prof"]); Tt = torch.tensor(t_tr)
    mask = torch.tensor(~tr["junk"])          # двойника из обучения исключаем: он не ответ
    ok = torch.tensor(ok_tr)
    model = Selector(Zt.shape[0], Pt.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    best, best_state = -1, None
    for step in range(1, args.steps + 1):
        model.train()
        s, _ = model(Pt, Zt)
        s = s.masked_fill(~mask, -1e9)
        loss = -(Tt[ok] * torch.log_softmax(s[ok], dim=1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                se, w = model(torch.tensor(ev["prof"]), torch.tensor(ev["Z"]))
                st, _ = model(Pt, Zt)
            v_tr, v_ev = evaluate(tr, st.numpy()), evaluate(ev, se.numpy())
            if v_tr > best:
                best, best_state = v_tr, (v_ev, w.mean(1).detach().numpy())
            print(f"  шаг {step:4d}  loss {loss.item():.4f}  обучающий фолд {v_tr:.4f}  "
                  f"проверочный {v_ev:.4f}")
    v_ev, w = best_state
    print(f"\nпо лучшему шагу на обучающем фолде: проверочный {v_ev:.4f} "
          f"(база {evaluate(ev, ev['Z'][0]):.4f})")
    print("средние веса доверия: " + "  ".join(f"{n}={x:.2f}" for n, x in zip(tr["names"], w)))


if __name__ == "__main__":
    main()
