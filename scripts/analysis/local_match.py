"""Локальное сопоставление патчей на высоком разрешении как вторая ступень.

Почему стоит вернуться к тому, что провалилось. A-049 мерил соответствие патчей при входе 280: там
патч 14 px покрывает 5% кропа и кодирует форму кузова, поэтому у двух разных машин одной модели
патчи сходились почти идеально, и оценка монотонно вредила. При входе 392 патч покрывает 1,3%
кропа, при 560 — 0,6%: это масштаб эмблемы, диска, наклейки. Разбор ошибок показал случаи, где
решает ровно эмблема (VW против Citroën).

ТЗ §3 разрешает local-feature модели на переранжировании внутри top-K, §7 выводит переранжирование
из замера скорости, поэтому дорогая модель здесь ничего не стоит по баллам.

Считаются три оценки пары, все по матрице сходств патчей:

  best   — как в A-049: среднее по лучшей половине лучших соответствий, симметрично по сторонам;
  mutual — доля взаимных пар (патч A выбрал B, и B выбрал A) с силой выше порога: взаимность
           отсеивает патчи фона и ровных панелей, которые находят себе пару всегда;
  geom   — доля взаимных пар, чьё смещение согласуется с медианным: у одной машины смещения
           образуют плавное поле, у разных машин одной модели — разнонаправленный шум.

    PYTHONPATH=. python scripts/analysis/local_match.py --fold 0 \\
        --base artifacts/emb/pool_cls_top16_f0_cls_top16 artifacts/emb/div_clip_f0_cls_top16 \\
               artifacts/emb/div_siglip_f0_top16 \\
        --model artifacts/models/hires392_f0 --cache artifacts/crops1024
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from reid.eval.eval_local import FoldTruth, ranking_metrics
from reid.model import load, load_config
from reid.model.interpret import tokens_for
from scripts.analysis.hub_penalty import parts


def pair_features(tq: torch.Tensor, tc: torch.Tensor, side: int, top_share: float,
                  tau: float, radius: float) -> dict[str, torch.Tensor]:
    """[P,D] и [C,P,D] → по три оценки на каждого кандидата."""
    tq = torch.nn.functional.normalize(tq.float(), dim=-1)
    tc = torch.nn.functional.normalize(tc.float(), dim=-1)
    sim = torch.einsum("pd,cqd->cpq", tq, tc)                      # [C, P, P]
    c, p, _ = sim.shape

    def top_mean(x):
        k = max(1, int(x.shape[1] * top_share))
        return x.topk(k, dim=1).values.mean(dim=1)

    best_q, idx_q = sim.max(dim=2)                                 # для каждого патча query
    best_c, idx_c = sim.max(dim=1)                                 # для каждого патча кандидата
    best = 0.5 * (top_mean(best_q) + top_mean(best_c))

    ar = torch.arange(p, device=sim.device)
    back = torch.gather(idx_c, 1, idx_q)                           # кого выбрал выбранный
    mutual = (back == ar.expand(c, p)) & (best_q > tau)
    mutual_share = mutual.float().mean(dim=1)

    qy, qx = ar // side, ar % side
    cy, cx = idx_q // side, idx_q % side
    dy = (cy - qy.expand(c, p)).float()
    dx = (cx - qx.expand(c, p)).float()
    big = torch.full_like(dx, 1e3)
    dyc = torch.where(mutual, dy, big)
    dxc = torch.where(mutual, dx, big)
    med_y = dyc.masked_fill(~mutual, float("nan")).nanmedian(dim=1).values
    med_x = dxc.masked_fill(~mutual, float("nan")).nanmedian(dim=1).values
    dist = ((dy - med_y[:, None]) ** 2 + (dx - med_x[:, None]) ** 2).sqrt()
    geom = (mutual & (dist <= radius)).float().mean(dim=1)
    return {"best": best, "mutual": mutual_share, "geom": geom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--model", required=True, help="модель высокого разрешения для сопоставления")
    ap.add_argument("--cache", default="artifacts/crops1024")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--top-share", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=0.5, help="порог силы для взаимной пары")
    ap.add_argument("--radius", type=float, default=2.0, help="допуск согласованности смещений, патчей")
    ap.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    truth = FoldTruth.load(args.fold)
    S, H, q_ids, g_ids = parts(args.base, args.fold, 3)
    base = S - args.lam * H[None, :]
    order = np.argsort(-base, axis=1)[:, : args.window]
    b = ranking_metrics(truth, {q: [g_ids[j] for j in order[i][:10]]
                                for i, q in enumerate(q_ids)})["map10"]
    print(f"фолд {args.fold}, окно {args.window}, ступень 1 со штрафом: {b:.4f}", flush=True)

    cfg = load_config(args.model)
    model = load(args.model, args.device)
    side = cfg["input_size"][0] // model.backbone.patch_embed.patch_size[0]
    need = sorted({g_ids[j] for row in order for j in row})
    print(f"патч-сетка {side}x{side}, кадров галереи в окнах: {len(need)}", flush=True)
    tok_q = tokens_for(model, args.cache, q_ids, cfg, args.device)
    tok_g = tokens_for(model, args.cache, need, cfg, args.device)
    gpos = {g: i for i, g in enumerate(need)}

    feats = {k: np.zeros((len(q_ids), args.window), dtype=np.float32)
             for k in ("best", "mutual", "geom")}
    with torch.inference_mode():
        for i in range(len(q_ids)):
            idx = [gpos[g_ids[j]] for j in order[i]]
            out = pair_features(tok_q[i].to(args.device), tok_g[idx].to(args.device),
                                side, args.top_share, args.tau, args.radius)
            for k, v in out.items():
                feats[k][i] = v.cpu().numpy()
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(q_ids)}", flush=True)

    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    print(f"\n{'оценка':8s}" + "".join(f"{w:>9}" for w in args.weights))
    for name, F in feats.items():
        row = []
        for w in args.weights:
            ranked = {}
            for i, q in enumerate(q_ids):
                mix = z(base[i, order[i]]) + w * z(F[i])
                ranked[q] = [g_ids[order[i][j]] for j in np.argsort(-mix)][:10]
            row.append(ranking_metrics(truth, ranked)["map10"])
        print(f"{name:8s}" + "".join(f"{v:>9.4f}" for v in row)
              + f"   лучшее {max(row) - b:+.4f}")


if __name__ == "__main__":
    main()
