"""Две политики уверенности против двух возможных устройств открытого множества.

top1   — как сейчас: уверенность = косинус лучшего кадра.
rank2  — уверенность = косинус ВТОРОГО кадра. Идея: двойник с камеры запроса почти всегда занимает
         первое место, поэтому второй кадр отвечает на вопрос «есть ли у машины ещё один кадр,
         кроме почти-дубликата». camera_id не используется, зависимости от других query нет.
"""
import numpy as np
from reid.eval.eval_local import FoldTruth, candidate_metrics

EMB = "artifacts/emb_final/final_oof_f{k}_cls_avg"


def build(k, twin_share, rng):
    d = np.load(f"{EMB.format(k=k)}/fold{k}.npz")
    q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
    t = FoldTruth.load(k)
    if twin_share == 0:
        return t, q_ids, g_ids, d["query_emb"] @ d["gallery_emb"].T
    n_pos = t.n_pos()
    q_cams, g_cams = {}, {}
    for q in q_ids: q_cams.setdefault(t.vid[q], set()).add(t.cam[q])
    for g in g_ids: g_cams.setdefault(t.vid[g], set()).add(t.cam[g])
    paired = sorted({t.vid[q] for q in q_ids if n_pos[q] > 0}
                    & {v for v, c in g_cams.items() if len(c) > 1})
    surv = {}
    for v in rng.choice(paired, size=int(len(paired) * twin_share), replace=False).tolist():
        o = sorted(g_cams[v] & q_cams.get(v, set())) or sorted(g_cams[v])
        surv[v] = o[rng.integers(len(o))]
    keep = [i for i, g in enumerate(g_ids) if t.vid[g] not in surv or t.cam[g] == surv[t.vid[g]]]
    gk = [g_ids[i] for i in keep]
    return FoldTruth(q_ids, gk, t.vid, t.cam), q_ids, gk, d["query_emb"] @ d["gallery_emb"][keep].T


def score(policy, thr, twin_share, seed=0):
    out = []
    for k in range(5):
        t, q_ids, g_ids, sim = build(k, twin_share, np.random.default_rng(seed))
        order = np.argsort(-sim, axis=1)
        rank = 0 if policy == "top1" else 1
        cands = []
        for i, q in enumerate(q_ids):
            if sim.shape[1] <= rank:
                continue
            j = order[i, rank]
            conf = float(sim[i, j])
            if conf >= thr:
                # отвечаем всегда лучшим кадром, уверенность берём по политике
                cands.append((q, g_ids[order[i, 0]], conf))
        m = candidate_metrics(t, cands, "official")
        out.append(0.7 * m["f1"] + 0.3 * m["tnr"])
    return float(np.mean(out))


for policy in ("top1", "rank2"):
    best = max(((score(policy, thr, 0.0), thr) for thr in np.arange(0.35, 0.80, 0.01)))
    s_norm, thr = best
    print(f"{policy:6s} порог {thr:.2f}  штатно {s_norm:.4f}  "
          f"стресс-20% {score(policy, thr, 0.20):.4f}  стресс-50% {score(policy, thr, 0.50):.4f}")
