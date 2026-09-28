"""Что будет с режимом кандидатов, если открытое множество устроено иначе (стресс-тест).

Наши фолды строятся по допущению: у машины без пары gallery-кадры убраны целиком, поэтому у такого
query нет и двойника с его камеры. Если в закрытом тесте кадр той же камеры оставлен, то у query
без пары двойник есть, косинус будет высоким, и мы на него ответим — TNR просядет.

Проверить это на тесте нельзя (правило 4). Поэтому пересобираем сценарий на train: у части машин
убираем из gallery кросс-камерные кадры, оставляя только кадр с камеры запроса. Эмбеддинги уже
посчитаны, извлекать заново ничего не нужно — меняется только состав gallery.

    PYTHONPATH=. python scripts/analysis/openset_stress.py
"""
import argparse

import numpy as np

from reid.eval.eval_local import FoldTruth, candidate_metrics
from reid.post.policy import DEFAULT_CANDIDATE_THRESHOLD


def score_fold(k: int, emb_pattern: str, threshold: float, twin_share: float, rng: np.random.Generator):
    d = np.load(f"{emb_pattern.format(k=k)}/fold{k}.npz")
    q_ids, g_ids = d["query_ids"].tolist(), d["gallery_ids"].tolist()
    truth = FoldTruth.load(k)

    # Машины, которые делаем «без пары, но с двойником». У жертвы в gallery остаётся ровно один
    # кадр — с одной выбранной камеры. Тогда её query с этой камеры теряют кросс-камерный позитив,
    # но сохраняют двойника: это и есть проверяемый сценарий. Берём машины, у которых сейчас есть
    # кросс-камерный позитив и хотя бы один query с той камеры, чей кадр останется.
    n_pos_before = truth.n_pos()
    q_cams: dict[int, set[int]] = {}
    for q in q_ids:
        q_cams.setdefault(truth.vid[q], set()).add(truth.cam[q])
    g_cams: dict[int, set[int]] = {}
    for g in g_ids:
        g_cams.setdefault(truth.vid[g], set()).add(truth.cam[g])

    paired = sorted({truth.vid[q] for q in q_ids if n_pos_before[q] > 0}
                    & {v for v, cams in g_cams.items() if len(cams) > 1})
    victims = rng.choice(paired, size=int(len(paired) * twin_share), replace=False).tolist()
    # камера, кадр которой остаётся: та, с которой есть query, иначе сценарий не воспроизводится
    survivor = {}
    for v in victims:
        options = sorted(g_cams[v] & q_cams.get(v, set())) or sorted(g_cams[v])
        survivor[v] = options[rng.integers(len(options))]

    keep = [i for i, g in enumerate(g_ids)
            if truth.vid[g] not in survivor or truth.cam[g] == survivor[truth.vid[g]]]

    g_keep = [g_ids[i] for i in keep]
    stressed = FoldTruth(q_ids, g_keep, truth.vid, truth.cam)
    sim = d["query_emb"] @ d["gallery_emb"][keep].T
    best = np.argmax(sim, axis=1)
    cands = [(q, g_keep[best[i]], float(sim[i, best[i]]))
             for i, q in enumerate(q_ids) if sim[i, best[i]] >= threshold]
    n_pos = stressed.n_pos()
    return (candidate_metrics(stressed, cands, "official"),
            sum(n_pos[q] == 0 for q in q_ids) / len(q_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="artifacts/emb_final/final_oof_f{k}_cls_avg")
    ap.add_argument("--threshold", type=float, default=DEFAULT_CANDIDATE_THRESHOLD)
    ap.add_argument("--twin-share", type=float, default=0.20,
                    help="доля машин, переводимых в режим «без пары, но с двойником»")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = [score_fold(k, args.emb, args.threshold, args.twin_share, rng) for k in range(5)]
    for name, key in (("F1", "f1"), ("TNR", "tnr")):
        print(f"{name:4s} {np.mean([r[0][key] for r in rows]):.4f}")
    score = np.mean([0.7 * r[0]["f1"] + 0.3 * r[0]["tnr"] for r in rows])
    print(f"балл 0,7·F1+0,3·TNR  {score:.4f}   (штатный сценарий 0,959)")
    print(f"query без пары в стресс-сценарии: {np.mean([r[1] for r in rows]) * 100:.1f}%")


if __name__ == "__main__":
    main()
