"""Локальная оценка по протоколу организаторов (контракт C2 в docs/contracts.md).

Сверено с эталонным third_party/official/evaluate.py (A1.5, 20.09):
  * junk — кадр gallery с тем же vehicle_id И тем же camera_id, что у query; удаляется из всего
    поданного списка, и только после этого берутся первые 10;
  * AP@10 нормируется на min(n_pos, 10); query с n_pos = 0 исключаются из mAP и Rank;
    query, которого нет в submission, получает AP = 0;
  * кандидаты: берётся кандидат с максимальным confidence, junk из кандидатов НЕ удаляется,
    и ответ считается верным при совпадении vehicle_id — то есть двойник с той же камеры
    засчитывается как верный ответ (режим "official", см. DECISIONS D-023);
  * режим "cross_cam" — диагностический: верным считается только кросс-камерный позитив.
    Разница между режимами показывает, сколько F1 держится на двойниках.

Пример:
    python -m reid.eval.eval_local --fold 0 --submission S.csv --candidates C.csv --embeddings E.npy
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

CANDIDATE_MODES = ("official", "cross_cam")


@dataclass
class FoldTruth:
    """Разметка одного фолда: порядок query/gallery и их (vehicle_id, camera_id)."""
    query_ids: list[str]
    gallery_ids: list[str]
    vid: dict[str, int]
    cam: dict[str, int]

    @classmethod
    def load(cls, fold: int, folds_path: str = "splits/folds.csv") -> "FoldTruth":
        df = pd.read_csv(folds_path, dtype={"image_id": str})
        df = df[df.fold == fold]
        return cls(
            query_ids=df[df.role == "query"].image_id.tolist(),
            gallery_ids=df[df.role == "gallery"].image_id.tolist(),
            vid=dict(zip(df.image_id, df.vehicle_id)),
            cam=dict(zip(df.image_id, df.camera_id)),
        )

    def is_junk(self, q: str, g: str) -> bool:
        return self.vid[g] == self.vid[q] and self.cam[g] == self.cam[q]

    def is_pos(self, q: str, g: str) -> bool:
        return self.vid[g] == self.vid[q] and self.cam[g] != self.cam[q]

    def n_pos(self) -> dict[str, int]:
        """Число валидных позитивов (та же машина, другая камера) в gallery для каждого query."""
        per_vid_cam = {}
        for g in self.gallery_ids:
            key = (self.vid[g], self.cam[g])
            per_vid_cam[key] = per_vid_cam.get(key, 0) + 1
        per_vid = {}
        for (v, _), n in per_vid_cam.items():
            per_vid[v] = per_vid.get(v, 0) + n
        return {q: per_vid.get(self.vid[q], 0) - per_vid_cam.get((self.vid[q], self.cam[q]), 0)
                for q in self.query_ids}


def ranking_metrics(truth: FoldTruth, ranking: dict[str, list[str]]) -> dict:
    """mAP@10, Rank-1, Rank-5 по top-10 из submission (junk удаляется, без добора)."""
    n_pos = truth.n_pos()
    aps, r1, r5 = [], [], []
    for q in truth.query_ids:
        if n_pos[q] == 0:
            continue
        if q not in ranking:  # как в эталоне: пропущенный query получает AP = 0
            aps.append(0.0)
            r1.append(False)
            r5.append(False)
            continue
        kept = [g for g in ranking[q] if not truth.is_junk(q, g)][:10]
        rel = [truth.is_pos(q, g) for g in kept]
        hits, ap = 0, 0.0
        for k, r in enumerate(rel, start=1):
            if r:
                hits += 1
                ap += hits / k
        aps.append(ap / min(n_pos[q], 10))
        r1.append(bool(rel[:1]) and rel[0])
        r5.append(any(rel[:5]))
    return {"map10": float(np.mean(aps)), "rank1": float(np.mean(r1)), "rank5": float(np.mean(r5)),
            "n_query_eval": len(aps)}


def candidate_metrics(truth: FoldTruth, candidates: list[tuple[str, str, float]], mode: str) -> dict:
    """F1, TNR и балл режима кандидатов на уровне query, micro.

    mode: official  — как в эталонном evaluate.py: верным считается совпадение vehicle_id,
                      поэтому junk-двойник с той же камеры — тоже верный ответ;
          cross_cam — диагностика: верным считается только кросс-камерный позитив.
    """
    assert mode in CANDIDATE_MODES
    n_pos = truth.n_pos()
    by_q: dict[str, list[tuple[str, float]]] = {}
    for q, g, c in candidates:
        by_q.setdefault(q, []).append((g, float(c)))
    tp = fp = fn = tn = fp_open = 0
    scores, labels = [], []
    for q in truth.query_ids:
        has_pair = n_pos[q] > 0
        cands = by_q.get(q, [])
        labels.append(int(has_pair))
        scores.append(max((c for _, c in cands), default=float("-inf")))
        if not cands:
            fn += has_pair
            tn += not has_pair
            continue
        top = max(cands, key=lambda gc: gc[1])[0]  # max() берёт первый из равных — порядок файла
        correct = truth.is_pos(q, top) or (mode == "official" and truth.is_junk(q, top))
        if has_pair and correct:
            tp += 1
        else:
            fp += 1
            fp_open += not has_pair
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    tnr = tn / (tn + fp_open) if tn + fp_open else float("nan")
    return {"f1": f1, "tnr": tnr, "candidate_score": 0.7 * f1 + 0.3 * tnr,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "pr_auc": pr_auc(np.array(scores), np.array(labels)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average Precision для решения «у query есть пара». Как в эталонном evaluate.py."""
    finite = np.isfinite(scores)
    if labels.sum() == 0 or not finite.any():
        return float("nan")
    s = np.where(finite, scores, np.min(scores[finite]) - 1.0)
    y = labels[np.argsort(-s, kind="stable")]
    cum = np.cumsum(y)
    prec = cum / (np.arange(len(y)) + 1)
    rec = cum / labels.sum()
    ap, prev = 0.0, 0.0
    for p_, r_ in zip(prec, rec):
        ap += p_ * (r_ - prev)
        prev = r_
    return float(ap)


def embedding_metrics(truth: FoldTruth, emb: np.ndarray) -> dict:
    """Полный mAP и mINP по косинусному ранжированию всей gallery (справочно)."""
    nq = len(truth.query_ids)
    assert emb.shape[0] == nq + len(truth.gallery_ids), "embeddings: query + gallery rows expected"
    e = emb.astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
    sim = e[:nq] @ e[nq:].T
    g_vid = np.array([truth.vid[g] for g in truth.gallery_ids])
    g_cam = np.array([truth.cam[g] for g in truth.gallery_ids])
    aps, inps = [], []
    for i, q in enumerate(truth.query_ids):
        order = np.argsort(-sim[i], kind="stable")  # равные score — по порядку gallery
        same_vid = g_vid[order] == truth.vid[q]
        junk = same_vid & (g_cam[order] == truth.cam[q])
        pos = (same_vid & ~junk)[~junk]
        if not pos.any():
            continue
        ranks = np.flatnonzero(pos) + 1
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
        inps.append(len(ranks) / ranks[-1])
    return {"full_map": float(np.mean(aps)), "minp": float(np.mean(inps))}


def evaluate(fold: int, ranking: dict[str, list[str]],
             candidates: list[tuple[str, str, float]] | None = None,
             emb: np.ndarray | None = None, folds_path: str = "splits/folds.csv") -> dict:
    truth = FoldTruth.load(fold, folds_path)
    n_pos = truth.n_pos()
    out = {"fold": fold, "n_query": len(truth.query_ids),
           "n_query_openset": sum(n == 0 for n in n_pos.values())}
    out.update(ranking_metrics(truth, ranking))
    if candidates is not None:
        for mode in CANDIDATE_MODES:
            out[mode] = candidate_metrics(truth, candidates, mode)
        out["candidate_score"] = out["official"]["candidate_score"]  # официальная трактовка
    if emb is not None:
        out.update(embedding_metrics(truth, emb))
    return out


def ranking_from_embeddings(q_emb: np.ndarray, g_emb: np.ndarray, gallery_ids: list[str],
                            query_ids: list[str], k: int = 10) -> dict[str, list[str]]:
    """Простой косинусный top-k без какой-либо обработки двойников (baseline)."""
    q = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    g = g_emb / (np.linalg.norm(g_emb, axis=1, keepdims=True) + 1e-12)
    sim = q @ g.T
    top = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    return {qid: [gallery_ids[j] for j in row] for qid, row in zip(query_ids, top)}


def read_submission(path: str) -> dict[str, list[str]]:
    """Эталонный evaluate.py читает submission.csv БЕЗ заголовка; заголовок, если он есть, пропускаем."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r and r[0].strip()]
    if rows and rows[0][0].strip() == "query_id":
        rows = rows[1:]
    return {r[0].strip(): [g.strip() for g in r[1:] if g.strip()] for r in rows}


def read_candidates(path: str) -> list[tuple[str, str, float]]:
    with open(path, newline="") as f:
        return [(r["query_id"], r["gallery_id"], float(r["confidence"])) for r in csv.DictReader(f)
                if r["gallery_id"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--candidates")
    ap.add_argument("--embeddings")
    ap.add_argument("--folds", default="splits/folds.csv")
    ap.add_argument("--out")
    args = ap.parse_args()
    res = evaluate(args.fold, read_submission(args.submission),
                   read_candidates(args.candidates) if args.candidates else None,
                   np.load(args.embeddings) if args.embeddings else None, args.folds)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
