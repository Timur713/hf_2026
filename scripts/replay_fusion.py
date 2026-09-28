"""Replay C4 TRAIN-fold dumps through production postprocessing, without a GPU.

No parameter search, test inputs, pickle loading, or training. Cached logits must
cover the exact production top-10 in the exact order; stale candidate windows fail.
Reported metrics do NOT certify that the supplied model weights were truly OOF.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from reid.eval.eval_local import FoldTruth, evaluate, read_candidates, read_submission
from reid.eval.validate_outputs import _read_input_ids, validate_outputs
from reid.infer.run import InputRow, rank_embeddings, write_outputs
from reid.post.gallery_policy import GalleryPolicy
from reid.post.policy import candidate_threshold_for, rerank_weight_for
from scripts.audit_release import file_record
from scripts.prepare_fusion_artifact import gallery_similarity_std


def load_components(paths: list[Path], query_ids: list[str], gallery_ids: list[str]):
    if len(paths) < 2:
        raise ValueError("at least two C4 dumps are required")
    arrays = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if not {"query_ids", "gallery_ids", "query_emb", "gallery_emb"} <= set(data.files):
                raise ValueError(f"missing C4 keys: {path}")
            qids, gids = data["query_ids"], data["gallery_ids"]
            if qids.ndim != 1 or gids.ndim != 1 or qids.tolist() != query_ids or gids.tolist() != gallery_ids:
                raise ValueError(f"C4 IDs/order do not match the train fold: {path}")
            q, g = data["query_emb"], data["gallery_emb"]
            for matrix, count in ((q, len(query_ids)), (g, len(gallery_ids))):
                if (matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[0] != count
                        or matrix.shape[1] == 0 or not np.isfinite(matrix).all()):
                    raise ValueError(f"invalid C4 embedding matrix: {path}")
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                if not np.allclose(norms, 1.0, atol=1e-3):
                    raise ValueError(f"C4 embeddings must be L2-normalized: {path}")
            if q.shape[1] != g.shape[1]:
                raise ValueError(f"query/gallery feature dimensions differ: {path}")
            arrays.append((q / np.linalg.norm(q, axis=1, keepdims=True),
                           g / np.linalg.norm(g, axis=1, keepdims=True)))
    return arrays


def fuse_components(arrays, scales=None):
    scales = ([gallery_similarity_std(g) for _, g in arrays] if scales is None else list(scales))
    if len(scales) != len(arrays) or any(not np.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("one positive finite similarity_std per component is required")
    reference = math.sqrt(min(scales))
    coefficients = [reference / math.sqrt(s) for s in scales]
    if min(coefficients) < np.finfo(np.float32).tiny:
        raise ValueError("similarity_std dynamic range is too wide for float32")
    q = np.concatenate([pair[0] * c for pair, c in zip(arrays, coefficients)], axis=1)
    g = np.concatenate([pair[1] * c for pair, c in zip(arrays, coefficients)], axis=1)
    return (q / np.linalg.norm(q, axis=1, keepdims=True),
            g / np.linalg.norm(g, axis=1, keepdims=True), scales)


class CachedVerifier:
    def __init__(self, path: Path, fold: int, query_ids: list[str], rankings: list[list[str]]):
        with np.load(path, allow_pickle=False) as data:
            for key in ("fold", "query_ids", "candidates", "logits"):
                if key not in data.files:
                    raise ValueError(f"logit cache missing {key}")
            stored_fold = data["fold"]
            if stored_fold.shape != () or stored_fold.item() != fold:
                raise ValueError("logit cache belongs to a different fold")
            if data["query_ids"].tolist() != query_ids or data["candidates"].tolist() != rankings:
                raise ValueError("stale logit cache: query IDs or exact production top-10/order differ")
            logits = data["logits"]
            if (logits.shape != (len(query_ids), 10) or logits.dtype != np.float32
                    or not np.isfinite(logits).all()):
                raise ValueError("logits must be finite float32 with shape (N_query, 10)")
            self.rows = {qid: (tuple(cands), row.copy())
                         for qid, cands, row in zip(query_ids, rankings, logits)}

    def score_pairs(self, query, candidates):
        ids, logits = self.rows[str(query[0])]
        if tuple(str(item[0]) for item in candidates) != ids:
            raise ValueError("runtime candidate order differs from verified logit cache")
        return logits.copy()


def replay(paths: list[Path], fold: int, out: Path, *, splits: Path = Path("splits"),
           logits: Path | None = None, scales=None, rerank_weight: float = 0.2,
           candidate_threshold: float = 6.0) -> dict:
    if type(fold) is not int or fold not in range(5):
        raise ValueError("only train folds 0..4 are accepted")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite replay: {out}")
    cfg = {"candidate_policy": "gallery_z", "rerank_mode": "standardized",
           "rerank_weight": rerank_weight_for({}, rerank_weight),
           "candidate_threshold": candidate_threshold}
    threshold = candidate_threshold_for(cfg)
    query_csv, gallery_csv = splits / f"fold{fold}_query.csv", splits / f"fold{fold}_gallery.csv"
    qids, gids = _read_input_ids(query_csv), _read_input_ids(gallery_csv)
    truth = FoldTruth.load(fold, str(splits / "folds.csv"))
    if qids != truth.query_ids or gids != truth.gallery_ids:
        raise ValueError("fold CSVs disagree with folds.csv")
    q, g, used_scales = fuse_components(load_components(paths, qids, gids), scales)
    # Paths are opaque ID keys: the cached scorer never opens an image.
    qrows = [InputRow(i, Path(i), (0, 0, 1, 1)) for i in qids]
    grows = [InputRow(i, Path(i), (0, 0, 1, 1)) for i in gids]
    results = {"fusion": GalleryPolicy(g, gids, hub_weight=0).search(q),
               "hub": rank_embeddings(qrows, grows, q, g, cfg)}
    base_window = results["hub"].ranking
    if logits is not None:
        scorer = CachedVerifier(logits, fold, qids, base_window)
        results["verifier"] = rank_embeddings(qrows, grows, q, g, cfg, scorer)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        tracked_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        tracked_dirty = None
    report = {
        "kind": "train_fold_production_postprocessing_replay", "fold": fold,
        "code_commit": commit, "tracked_worktree_dirty": tracked_dirty,
        "numpy_version": np.__version__,
        "replay_script": file_record(Path(__file__)),
        "dumps": [{"path": str(p), **file_record(p)} for p in paths],
        "logits": {"path": str(logits), **file_record(logits)} if logits is not None else None,
        "inputs": {p.name: file_record(p) for p in (query_csv, gallery_csv, splits / "folds.csv")},
        "policy": {**cfg, "hub_k": 3, "hub_weight": 0.5, "top_k": 10},
        "similarity_std": used_scales,
        "scale_source": "provided fixed constants" if scales is not None else "this static fold gallery, diagonal included",
        "confidence_statistics": "gallery off-diagonal, ddof=0, epsilon=1e-12",
        "ranking_statistics": "per-query only, ddof=0, epsilon=1e-9; stable gallery-order ties",
        "training_provenance_verified": False,
        "scope": "C4 postprocessing replay only; OOF training provenance, full-frame FP16 embeddings, GPU speed and plate invariance NOT verified",
        "stages": {},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temp:
        staged = Path(temp) / "result"
        staged.mkdir()
        np.savez_compressed(staged / "candidate_window.npz", fold=np.array(fold),
                            query_ids=np.array(qids), candidates=np.array(base_window))
        for stage, result in results.items():
            destination = staged / stage
            write_outputs(destination, qrows, grows, q, g, threshold, result)
            validate_outputs(query_csv, gallery_csv, destination / "submission.csv",
                             destination / "embeddings.npy", destination / "candidates.csv")
            metrics = evaluate(fold, read_submission(str(destination / "submission.csv")),
                               read_candidates(str(destination / "candidates.csv")),
                               folds_path=str(splits / "folds.csv"))
            report["stages"][stage] = {"metrics": metrics, "files": {
                name: file_record(destination / name)
                for name in ("submission.csv", "embeddings.npy", "candidates.csv")}}
        (staged / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"replay appeared while computing: {out}")
        staged.rename(out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, action="append", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--logits", type=Path)
    parser.add_argument("--similarity-std", type=float, action="append")
    parser.add_argument("--rerank-weight", type=float, default=0.2)
    parser.add_argument("--candidate-threshold", type=float, default=6.0)
    args = parser.parse_args()
    try:
        report = replay(args.dump, args.fold, args.out, splits=args.splits, logits=args.logits,
                        scales=args.similarity_std, rerank_weight=args.rerank_weight,
                        candidate_threshold=args.candidate_threshold)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"REPLAY FAILED: {exc}\n")
    for stage, result in report["stages"].items():
        print(f"{stage}: map10={result['metrics']['map10']:.6f}")
    print("Training provenance unverified; these are not certified OOF or end-to-end GPU results.")


if __name__ == "__main__":
    main()
