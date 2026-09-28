"""Replay TRAIN fold-0 C6 embeddings through production fusion+hub, without weights.

No extraction, GPU, verifier, parameter search or image decoding. A .npy file has
no IDs: CSV order is checked against folds.csv, but embedding row identity and
the claim that these vectors came from full-frame FP16 extraction are NOT proved.
Run from the repository root with PYTHONPATH=.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from reid.eval.eval_local import FoldTruth, evaluate, read_candidates, read_submission
from reid.eval.validate_outputs import validate_outputs
from reid.infer.fusion import _positive_number, _unique_object
from reid.infer.run import rank_embeddings, read_inputs, write_outputs
from reid.post.policy import candidate_threshold_for, rerank_weight_for
from scripts.audit_release import file_record


def read_metadata(path: Path) -> dict:
    """Validate JSON metadata only; never resolve/open the component weights."""
    cfg = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    required = {"schema_version", "name", "components", "candidate_policy",
                "candidate_threshold", "rerank_mode"}
    if not isinstance(cfg, dict) or not required <= set(cfg) or set(cfg) - required - {"rerank_weight"}:
        raise ValueError("invalid fusion metadata keys")
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("expected fusion schema_version 1")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("fusion name must be nonempty")
    if cfg["candidate_policy"] != "gallery_z" or cfg["rerank_mode"] != "standardized":
        raise ValueError("expected gallery_z / standardized fusion metadata")
    cfg["candidate_threshold"] = _positive_number(cfg["candidate_threshold"], "candidate_threshold")
    if "rerank_weight" in cfg:
        cfg["rerank_weight"] = rerank_weight_for(cfg)
    if not isinstance(cfg["components"], list) or len(cfg["components"]) < 2:
        raise ValueError("expected at least two fusion components")
    seen = set()
    for component in cfg["components"]:
        if not isinstance(component, dict) or set(component) != {"path", "similarity_std"}:
            raise ValueError("invalid component metadata")
        name = component["path"]
        if (not isinstance(name, str) or not name.strip() or Path(name).is_absolute()
                or Path(name) == Path(".") or ".." in Path(name).parts or Path(name) in seen):
            raise ValueError("component path must be a unique contained relative path")
        seen.add(Path(name))
        _positive_number(component["similarity_std"], "similarity_std")
    return cfg


def replay(embeddings: Path, fusion_config: Path, images: Path, out: Path,
           *, splits: Path = Path("splits")) -> dict:
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite replay: {out}")
    cfg = read_metadata(fusion_config)
    query_csv, gallery_csv = splits / "fold0_query.csv", splits / "fold0_gallery.csv"
    truth = FoldTruth.load(0, str(splits / "folds.csv"))
    qrows, grows = read_inputs(query_csv, images), read_inputs(gallery_csv, images)
    qids, gids = [row.image_id for row in qrows], [row.image_id for row in grows]
    if qids != truth.query_ids or gids != truth.gallery_ids:
        raise ValueError("fold0 CSV IDs/order disagree with folds.csv")
    if not qids or len(gids) < 10 or set(qids) & set(gids):
        raise ValueError("need nonempty disjoint query/gallery and at least ten gallery rows")
    matrix = np.load(embeddings, allow_pickle=False)
    if (not isinstance(matrix, np.ndarray) or matrix.dtype != np.float32 or matrix.ndim != 2
            or matrix.shape[0] != len(qids) + len(gids) or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()):
        raise ValueError("expected finite float32 C6 matrix [N_query+N_gallery, D>0]")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=0, atol=1e-3):
        raise ValueError("C6 embeddings must be L2-normalized; no silent normalization")
    q, g = matrix[:len(qrows)], matrix[len(qrows):]
    result = rank_embeddings(qrows, grows, q, g, cfg, verifier=None)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    report = {
        "kind": "train_fold0_existing_C6_fusion_hub_replay_no_verifier", "fold": 0,
        "code_commit": commit, "tracked_worktree_dirty": dirty, "numpy_version": np.__version__,
        "script": file_record(Path(__file__)),
        "input_embeddings": {"path": str(embeddings), **file_record(embeddings)},
        "input_fusion_config": {"path": str(fusion_config), **file_record(fusion_config)},
        "inputs": {path.name: file_record(path) for path in (query_csv, gallery_csv, splits / "folds.csv")},
        "metadata": cfg, "verifier_applied": False,
        "component_scales_reapplied": False, "hub_k": 3, "hub_weight": 0.5,
        "shape": list(matrix.shape), "l2_norm_range": [float(norms.min()), float(norms.max())],
        "csv_order_verified_against_fold_truth": True,
        "embedding_row_identity_verified": False, "extraction_provenance_verified": False,
        "scope": "Existing fused C6 vectors only; no weights, extraction, GPU, decoding, test or tuning. Row identity and claimed full-frame/FP16/OOF provenance are not certified. Component dimension/config compatibility cannot be inferred from this JSON alone.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temporary:
        staged = Path(temporary) / "result"
        write_outputs(staged, qrows, grows, q, g, candidate_threshold_for(cfg), result)
        validate_outputs(query_csv, gallery_csv, staged / "submission.csv",
                         staged / "embeddings.npy", staged / "candidates.csv")
        report["metrics"] = evaluate(0, read_submission(str(staged / "submission.csv")),
                                     read_candidates(str(staged / "candidates.csv")),
                                     folds_path=str(splits / "folds.csv"))
        report["files"] = {name: file_record(staged / name) for name in
                           ("submission.csv", "embeddings.npy", "candidates.csv")}
        (staged / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"replay appeared while computing: {out}")
        staged.rename(out)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--fusion-config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="read_inputs checks existence only; no decoding")
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = replay(args.embeddings, args.fusion_config, args.images, args.out, splits=args.splits)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"REPLAY FAILED: {exc}\n")
    print(f"fusion+hub (no verifier): map10={report['metrics']['map10']:.9f}")
    print("CSV order checked; embedding row identity and full-frame/FP16/OOF provenance unverified.")


if __name__ == "__main__":
    main()
