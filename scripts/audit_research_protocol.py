"""CPU-only counterexamples for the research/production protocol audit.

This is NOT an evaluation of trained models. It reads only committed train-fold
IDs/labels, generates synthetic embeddings/rankings, and never opens images,
test CSVs, checkpoints or a GPU. Metrics use the canonical eval_local evaluator.

    PYTHONPATH=. python scripts/audit_research_protocol.py \
        --out outputs/protocol-audit-20260925.json

The output must be new; existing files (including symlinks) are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np

from reid.eval.eval_local import FoldTruth, ranking_metrics
from scripts.analysis.hub_penalty import concat_parts, parts
from scripts.analysis.rerank_mix import Bench, z

SEED = 20260925
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/audit_research_protocol.py",
    "scripts/analysis/hub_penalty.py",
    "scripts/analysis/rerank_mix.py",
    "reid/eval/eval_local.py",
    "splits/folds.csv",
)


def fusion_probe(rng: np.random.Generator) -> dict:
    """Exercise both actual research helpers using in-memory synthetic C4 data."""
    nq, ng, dim = 40, 80, 32
    dumps = {}
    unit = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    for i in range(3):
        q = rng.normal(size=(nq, dim))
        g = rng.normal(size=(ng, dim))
        # Different anisotropy makes query-row and fixed-gallery scales differ.
        if i == 1:
            q[:, 0] += 5
            g[:, 0] += 5
        if i == 2:
            q[:, 0] += rng.uniform(1, 20, size=nq)
            g[:, 0] += 10
        dumps[f"m{i}/fold0.npz"] = {
            "query_emb": unit(q), "gallery_emb": unit(g),
            "query_ids": np.array([f"q{j}" for j in range(nq)]),
            "gallery_ids": np.array([f"g{j}" for j in range(ng)]),
        }
    # No NPZ is read or written: exercise the existing formulas, not a copy.
    with patch("numpy.load", side_effect=lambda name: dumps[name]):
        a, ha, _, _ = parts(["m0", "m1", "m2"], 0, 3)
        b, hb, _, _ = concat_parts(["m0", "m1", "m2"], 0, 3)
    ar = np.argsort(-(a - 0.3 * ha), axis=1, kind="stable")[:, :10]
    br = np.argsort(-(b - 0.5 * hb), axis=1, kind="stable")[:, :10]
    return {
        "kind": "synthetic_counterexample_not_model_quality",
        "n_query": nq, "n_gallery": ng, "component_dimension": dim,
        "n_components": 3, "hub_k": 3,
        "row_z_sum_hub_lambda": 0.3, "concat_single_hub_lambda": 0.5,
        "different_top1_queries": int(np.sum(ar[:, 0] != br[:, 0])),
        "different_top10_order_queries": int(np.sum(np.any(ar != br, axis=1))),
        "conclusion": "The two algorithms are not mathematically equivalent; no real-data effect size is inferred.",
    }


def verifier_window_probe(rng: np.random.Generator) -> dict:
    sim = np.sort(rng.normal(size=(100, 30)), axis=1)[:, ::-1]
    logits = rng.normal(size=(100, 30))
    scores10 = z(sim[:, :10]) + 0.2 * z(logits[:, :10])
    scores30 = z(sim) + 0.2 * z(logits)
    order10 = np.argsort(-scores10, axis=1, kind="stable")
    # Isolate normalization from the separate effect of expanding the candidates.
    order30restricted = np.argsort(-scores30[:, :10], axis=1, kind="stable")
    order30 = np.argsort(-scores30, axis=1, kind="stable")[:, :10]
    return {
        "kind": "synthetic_counterexample_not_model_quality", "n_query": 100,
        "weight": 0.2, "windows": [10, 30],
        "normalization_only_different_top1_queries": int(np.sum(order10[:, 0] != order30restricted[:, 0])),
        "normalization_only_different_top10_order_queries": int(np.sum(np.any(order10 != order30restricted, axis=1))),
        "expanded_window_different_top1_queries": int(np.sum(order10[:, 0] != order30[:, 0])),
        "expanded_window_different_top10_order_queries": int(np.sum(np.any(order10 != order30, axis=1))),
    }


def split_probe() -> dict:
    rows = []
    for fold in range(5):
        truth = FoldTruth.load(fold, str(ROOT / "splits/folds.csv"))
        even = {truth.vid[q] for q in truth.query_ids[::2]}
        odd = {truth.vid[q] for q in truth.query_ids[1::2]}
        counts = {}
        for g in truth.gallery_ids:
            key = (truth.vid[g], truth.cam[g])
            counts[key] = counts.get(key, 0) + 1
        rows.append({
            "fold": fold, "n_query": len(truth.query_ids),
            "even_query_vehicle_count": len(even), "odd_query_vehicle_count": len(odd),
            "overlapping_vehicle_count": len(even & odd),
            "union_vehicle_count": len(even | odd),
            "gallery_rows": len(truth.gallery_ids),
            "unique_gallery_vehicle_camera_pairs": len(counts),
            "max_frames_per_gallery_vehicle_camera": max(counts.values()),
            "duplicate_gallery_vehicle_camera_groups": sum(n > 1 for n in counts.values()),
        })
    return {"kind": "committed_train_splits_only", "folds": rows}


def junk_probe() -> dict:
    gallery = ["junk"] + [f"n{i}" for i in range(9)] + ["positive"]
    vid = {"q": 1, "junk": 1, "positive": 1, **{f"n{i}": i + 2 for i in range(9)}}
    cam = {"q": 1, "junk": 1, "positive": 2, **{f"n{i}": 3 for i in range(9)}}
    truth = FoldTruth(["q"], gallery, vid, cam)
    return {
        "kind": "toy_ranking_not_model_quality",
        "strict_top10_map10": ranking_metrics(truth, {"q": gallery[:10]})["map10"],
        "long_list_map10": ranking_metrics(truth, {"q": gallery})["map10"],
        "description": "Junk first, nine negatives, sole cross-camera positive at position 11.",
        "conclusion": "Long-list inflation is ranking-dependent, not a universal correction constant.",
    }


def bench_probe(rng: np.random.Generator) -> dict:
    truth = FoldTruth.load(0, str(ROOT / "splits/folds.csv"))
    candidates = []
    for q in truth.query_ids:
        same = [g for g in truth.gallery_ids if truth.vid[g] == truth.vid[q]]
        other = [g for g in truth.gallery_ids if truth.vid[g] != truth.vid[q]]
        candidates.append(same + rng.choice(other, 30 - len(same), replace=False).tolist())
    candidates = np.array(candidates)
    scores = rng.normal(size=candidates.shape)
    synthetic = {
        "fold": 0, "query_ids": np.array(truth.query_ids), "candidates": candidates,
        "sim": scores, "logits": np.zeros_like(scores),
    }
    with patch("numpy.load", return_value=synthetic), patch.object(FoldTruth, "load", return_value=truth):
        bench = Bench("in_memory_synthetic_ranking")
    order = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    ranking = {q: candidates[i, indices].tolist()
               for i, (q, indices) in enumerate(zip(truth.query_ids, order))}
    canonical = ranking_metrics(truth, ranking)["map10"]
    comparison = bench.map10(scores)
    delta = abs(canonical - comparison)
    if delta > 1e-12:
        raise AssertionError(f"Research Bench differs from canonical eval_local: {delta}")
    return {
        "kind": "synthetic_scores_on_train_ids_not_model_quality", "fold": 0,
        "candidate_window": 30, "n_query": len(truth.query_ids),
        "eval_local_synthetic_map10": canonical, "bench_synthetic_map10": comparison,
        "absolute_difference": delta, "tolerance": 1e-12,
    }


def source_metadata() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            # The offline runtime intentionally contains neither git nor .git.
            # Retain file digests and report unknown provenance, not a fake SHA.
            return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "tracked_worktree_dirty": None if status is None else bool(status),
        "source_file_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES
        },
    }


def build_report() -> dict:
    rng = np.random.default_rng(SEED)
    return {
        "schema_version": 1, "seed": SEED,
        "scope": "CPU synthetic counterexamples and committed train-fold metadata only; no trained-model quality, test inputs, images, checkpoint access or GPU.",
        "numpy_version": np.__version__,
        **source_metadata(),
        "fusion_algorithms": fusion_probe(rng),
        "verifier_window": verifier_window_probe(rng),
        "train_split_structure": split_probe(),
        "junk_top10": junk_probe(),
        "research_bench_crosscheck": bench_probe(rng),
    }


def write_report(out: Path, report: dict) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New JSON file; existing paths are refused.")
    args = parser.parse_args()
    try:
        write_report(args.out, build_report())
    except FileExistsError:
        parser.error(f"refusing to replace existing output: {args.out}")
    print(f"Saved CPU protocol counterexamples (not model quality): {args.out}")


if __name__ == "__main__":
    main()
