"""Train-only fusion acceptance: vary query batching/order/neighbours, not gallery.

Reports embedding drift without demanding bitwise embeddings. Top-10 and complete
candidate CSV rows (including confidence) must match exactly after query-ID alignment.
No quality metrics, test inputs, parameter search, or production changes.
An exact-confidence mismatch may be normal FP16 batch-shape rounding; by itself
it does NOT prove prohibited cross-query statistics or a competition violation.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from reid.infer.extractor import Extractor
from reid.infer.fusion import load_fusion_config
from reid.infer.run import CSV_HEADER, extract_rows, rank_embeddings, write_outputs
from reid.post.policy import candidate_threshold_for, rerank_weight_for
from reid.post.verifier import PairVerifier
from scripts.audit_release import audit_submission, file_record
from scripts.check_determinism import configure_determinism
from scripts.diagnose_c3_gap import train_rows


def write_inputs(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows((row.image_id, *row.bbox) for row in rows)


def snapshot(out, query_rows):
    ids = [row.image_id for row in query_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Query IDs must be unique")
    with (out / "submission.csv").open(newline="") as handle:
        rankings = {row[0]: tuple(row[1:]) for row in csv.reader(handle)}
    candidates = {iid: [] for iid in ids}
    with (out / "candidates.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            candidates[row["query_id"]].append((row["gallery_id"], row["confidence"]))
    embeddings = np.load(out / "embeddings.npy", allow_pickle=False, mmap_mode="r")
    return {iid: {"ranking": rankings[iid], "candidates": candidates[iid],
                  "embedding": embeddings[index].copy()} for index, iid in enumerate(ids)}


def compare(baseline, current):
    if not set(baseline) <= set(current):
        raise ValueError("Comparison is missing baseline query IDs")
    changed_rank, changed_candidates, changed_ids, changed_acceptance = [], [], [], []
    drift, confidence_drift = [], []
    for iid, first in baseline.items():
        second = current[iid]
        if first["ranking"] != second["ranking"]:
            changed_rank.append(iid)
        if first["candidates"] != second["candidates"]:
            changed_candidates.append(iid)
        first_ids = [row[0] for row in first["candidates"]]
        second_ids = [row[0] for row in second["candidates"]]
        if bool(first_ids) != bool(second_ids):
            changed_acceptance.append(iid)
        if first_ids != second_ids:
            changed_ids.append(iid)
        elif first_ids:
            confidence_drift.extend(abs(float(a[1]) - float(b[1]))
                                    for a, b in zip(first["candidates"], second["candidates"]))
        a, b = first["embedding"].astype(np.float64), second["embedding"].astype(np.float64)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Invalid or mismatched embedding shapes")
        drift.append(float(np.max(np.abs(a - b))))
    return {"passed": not changed_rank and not changed_candidates,
            "strict_byte_equivalence": not changed_rank and not changed_candidates,
            "strict_byte_equivalence_scope": "query-ID-aligned ranking/candidate fields; not raw file order or embeddings",
            "semantic_ranked_ids_equal": not changed_rank,
            "semantic_candidate_ids_equal": not changed_ids,
            "semantic_acceptance_equal": not changed_acceptance,
            "compared_queries": len(baseline), "changed_top10_query_ids": changed_rank,
            "changed_candidate_rows_query_ids": changed_candidates,
            "changed_candidate_ids_or_acceptance_query_ids": changed_ids,
            "changed_acceptance_query_ids": changed_acceptance,
            "max_abs_embedding_drift": max(drift, default=0.0),
            "max_abs_confidence_drift_for_same_candidates": max(confidence_drift, default=0.0),
            "criterion": "exact top10 and candidate rows by query ID; embedding equality is not required",
            "interpretation": "Confidence-only ULP drift can result from FP16 batch-shape rounding; "
                              "a strict failure alone is not evidence of prohibited query statistics"}


def execute(extractor, verifier, fixed, extras, gallery, out):
    """Run actual Extractor/PairVerifier interfaces; separated for CPU mock tests."""
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    if not fixed or not extras or len({r.image_id for r in fixed + extras}) != len(fixed) + len(extras):
        raise ValueError("Need unique, disjoint fixed and extra query IDs")
    cfg = extractor.cfg
    if "rerank_weight" not in cfg or "candidate_threshold" not in cfg or rerank_weight_for(cfg) <= 0:
        raise ValueError("Artifact must explicitly configure threshold and enabled verifier weight")
    threshold = candidate_threshold_for(cfg)
    out.mkdir(parents=True, exist_ok=False)
    gallery_csv = out / "gallery.csv"
    write_inputs(gallery_csv, gallery)
    gallery_emb = extract_rows(extractor, gallery, 32)
    interleaved = []
    for index, row in enumerate(fixed):
        if index < len(extras):
            interleaved.append(extras[index])
        interleaved.append(row)
    interleaved.extend(extras[len(fixed):])
    variants = [("batch1", fixed, 1), ("batch32", fixed, 32),
                ("reverse", list(reversed(fixed)), 32), ("with_extra_queries", interleaved, 32)]
    baseline, records = None, {}
    for name, rows, batch in variants:
        destination = out / name
        destination.mkdir()
        query_csv = destination / "query.csv"
        write_inputs(query_csv, rows)
        query_emb = extract_rows(extractor, rows, batch)
        result = rank_embeddings(rows, gallery, query_emb, gallery_emb, cfg, verifier)
        write_outputs(destination, rows, gallery, query_emb, gallery_emb, threshold, result)
        c6 = audit_submission(query_csv, gallery_csv, destination)
        (destination / "c6.json").write_text(json.dumps(c6, indent=2) + "\n")
        state = snapshot(destination, rows)
        if baseline is None:
            baseline = state
        comparison = compare(baseline, state)
        records[name] = {"batch_size": batch, "query_ids": [r.image_id for r in rows],
                         "c6": c6, "comparison": comparison}
        print(f"{name}: {'PASS' if comparison['passed'] else 'DIFF'}", flush=True)
    return {"passed": all(r["comparison"]["passed"] for r in records.values()),
            "gallery_extracted_once_batch": 32, "gallery_count": len(gallery),
            "fixed_query_ids": [r.image_id for r in fixed], "extra_query_ids": [r.image_id for r in extras],
            "policy": {key: cfg.get(key) for key in ("candidate_policy", "candidate_threshold", "rerank_mode", "rerank_weight")},
            "variants": records}


def source_records(model, verifier):
    cfg = load_fusion_config(model)
    paths = [model / "fusion_config.json", verifier]
    for component in cfg["components"]:
        paths.extend(model / component["path"] / name for name in ("model_config.yaml", "weights.safetensors"))
    return [{"path": str(path), **file_record(path)} for path in paths]


def git_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--queries", type=int, choices=range(20, 33), default=24)
    parser.add_argument("--extras", type=int, choices=range(1, 33), default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    extractor = verifier = None
    try:
        if args.out.exists() or args.out.is_symlink():
            raise FileExistsError(f"Refusing to overwrite: {args.out}")
        if args.workers <= 0 or args.torch_threads <= 0:
            raise ValueError("Worker/thread counts must be positive")
        torch.set_num_threads(args.torch_threads)
        query, gallery = train_rows(args.data_root, args.splits, args.fold)
        if len(query) < args.queries + args.extras:
            raise ValueError("Train fold has insufficient distinct query rows")
        source = source_records(args.model, args.verifier)
        configure_determinism(2026)
        extractor = Extractor(args.model, args.device, args.workers)
        verifier = PairVerifier(args.verifier, args.device, args.workers)
        report = execute(extractor, verifier, query[:args.queries],
                         query[args.queries:args.queries + args.extras], gallery, args.out)
        report.update({"kind": "train_query_independence_acceptance", "fold": args.fold,
                       "code_commit": git_commit(), "sources": source, "device": args.device,
                       "torch_version": str(torch.__version__), "deterministic_algorithms": True,
                       "script": file_record(Path(__file__)),
                       "inputs": {str(p): file_record(p) for p in (args.data_root / "train.csv", args.splits / "folds.csv",
                                  args.splits / f"fold{args.fold}_query.csv", args.splits / f"fold{args.fold}_gallery.csv")},
                       "scope": "Fixed train-fold prefix; structural independence check, NOT quality evaluation, calibration, "
                                "proof for all possible inputs or test-set use. GPU floating-point drift is reported, not hidden."})
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return 0 if report["passed"] else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"CHECK FAILED: {exc}")
        return 2
    finally:
        if extractor is not None and extractor._preprocess_pool is not None:
            extractor._preprocess_pool.shutdown()
        if verifier is not None and verifier._pool is not None:
            verifier._pool.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
