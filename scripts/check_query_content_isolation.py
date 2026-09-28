"""Supplementary train-only query-content isolation check at fixed batch shape.

Exactly three 32-query batches: eight fixed anchors plus 24 neighbours, replacement
of all 24 neighbours, then reversal of the replacement batch. Gallery is extracted
once and remains fixed. Anchor embeddings, top-10 and candidate rows must be exact.
This does not replace check_query_independence.py or its batch=1/32 strict result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from reid.infer.extractor import Extractor
from reid.infer.run import extract_rows, rank_embeddings, write_outputs
from reid.post.policy import candidate_threshold_for, rerank_weight_for
from reid.post.verifier import PairVerifier
from scripts.audit_release import audit_submission, file_record
from scripts.check_determinism import configure_determinism
from scripts.check_query_independence import compare, git_commit, snapshot, source_records, write_inputs
from scripts.diagnose_c3_gap import train_rows


BATCH_SIZE = 32
ANCHOR_COUNT = 8
NEIGHBOUR_COUNT = BATCH_SIZE - ANCHOR_COUNT
REQUIRED_QUERY_COUNT = ANCHOR_COUNT + 2 * NEIGHBOUR_COUNT
SEED = 2026


def compare_anchors(baseline, current):
    """Preserve the existing exact-output comparison and additionally require bits."""
    report = compare(baseline, current)
    changed = []
    for iid, state in baseline.items():
        first, second = state["embedding"], current[iid]["embedding"]
        if first.dtype != second.dtype or first.tobytes() != second.tobytes():
            changed.append(iid)
    report["exact_output_fields_equal"] = report["passed"]
    report["anchor_embeddings_bitwise_equal"] = not changed
    report["changed_embedding_query_ids"] = changed
    report["passed"] = report["passed"] and not changed
    report["criterion"] = "fixed-shape query-ID-aligned anchor embeddings and top10/candidate fields exactly equal"
    report["interpretation"] = (
        "Fixed-shape content/permutation diagnostic only; a difference requires investigation, "
        "not attribution to cross-query leakage without further evidence. No tolerance is applied."
    )
    return report


def execute(extractor, verifier, anchors, neighbours, replacements, gallery, out):
    """Exercise actual extraction and production postprocessing, with CPU mocks allowed."""
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    if (len(anchors), len(neighbours), len(replacements)) != (ANCHOR_COUNT, NEIGHBOUR_COUNT, NEIGHBOUR_COUNT):
        raise ValueError("Need exactly 8 anchors and two groups of 24 neighbours")
    groups = [*anchors, *neighbours, *replacements, *gallery]
    ids = [row.image_id for row in groups]
    if any(not isinstance(iid, str) or not iid for iid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Anchor, neighbour, replacement and gallery IDs must be unique and disjoint")
    if len(gallery) < 10:
        raise ValueError("Gallery must contain at least ten images")
    cfg = extractor.cfg
    if (cfg.get("candidate_policy") != "gallery_z" or "candidate_threshold" not in cfg
            or "rerank_weight" not in cfg or rerank_weight_for(cfg) <= 0 or verifier is None):
        raise ValueError("Need explicit fusion threshold, gallery_z policy and enabled verifier")
    threshold = candidate_threshold_for(cfg)
    out.mkdir(parents=True, exist_ok=False)
    gallery_csv = out / "gallery.csv"
    write_inputs(gallery_csv, gallery)
    gallery_emb = extract_rows(extractor, gallery, BATCH_SIZE)
    gallery_emb.setflags(write=False)
    rows = [*anchors, *replacements]
    variants = [("baseline", [*anchors, *neighbours]), ("replaced_neighbours", rows),
                ("reversed_replacement", list(reversed(rows)))]
    anchor_ids = [row.image_id for row in anchors]
    baseline, records = None, {}
    for name, query_rows in variants:
        destination = out / name
        destination.mkdir()
        query_csv = destination / "query.csv"
        write_inputs(query_csv, query_rows)
        query_emb = extract_rows(extractor, query_rows, BATCH_SIZE)
        result = rank_embeddings(query_rows, gallery, query_emb, gallery_emb, cfg, verifier)
        write_outputs(destination, query_rows, gallery, query_emb, gallery_emb, threshold, result)
        c6 = audit_submission(query_csv, gallery_csv, destination)
        (destination / "c6.json").write_text(json.dumps(c6, indent=2) + "\n")
        full_state = snapshot(destination, query_rows)
        state = {iid: full_state[iid] for iid in anchor_ids}
        if baseline is None:
            baseline = state
        comparison = compare_anchors(baseline, state)
        records[name] = {"batch_size": BATCH_SIZE, "query_ids": [r.image_id for r in query_rows],
                         "anchor_positions": [next(i for i, row in enumerate(query_rows) if row.image_id == iid)
                                              for iid in anchor_ids],
                         "comparison": comparison, "c6": c6}
        print(f"{name}: {'PASS' if comparison['passed'] else 'DIFF'}", flush=True)
    return {
        "passed": all(record["comparison"]["passed"] for record in records.values()),
        "fixed_shape_content_isolation_passed": records["replaced_neighbours"]["comparison"]["passed"],
        "fixed_shape_content_and_permutation_passed": records["reversed_replacement"]["comparison"]["passed"],
        "batch_size": BATCH_SIZE, "query_batch_count": len(variants),
        "gallery_extracted_once_batch": BATCH_SIZE, "gallery_count": len(gallery),
        "gallery_embedding_raw_bytes_sha256": hashlib.sha256(gallery_emb.tobytes(order="C")).hexdigest(),
        "gallery_embedding_shape": list(gallery_emb.shape), "gallery_embedding_dtype": str(gallery_emb.dtype),
        "anchor_ids": anchor_ids, "neighbour_ids": [row.image_id for row in neighbours],
        "replacement_ids": [row.image_id for row in replacements],
        "policy": {key: cfg.get(key) for key in ("candidate_policy", "candidate_threshold", "rerank_mode", "rerank_weight")},
        "variants": records,
        "scope": "Eight fixed train anchors; same-shape neighbour replacement and permutation only. "
                 "NOT a quality evaluation, calibration, cross-shape equality test or proof for all possible inputs. "
                 "The existing check_query_independence.py strict diagnostic is unchanged and must be reported separately.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
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
        if len(query) < REQUIRED_QUERY_COUNT:
            raise ValueError("Train fold needs at least 56 distinct query rows")
        sources = source_records(args.model, args.verifier)
        configure_determinism(SEED)
        extractor = Extractor(args.model, args.device, args.workers)
        verifier = PairVerifier(args.verifier, args.device, args.workers)
        report = execute(extractor, verifier, query[:ANCHOR_COUNT], query[ANCHOR_COUNT:BATCH_SIZE],
                         query[BATCH_SIZE:REQUIRED_QUERY_COUNT], gallery, args.out)
        device = extractor.device
        report.update({
            "kind": "train_query_content_isolation_acceptance", "fold": args.fold,
            "code_commit": git_commit(), "sources": sources, "seed": SEED,
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_check_executed": device.type == "cuda", "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "script": file_record(Path(__file__)),
            "inputs": {str(path): file_record(path) for path in (
                args.data_root / "train.csv", args.splits / "folds.csv",
                args.splits / f"fold{args.fold}_query.csv", args.splits / f"fold{args.fold}_gallery.csv")},
        })
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
