"""Small train-only D082 full-pipeline query-isolation diagnostic.

Uses six prefix query rows: two anchors, two neighbours and two replacements.
Runs three four-query variants with the same gallery. No quality measurement,
model selection, threshold tuning, test/control2 access or acceptance claim.
The optional batch1 comparison is separate from fixed-shape isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from reid.infer.d082 import load_d082_runtime
from reid.infer.observability import configure_logging
from reid.infer.run import read_inputs, run_pipeline
from scripts.audit_release import audit_submission, file_record
from scripts.check_determinism import configure_determinism, output_hashes
from scripts.check_query_content_isolation import compare_anchors
from scripts.check_query_independence import git_commit, snapshot, write_inputs


BATCH_SIZE = 4
SEED = 2026


def _gallery_identity(destination, n_query):
    embeddings = np.load(
        destination / "embeddings.npy", allow_pickle=False, mmap_mode="r"
    )
    gallery = np.ascontiguousarray(embeddings[n_query:])
    return {
        "sha256": hashlib.sha256(gallery.tobytes()).hexdigest(),
        "shape": list(gallery.shape),
        "dtype": str(gallery.dtype),
    }


def execute(args):
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {args.out}")
    if args.batch_size != BATCH_SIZE or type(args.batch_size) is not int:
        raise ValueError("fixed-shape diagnostic requires batch-size4")
    if args.gallery_limit is not None and (
        type(args.gallery_limit) is not int or args.gallery_limit < 10
    ):
        raise ValueError(
            "gallery-limit must be at least10, or omitted for the full gallery"
        )
    queries = read_inputs(args.query, args.images)
    gallery = read_inputs(args.gallery, args.images)
    if len(queries) < 6 or len(gallery) < 10:
        raise ValueError("need at least6 query rows and10 gallery rows")
    if args.gallery_limit is not None:
        if args.gallery_limit > len(gallery):
            raise ValueError("gallery-limit exceeds available gallery")
        selected_gallery = gallery[: args.gallery_limit]
    else:
        selected_gallery = gallery
    # Validate every supplied ID, not just the selected prefix. This invokes
    # the real frozen-recipe/OOF/allowlist gate before any encoder forward.
    runtime = load_d082_runtime(
        args.model,
        args.d082_manifest_sha256,
        [r.image_id for r in queries],
        [r.image_id for r in gallery],
    )
    identity = runtime.identity
    del runtime
    torch.set_num_threads(1)
    configure_determinism(SEED)
    args.out.mkdir(parents=True, exist_ok=False)
    gallery_csv = args.out / "gallery.csv"
    write_inputs(gallery_csv, selected_gallery)
    anchors, neighbours, replacements = queries[:2], queries[2:4], queries[4:6]
    replaced = [*anchors, *replacements]
    variants = [
        ("baseline", [*anchors, *neighbours], BATCH_SIZE),
        ("replaced_neighbours", replaced, BATCH_SIZE),
        ("reversed_replacement", list(reversed(replaced)), BATCH_SIZE),
    ]
    if args.strict_batch1:
        variants.append(("batch1", [*anchors, *neighbours], 1))
    anchor_ids = [r.image_id for r in anchors]
    baseline, baseline_gallery, records = None, None, {}
    for name, rows, batch in variants:
        destination = args.out / name
        destination.mkdir()
        query_csv = destination / "query.csv"
        write_inputs(query_csv, rows)
        configure_determinism(SEED)
        started = time.perf_counter()
        # Actual CLI pipeline, including fresh encoder/gallery/verifier work
        # and its runtime gate. No mocked/injected scorer or fused C4 replay.
        run_pipeline(
            images_dir=args.images,
            query_csv=query_csv,
            gallery_csv=gallery_csv,
            out_dir=destination,
            model_dir=args.model,
            device=args.device,
            batch_size=batch,
            candidate_threshold=None,
            d082_manifest_sha256=args.d082_manifest_sha256,
        )
        duration = time.perf_counter() - started
        c6 = audit_submission(query_csv, gallery_csv, destination)
        state = snapshot(destination, rows)
        gallery_identity = _gallery_identity(destination, len(rows))
        if baseline is None:
            baseline, baseline_gallery = state, gallery_identity
        compared = list(baseline) if name == "batch1" else anchor_ids
        comparison = compare_anchors(
            {i: baseline[i] for i in compared}, {i: state[i] for i in compared}
        )
        records[name] = {
            "batch_size": batch,
            "query_ids": [r.image_id for r in rows],
            "compared_query_ids": compared,
            "comparison": comparison,
            "c6": c6,
            "input_query": file_record(query_csv),
            "output_sha256": output_hashes(destination),
            "gallery_embeddings": gallery_identity,
            "gallery_embeddings_bitwise_equal": gallery_identity == baseline_gallery,
            "elapsed_seconds": duration,
        }
        print(
            f"{name}: {'PASS' if comparison['passed'] else 'DIFF'}; C6 PASS", flush=True
        )
    fixed_names = ("baseline", "replaced_neighbours", "reversed_replacement")
    fixed_pass = all(
        records[n]["comparison"]["passed"]
        and records[n]["gallery_embeddings_bitwise_equal"]
        for n in fixed_names
    )
    strict = {
        "executed": args.strict_batch1,
        "passed": None,
        "scope": "Separate batch1-vs4 numeric diagnostic; not the historical batch1/32 gate",
    }
    if args.strict_batch1:
        strict["passed"] = (
            records["batch1"]["comparison"]["passed"]
            and records["batch1"]["gallery_embeddings_bitwise_equal"]
        )
    passed = fixed_pass and (not args.strict_batch1 or strict["passed"])
    report = {
        "kind": "d082_train_query_isolation_diagnostic",
        "passed": passed,
        "fixed_shape_isolation_passed": fixed_pass,
        "strict_batch_shape": strict,
        "runtime": identity,
        "model_manifest_sha256": args.d082_manifest_sha256,
        "source_commit": git_commit(),
        "script": file_record(Path(__file__)),
        "inputs": {
            "query": file_record(args.query),
            "gallery": file_record(args.gallery),
            "selected_gallery": file_record(gallery_csv),
        },
        "anchor_ids": anchor_ids,
        "neighbour_ids": [r.image_id for r in neighbours],
        "replacement_ids": [r.image_id for r in replacements],
        "gallery_count": len(selected_gallery),
        "gallery_limit": args.gallery_limit,
        "gallery_reextracted_each_variant": True,
        "device": args.device,
        "gpu_check_executed": args.device == "cuda",
        "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "gpu_capability": list(torch.cuda.get_device_capability())
        if args.device == "cuda"
        else None,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "seed": SEED,
        "torch_threads": 1,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "variants": records,
        "fold2": "sealed; allowlist enforced before forward",
        "scope": "Prefix fixture only, no quality metrics or tuning. Exact ID-aligned anchor embeddings, top10 and candidate rows; no tolerance. Full gallery by default; an explicit prefix is a reduced fixture.",
        "limitations": [
            "Not a GPU/Docker/A5000 acceptance, full-fold quality run or proof for all inputs",
            "Historical strict batch1/32 FAIL is not superseded by this fixed-shape or optional batch1/4 diagnostic",
            "A numeric difference alone does not prove cross-query information leakage",
            "Elapsed durations include cold loading and are not an inference speed benchmark",
        ],
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "images", "query", "gallery", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--d082-manifest-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--batch-size", type=int, choices=(BATCH_SIZE,), default=BATCH_SIZE
    )
    parser.add_argument(
        "--gallery-limit", type=int, help="explicit prefix>=10; default whole gallery"
    )
    parser.add_argument(
        "--strict-batch1",
        action="store_true",
        help="also report separate strict batch1-vs4",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")
    try:
        report = execute(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"CHECK FAILED: {exc}")
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
