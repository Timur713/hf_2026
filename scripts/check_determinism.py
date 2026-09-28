#!/usr/bin/env python3
"""Run the complete inference pipeline twice and compare C6 artifacts byte-for-byte."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

# Required by CUDA deterministic GEMM. It must be set before the first CUDA call.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from reid.infer.extractor import Extractor
from reid.infer.run import CSV_HEADER, run_pipeline
from reid.post.verifier import PairVerifier


OUTPUT_FILES = ("submission.csv", "embeddings.npy", "candidates.csv")
DEFAULT_SEED = 2026


def configure_determinism(seed: int) -> None:
    """Reset every RNG used by inference and request deterministic Torch kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_hashes(out_dir: Path) -> dict[str, str]:
    missing = [name for name in OUTPUT_FILES if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing inference outputs in {out_dir}: {', '.join(missing)}")
    return {name: sha256_file(out_dir / name) for name in OUTPUT_FILES}


def compare_outputs(first: Path, second: Path) -> dict[str, dict[str, object]]:
    """Return per-artifact hashes and exact-equality flags."""

    first_hashes, second_hashes = output_hashes(first), output_hashes(second)
    return {
        name: {
            "run_1_sha256": first_hashes[name],
            "run_2_sha256": second_hashes[name],
            "identical": first_hashes[name] == second_hashes[name],
        }
        for name in OUTPUT_FILES
    }


def subset_csv(source: Path, destination: Path, limit: int | None) -> int:
    """Copy a deterministic prefix without inspecting labels or test statistics."""

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(f"{source}: expected header {CSV_HEADER!r}, got {reader.fieldnames!r}")
        rows = []
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) == limit:
                break
    if not rows:
        raise ValueError(f"{source}: input CSV has no data rows")
    if limit is not None and len(rows) != limit:
        raise ValueError(f"{source}: requested {limit} rows, found {len(rows)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare_cuda_measurement(device: torch.device) -> float:
    """Initialize the allocator before resetting its counters (cold-process safe)."""
    if device.type != "cuda":
        return 0.0
    started = time.perf_counter()
    torch.cuda.init()
    _sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter() - started


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def run_once(
    *,
    run_index: int,
    images: Path,
    query: Path,
    gallery: Path,
    out_dir: Path,
    model: Path,
    verifier: Path | None,
    device_name: str,
    batch_size: int,
    candidate_threshold: float | None,
    rerank_weight: float | None,
    seed: int,
) -> dict[str, object]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {out_dir}")
    configure_determinism(seed)
    device = torch.device(device_name)
    cuda_initialization_seconds = prepare_cuda_measurement(device)

    started = time.perf_counter()
    extractor = Extractor(model, device_name)
    _sync(device)
    model_load_seconds = time.perf_counter() - started

    active_verifier = None
    verifier_load_seconds = None
    if verifier is not None:
        started = time.perf_counter()
        active_verifier = PairVerifier(verifier, device_name)
        _sync(device)
        verifier_load_seconds = time.perf_counter() - started

    pipeline_started = time.perf_counter()
    run_pipeline(
        images,
        query,
        gallery,
        out_dir,
        model,
        device_name,
        batch_size,
        candidate_threshold,
        extractor=extractor,
        verifier=active_verifier,
        rerank_weight=rerank_weight,
    )
    _sync(device)
    pipeline_seconds = time.perf_counter() - pipeline_started
    hashes = output_hashes(out_dir)
    memory = None
    if device.type == "cuda":
        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "scope": "embedder + verifier + full pipeline; Torch allocator, not total process VRAM",
        }

    del active_verifier, extractor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "run": run_index,
        "cuda_initialization_seconds": cuda_initialization_seconds,
        "model_load_seconds": model_load_seconds,
        "verifier_load_seconds": verifier_load_seconds,
        "pipeline_seconds": pipeline_seconds,
        "cuda_memory": memory,
        "sha256": hashes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verifier", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-threshold", type=float)
    parser.add_argument("--rerank-weight", type=float, default=None)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--gallery-limit", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0:
        print("ERROR: --batch-size must be positive", file=sys.stderr)
        return 2
    if args.query_limit is not None and args.query_limit <= 0:
        print("ERROR: --query-limit must be positive", file=sys.stderr)
        return 2
    if args.gallery_limit is not None and args.gallery_limit < 10:
        print("ERROR: --gallery-limit must be at least 10 for top-10 output", file=sys.stderr)
        return 2
    if args.out.exists():
        print(f"ERROR: refusing to reuse output directory: {args.out}", file=sys.stderr)
        return 2

    try:
        args.out.mkdir(parents=True)
        input_dir = args.out / "inputs"
        query_csv, gallery_csv = input_dir / "query.csv", input_dir / "gallery.csv"
        query_count = subset_csv(args.query, query_csv, args.query_limit)
        gallery_count = subset_csv(args.gallery, gallery_csv, args.gallery_limit)

        runs = [
            run_once(
                run_index=index,
                images=args.images,
                query=query_csv,
                gallery=gallery_csv,
                out_dir=args.out / f"run_{index}",
                model=args.model,
                verifier=args.verifier,
                device_name=args.device,
                batch_size=args.batch_size,
                candidate_threshold=args.candidate_threshold,
                rerank_weight=args.rerank_weight,
                seed=args.seed,
            )
            for index in (1, 2)
        ]
        files = compare_outputs(args.out / "run_1", args.out / "run_2")
        identical = all(bool(value["identical"]) for value in files.values())
        device = torch.device(args.device)
        report = {
            "commit": _git_commit(),
            "seed": args.seed,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "device": {
                "requested": args.device,
                "name": torch.cuda.get_device_name(device) if device.type == "cuda" else args.device,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            "model": str(args.model),
            "verifier": str(args.verifier) if args.verifier is not None else None,
            "query_count": query_count,
            "gallery_count": gallery_count,
            "runs": runs,
            "files": files,
            "byte_identical": identical,
        }
        report_path = args.report or args.out / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for name, result in files.items():
        print(f"{name}: {'IDENTICAL' if result['identical'] else 'DIFF'} {result['run_1_sha256']}")
    print(f"model load: {runs[0]['model_load_seconds']:.3f}s / {runs[1]['model_load_seconds']:.3f}s")
    print(f"report: {report_path}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
