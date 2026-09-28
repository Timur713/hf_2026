#!/usr/bin/env python3
"""Run C5/C6 with a deterministic solid fill over the approximate plate area."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import torch

from reid.infer.plate_mask import (
    DEFAULT_PLATE_MASK,
    PlateMaskConfig,
    PlateMaskedExtractor,
    PlateMaskedPairVerifier,
)
from reid.infer.run import CSV_HEADER, read_inputs, run_pipeline
from reid.post.policy import candidate_threshold_for


OUTPUT_FILES = ("submission.csv", "embeddings.npy", "candidates.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subset_csv(source: Path, destination: Path, limit: int | None) -> Path:
    if limit is None:
        return source
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(f"{source}: expected header {CSV_HEADER!r}, got {reader.fieldnames!r}")
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError(f"{source}: requested {limit} rows, found {len(rows)}")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def run_control(
    *,
    images: Path,
    query: Path,
    gallery: Path,
    out: Path,
    model: Path,
    verifier: Path | None,
    device: str,
    batch_size: int,
    candidate_threshold: float | None,
    rerank_weight: float | None,
    mask: PlateMaskConfig,
) -> float:
    extractor = PlateMaskedExtractor(model, device, mask=mask)
    active_verifier = (
        PlateMaskedPairVerifier(verifier, device, mask=mask) if verifier is not None else None
    )
    active_threshold = candidate_threshold_for(extractor.cfg, candidate_threshold)
    run_pipeline(
        images,
        query,
        gallery,
        out,
        model,
        device,
        batch_size,
        active_threshold,
        extractor=extractor,
        verifier=active_verifier,
        rerank_weight=rerank_weight,
    )
    return active_threshold


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=Path(os.environ.get("MODEL_DIR", "artifacts/models/final"))
    )
    verifier_default = os.environ.get("VERIFIER_PATH")
    parser.add_argument(
        "--verifier", type=Path, default=Path(verifier_default) if verifier_default else None
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-threshold", type=float)
    parser.add_argument("--rerank-weight", type=float, default=None)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--gallery-limit", type=int)
    parser.add_argument("--mask-width-ratio", type=float, default=DEFAULT_PLATE_MASK.width_ratio)
    parser.add_argument("--mask-aspect-ratio", type=float, default=DEFAULT_PLATE_MASK.aspect_ratio)
    parser.add_argument("--mask-center-x", type=float, default=DEFAULT_PLATE_MASK.center_x)
    parser.add_argument("--mask-center-y", type=float, default=DEFAULT_PLATE_MASK.center_y)
    parser.add_argument("--mask-fill", type=int, default=DEFAULT_PLATE_MASK.fill)
    parser.add_argument("--report", type=Path)
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

    try:
        mask = PlateMaskConfig(
            width_ratio=args.mask_width_ratio,
            aspect_ratio=args.mask_aspect_ratio,
            center_x=args.mask_center_x,
            center_y=args.mask_center_y,
            fill=args.mask_fill,
        )
        with tempfile.TemporaryDirectory(prefix="reid-plate-mask-") as temp_dir:
            temp_root = Path(temp_dir)
            query = _subset_csv(args.query, temp_root / "query.csv", args.query_limit)
            gallery = _subset_csv(args.gallery, temp_root / "gallery.csv", args.gallery_limit)
            query_count = len(read_inputs(query, args.images))
            gallery_count = len(read_inputs(gallery, args.images))
            active_threshold = run_control(
                images=args.images,
                query=query,
                gallery=gallery,
                out=args.out,
                model=args.model,
                verifier=args.verifier,
                device=args.device,
                batch_size=args.batch_size,
                candidate_threshold=args.candidate_threshold,
                rerank_weight=args.rerank_weight,
                mask=mask,
            )
        report = {
            "mode": "plate_mask_control",
            "model": str(args.model),
            "verifier": str(args.verifier) if args.verifier is not None else None,
            "device": args.device,
            "query_count": query_count,
            "gallery_count": gallery_count,
            "candidate_threshold": active_threshold,
            "mask": mask.as_dict(),
            "sha256": {name: _sha256(args.out / name) for name in OUTPUT_FILES},
        }
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote masked control outputs for {query_count} query / {gallery_count} gallery "
        f"to {args.out}"
    )
    if args.report is not None:
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
