"""End-to-end vehicle ReID inference pipeline (contract C5).

The command reads query/gallery CSV files, extracts normalized embeddings,
performs deterministic cosine search, and writes the three C6 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from reid.eval.validate_outputs import validate_outputs
from reid.infer.extractor import Extractor
from reid.post.policy import DEFAULT_CANDIDATE_THRESHOLD, candidate_threshold_for, rerank_weight_for
from reid.post.search import SearchResult, cosine_search
from reid.post.verifier import PairScorer, PairVerifier, rerank_topk


CSV_HEADER = ["image_id", "x", "y", "w", "h"]
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass(frozen=True)
class InputRow:
    image_id: str
    image_path: Path
    bbox: tuple[float, float, float, float]


# Сколько кандидатов писать в candidates.csv для отвеченного запроса (D-048).
CANDIDATES_PER_QUERY = 3


def _image_path(images_dir: Path, image_id: str) -> Path:
    direct = images_dir / image_id
    if direct.is_file():
        return direct
    if direct.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        raise FileNotFoundError(f"image not found for {image_id!r}: {direct}")

    tried = []
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        path = images_dir / f"{image_id}{suffix}"
        tried.append(path.name)
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"image not found for {image_id!r} under {images_dir}; tried {', '.join(tried)}"
    )


def read_inputs(csv_path: Path, images_dir: Path) -> list[InputRow]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(f"{csv_path}: expected header {CSV_HEADER!r}, got {reader.fieldnames!r}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            image_id = row["image_id"]
            if not image_id:
                raise ValueError(f"{csv_path}:{line_number}: empty image_id")
            try:
                bbox = tuple(float(row[key]) for key in ("x", "y", "w", "h"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}:{line_number}: invalid bbox") from exc
            if not np.isfinite(bbox).all() or bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"{csv_path}:{line_number}: bbox must be finite with w,h > 0")
            rows.append(InputRow(image_id, _image_path(images_dir, image_id), bbox))
    ids = [row.image_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{csv_path}: image_id values must be unique")
    return rows


def extract_rows(extractor: Extractor, rows: Sequence[InputRow], batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        items = [(row.image_path, row.bbox) for row in batch]
        chunks.append(extractor.extract_batch(items))
    if not chunks:
        raise ValueError("input CSV has no data rows")
    result = np.concatenate(chunks).astype(np.float32, copy=False)
    if result.ndim != 2 or result.shape[0] != len(rows) or not np.isfinite(result).all():
        raise ValueError("extractor returned invalid embeddings")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("extractor returned a zero embedding")
    return (result / norms).astype(np.float32, copy=False)


def write_outputs(
    out_dir: Path,
    query_rows: Sequence[InputRow],
    gallery_rows: Sequence[InputRow],
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    candidate_threshold: float,
    search_result: SearchResult | None = None,
) -> None:
    if not np.isfinite(candidate_threshold):
        raise ValueError("candidate threshold must be finite")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = search_result or cosine_search(
        query_emb, gallery_emb, [row.image_id for row in gallery_rows]
    )

    with (out_dir / "submission.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row, ranked_ids in zip(query_rows, result.ranking):
            writer.writerow([row.image_id, *ranked_ids])

    np.save(out_dir / "embeddings.npy", np.concatenate([query_emb, gallery_emb]).astype(np.float32))

    with (out_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "gallery_id", "confidence"])
        for row, ranked_ids, confidence in zip(query_rows, result.ranking, result.top_scores):
            if float(confidence) < candidate_threshold:
                continue  # отказ кодируется полным отсутствием строк для этого query_id
            # Пишем несколько кандидатов, а не одного (D-048). Организаторы: кандидаты ниже
            # верхнего на F1 и TNR не влияют, поэтому на текущий эталон это ничего не меняет.
            # Но если junk начнут снимать из candidates.csv до выбора верхнего, наш единственный
            # ответ исчез бы и запрос стал отказом — а так его подхватит следующий кандидат.
            # Уверенность у запасных убывающая: точные сходства пайплайн наружу не отдаёт, а для
            # выбора верхнего важен только порядок.
            # Gallery-z is not bounded by one. Keep strict ordering even when
            # a float64 ULP exceeds the legacy 1e-6 decrement.
            step = max(1e-6, 2 * math.ulp(float(confidence)))
            for offset, gallery_id in enumerate(ranked_ids[:CANDIDATES_PER_QUERY]):
                score = float(confidence) - offset * step
                writer.writerow([row.image_id, gallery_id, f"{score:.8f}"])


def run_pipeline(
    images_dir: Path,
    query_csv: Path,
    gallery_csv: Path,
    out_dir: Path,
    model_dir: Path,
    device: str,
    batch_size: int,
    candidate_threshold: float | None,
    extractor: Extractor | None = None,
    verifier_path: Path | None = None,
    verifier: PairScorer | None = None,
    rerank_weight: float | None = None,
    rerank_mode: str | None = None,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    query_rows = read_inputs(query_csv, images_dir)
    gallery_rows = read_inputs(gallery_csv, images_dir)
    active_extractor = extractor or Extractor(model_dir, device)
    threshold = candidate_threshold_for(active_extractor.cfg, candidate_threshold)
    query_emb = extract_rows(active_extractor, query_rows, batch_size)
    gallery_emb = extract_rows(active_extractor, gallery_rows, batch_size)
    active_verifier = verifier
    if active_verifier is None and verifier_path is not None:
        active_verifier = PairVerifier(verifier_path, device)
    result = rank_embeddings(query_rows, gallery_rows, query_emb, gallery_emb,
                             active_extractor.cfg, active_verifier, rerank_weight, rerank_mode)
    write_outputs(out_dir, query_rows, gallery_rows, query_emb, gallery_emb, threshold, result)
    validate_outputs(query_csv, gallery_csv, out_dir / "submission.csv",
                     out_dir / "embeddings.npy", out_dir / "candidates.csv")


def rank_embeddings(
    query_rows: Sequence[InputRow], gallery_rows: Sequence[InputRow],
    query_emb: np.ndarray, gallery_emb: np.ndarray, cfg: dict,
    verifier: PairScorer | None = None, rerank_weight: float | None = None,
    rerank_mode: str | None = None,
) -> SearchResult:
    """One production postprocessing path, also used for audited C4 replay.

    Labels/camera IDs never enter this function. A replay may replace only the
    pair scorer with previously measured, exactly ID-aligned logits.
    """
    policy_name = cfg.get("candidate_policy", "cosine")
    if policy_name not in ("cosine", "gallery_z"):
        raise ValueError(f"unknown candidate policy: {policy_name!r}")
    mode = rerank_mode or cfg.get("rerank_mode", "raw")
    if mode not in ("raw", "standardized"):
        raise ValueError(f"unknown rerank mode: {mode!r}")
    weight = rerank_weight_for(cfg, rerank_weight)
    gallery_ids = [row.image_id for row in gallery_rows]
    gallery_policy = None
    if policy_name == "gallery_z":
        from reid.post.gallery_policy import GalleryPolicy

        # D-055/D-057: only the fixed gallery contributes to fitted statistics.
        gallery_policy = GalleryPolicy(gallery_emb, gallery_ids)
        result = gallery_policy.search(query_emb)
    else:
        result = cosine_search(query_emb, gallery_emb, gallery_ids)
    if verifier is not None:
        query_items = [(row.image_path, row.bbox) for row in query_rows]
        gallery_items = [(row.image_path, row.bbox) for row in gallery_rows]
        result = rerank_topk(
            query_items,
            gallery_items,
            query_emb,
            gallery_emb,
            gallery_ids,
            result,
            verifier,
            weight,
            score_mode=mode,
            ranking_scores=(
                gallery_policy.ranking_scores(query_emb) if gallery_policy is not None else None
            ),
        )
        if gallery_policy is not None:
            # Confidence belongs to the final selected gallery item, not to the
            # verifier's mixed ranking score. Never apply the old cosine cutoff.
            result = SearchResult(
                ranking=result.ranking,
                top_ids=result.top_ids,
                top_scores=gallery_policy.confidence_for(query_emb, result.top_ids),
            )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("MODEL_DIR", "artifacts/models/final")))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        help=(
            "threshold in the artifact's confidence units (cosine or gallery_z); "
            "defaults to artifact config or "
            f"the final five-fold calibrated value {DEFAULT_CANDIDATE_THRESHOLD:.3f} (D-044); "
            "published model-v1 retains 0.600"
        ),
    )
    verifier_default = os.environ.get("VERIFIER_PATH")
    parser.add_argument(
        "--verifier",
        type=Path,
        default=Path(verifier_default) if verifier_default else None,
        help="optional A3.3 verifier.pt or its containing directory",
    )
    parser.add_argument("--rerank-weight", type=float, default=None,
                        help="artifact rerank_weight, or legacy 0.2 if unspecified")
    parser.add_argument(
        "--rerank-mode", choices=("raw", "standardized"), default=None,
        help="artifact default; legacy raw or D-053 per-query standardized scores",
    )
    parser.add_argument(
        "--twins",
        choices=("on", "off"),
        default="off",
        help="compatibility flag; twin filtering is disabled by decision D-024",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0:
        print("ERROR: --batch-size must be positive", file=sys.stderr)
        return 2
    if args.twins == "on":
        print("WARNING: --twins=on is ignored; twin filtering was cancelled by D-024", file=sys.stderr)
    try:
        run_pipeline(
            args.images,
            args.query,
            args.gallery,
            args.out,
            args.model,
            args.device,
            args.batch_size,
            args.candidate_threshold,
            verifier_path=args.verifier,
            rerank_weight=args.rerank_weight,
            rerank_mode=args.rerank_mode,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote and validated submission.csv, embeddings.npy, candidates.csv in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
