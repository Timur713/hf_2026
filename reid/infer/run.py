"""End-to-end vehicle ReID inference pipeline (contract C5).

The command reads query/gallery CSV files, extracts normalized embeddings,
performs deterministic cosine search, and writes the three C6 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from reid.eval.validate_outputs import validate_outputs
from reid.infer.extractor import Extractor
from reid.infer.observability import LEVELS, artifact_identity, configure_logging, stage
from reid.post.policy import (
    DEFAULT_CANDIDATE_THRESHOLD,
    candidate_threshold_for,
    rerank_weight_for,
)
from reid.post.search import SearchResult, cosine_search
from reid.post.verifier import PairScorer
from reid.infer.verifier import PairVerifier


CSV_HEADER = ["image_id", "x", "y", "w", "h"]
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
# Keep the package logger even under ``python -m`` (where __name__ is __main__).
LOGGER = logging.getLogger("reid.infer.run")


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
            raise ValueError(
                f"{csv_path}: expected header {CSV_HEADER!r}, got {reader.fieldnames!r}"
            )
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
                raise ValueError(
                    f"{csv_path}:{line_number}: bbox must be finite with w,h > 0"
                )
            rows.append(InputRow(image_id, _image_path(images_dir, image_id), bbox))
    ids = [row.image_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{csv_path}: image_id values must be unique")
    return rows


def extract_rows(
    extractor: Extractor, rows: Sequence[InputRow], batch_size: int
) -> np.ndarray:
    chunks = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        LOGGER.debug(
            "extract_batch start=%d count=%d total=%d", start, len(batch), len(rows)
        )
        items = [(row.image_path, row.bbox) for row in batch]
        chunks.append(extractor.extract_batch(items))
    if not chunks:
        raise ValueError("input CSV has no data rows")
    result = np.concatenate(chunks).astype(np.float32, copy=False)
    if (
        result.ndim != 2
        or result.shape[0] != len(rows)
        or not np.isfinite(result).all()
    ):
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

    np.save(
        out_dir / "embeddings.npy",
        np.concatenate([query_emb, gallery_emb]).astype(np.float32),
    )

    with (out_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "gallery_id", "confidence"])
        for row, ranked_ids, confidence in zip(
            query_rows, result.ranking, result.top_scores
        ):
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
    d082_manifest_sha256: str | None = None,
    d082_plate_mask_control: bool = False,
    verifier_gallery_cache: bool = False,
    d082_production_manifest_sha256: str | None = None,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if verifier_gallery_cache and (
        d082_manifest_sha256 is None or verifier is not None
    ):
        raise ValueError(
            "experimental verifier cache requires pinned D082 and its own verifier"
        )
    if d082_production_manifest_sha256 is not None and (
        d082_manifest_sha256 is not None
        or verifier_gallery_cache
        or extractor is not None
        or verifier is not None
    ):
        raise ValueError(
            "production D082 forbids development pin, cache and injected models"
        )
    if (
        d082_production_manifest_sha256 is None
        and (model_dir / "d082_production.json").exists()
    ):
        raise ValueError(
            "production bundle requires --d082-production-manifest-sha256; no legacy fallback"
        )
    if (
        d082_manifest_sha256 is None
        and d082_production_manifest_sha256 is None
        and any(
            (model_dir / name).exists() for name in ("head", "development_ids.json")
        )
    ):
        raise ValueError(
            "graph artifact requires an explicit D082 manifest pin; no legacy fallback"
        )
    if d082_manifest_sha256 is None and (
        (model_dir / "ranker_runtime.json").exists()
        or (d082_plate_mask_control and d082_production_manifest_sha256 is None)
    ):
        raise ValueError(
            "D082 bundle/control requires --d082-manifest-sha256; graph omission is forbidden"
        )
    if d082_plate_mask_control and extractor is not None:
        raise ValueError("D082 mask control must construct its own masked extractor")
    with stage(
        LOGGER,
        "read_inputs",
        query=str(query_csv),
        gallery=str(gallery_csv),
        images=str(images_dir),
    ):
        query_rows = read_inputs(query_csv, images_dir)
        gallery_rows = read_inputs(gallery_csv, images_dir)
    LOGGER.info(
        "inputs query_count=%d gallery_count=%d", len(query_rows), len(gallery_rows)
    )
    d082 = None
    if d082_manifest_sha256 is not None or d082_production_manifest_sha256 is not None:
        if (
            candidate_threshold not in (None, 6.6)
            or rerank_weight not in (None, 0.15)
            or rerank_mode not in (None, "standardized")
            or verifier is not None
        ):
            raise ValueError(
                "D082 frozen recipe requires threshold6.6/weight.15/standardized and pinned verifier"
            )
        with stage(LOGGER, "load_d082_runtime"):
            if d082_production_manifest_sha256 is not None:
                from reid.infer.d082_production import load_production_runtime

                d082 = load_production_runtime(
                    model_dir, d082_production_manifest_sha256, verifier_path
                )
            else:
                from reid.infer.d082 import load_d082_runtime

                d082 = load_d082_runtime(
                    model_dir,
                    d082_manifest_sha256,
                    [r.image_id for r in query_rows],
                    [r.image_id for r in gallery_rows],
                    verifier_path,
                )
        verifier_path = d082.verifier_path
        LOGGER.info("d082_identity %s", json.dumps(d082.identity, sort_keys=True))
    with stage(LOGGER, "load_extractor", model=str(model_dir), device=device):
        if d082_plate_mask_control:
            from reid.infer.plate_mask import PlateMaskedExtractor

            active_extractor = PlateMaskedExtractor(
                model_dir, device, fusion_internal_batch_size=4
            )
        elif d082_production_manifest_sha256 is not None:
            from reid.infer.d082_extractor import D082Extractor

            active_extractor = D082Extractor(model_dir, device)
        else:
            active_extractor = extractor or (
                Extractor(model_dir, device)
                if d082 is None
                else Extractor(model_dir, device, fusion_internal_batch_size=4)
            )
    threshold = candidate_threshold_for(active_extractor.cfg, candidate_threshold)
    cfg = active_extractor.cfg
    effective = {
        "candidate_policy": cfg.get("candidate_policy", "cosine"),
        "candidate_threshold": threshold,
        "threshold_source": "CLI"
        if candidate_threshold is not None
        else "artifact"
        if "candidate_threshold" in cfg
        else "policy_default",
        "rerank_weight": rerank_weight_for(cfg, rerank_weight),
        "rerank_mode": rerank_mode or cfg.get("rerank_mode", "raw"),
        "verifier_enabled": verifier is not None or verifier_path is not None,
        "verifier_gallery_cache": verifier_gallery_cache,
        "device": device,
        "batch_size": batch_size,
        "extraction_runtime": type(active_extractor).__name__,
        "fusion_internal_batch_size": getattr(
            active_extractor, "fusion_internal_batch_size", None
        ),
    }
    LOGGER.info("effective_config %s", json.dumps(effective, sort_keys=True))
    if d082 is not None:
        LOGGER.info("d082_plate_mask_control enabled=%s", d082_plate_mask_control)
    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info(
            "artifact_identity %s",
            json.dumps(artifact_identity(model_dir, verifier_path), sort_keys=True),
        )
    with stage(LOGGER, "extract_query", query=str(query_csv)):
        if d082 is None:
            query_emb = extract_rows(active_extractor, query_rows, batch_size)
        else:
            from reid.infer.d082 import extract_component_rows

            query_components = extract_component_rows(
                active_extractor, query_rows, batch_size
            )
    with stage(LOGGER, "extract_gallery", gallery=str(gallery_csv)):
        if d082 is None:
            gallery_emb = extract_rows(active_extractor, gallery_rows, batch_size)
        else:
            gallery_components = extract_component_rows(
                active_extractor, gallery_rows, batch_size
            )
    active_verifier = verifier
    if active_verifier is None and verifier_path is not None:
        with stage(
            LOGGER, "load_verifier", checkpoint=str(verifier_path), device=device
        ):
            if d082_plate_mask_control:
                from reid.infer.plate_mask import PlateMaskedPairVerifier

                active_verifier = PlateMaskedPairVerifier(verifier_path, device)
            else:
                active_verifier = PairVerifier(verifier_path, device)
    if verifier_gallery_cache:
        from reid.infer.verifier_cache import StaticGalleryTokenVerifier

        with stage(LOGGER, "prepare_verifier_gallery_cache"):
            active_verifier = StaticGalleryTokenVerifier(
                active_verifier, [(r.image_path, r.bbox) for r in gallery_rows]
            )
        LOGGER.warning(
            "experimental_verifier_cache bytes=%d numeric_parity_not_assumed=true",
            active_verifier.cache_bytes,
        )
    with stage(
        LOGGER, "rank", query_count=len(query_rows), gallery_count=len(gallery_rows)
    ):
        if d082 is None:
            result = rank_embeddings(
                query_rows,
                gallery_rows,
                query_emb,
                gallery_emb,
                cfg,
                active_verifier,
                rerank_weight,
                rerank_mode,
            )
        else:
            with stage(LOGGER, "prepare_d082_gallery"):
                prepared = d082.prepare(gallery_components, gallery_rows)
            query_emb = prepared.fuse_queries(query_components)
            gallery_emb = prepared.gallery.embeddings
            with stage(LOGGER, "d082_graph_verifier_confidence"):
                result = prepared.rank(
                    query_components,
                    [(r.image_path, r.bbox) for r in query_rows],
                    active_verifier,
                )
    with stage(LOGGER, "write_outputs", out=str(out_dir)):
        write_outputs(
            out_dir, query_rows, gallery_rows, query_emb, gallery_emb, threshold, result
        )
    with stage(LOGGER, "validate_c6", out=str(out_dir)):
        validate_outputs(
            query_csv,
            gallery_csv,
            out_dir / "submission.csv",
            out_dir / "embeddings.npy",
            out_dir / "candidates.csv",
        )
    LOGGER.info(
        "pipeline_done queries=%d accepted=%d rejected=%d embedding_dim=%d out=%s",
        len(query_rows),
        np.count_nonzero(result.top_scores >= threshold),
        np.count_nonzero(result.top_scores < threshold),
        query_emb.shape[1],
        out_dir,
    )


def rank_embeddings(
    query_rows: Sequence[InputRow],
    gallery_rows: Sequence[InputRow],
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    cfg: dict,
    verifier: PairScorer | None = None,
    rerank_weight: float | None = None,
    rerank_mode: str | None = None,
) -> SearchResult:
    """One production postprocessing path, also used for audited C4 replay.

    Labels/camera IDs never enter this function. A replay may replace only the
    pair scorer with previously measured, exactly ID-aligned logits.
    """
    from reid.post.pipeline import PreparedGallery

    with stage(LOGGER, "prepare_gallery_statistics"):
        prepared = PreparedGallery(
            gallery_emb,
            [row.image_id for row in gallery_rows],
            [(row.image_path, row.bbox) for row in gallery_rows],
            cfg,
        )
    with stage(LOGGER, "search_verifier_confidence"):
        return prepared.rank(
            query_emb,
            [(row.image_path, row.bbox) for row in query_rows],
            verifier,
            rerank_weight,
            rerank_mode,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("MODEL_DIR", "artifacts/models/final")),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--d082-manifest-sha256",
        default=None,
        help="opt-in pinned OOF D082 development runtime; no test/control2 or release acceptance",
    )
    parser.add_argument(
        "--d082-production-manifest-sha256",
        default=None,
        help="explicit all-data D082 candidate pin; not a final acceptance claim",
    )
    parser.add_argument(
        "--d082-plate-mask-control",
        action="store_true",
        help="D082 diagnostic only: deterministic plate-area fill for both encoders and verifier",
    )
    parser.add_argument(
        "--verifier-gallery-cache",
        action="store_true",
        help="experimental D082 static-gallery token cache; off by default, requires parity acceptance",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LEVELS,
        default=os.environ.get("REID_LOG_LEVEL", "INFO"),
        help="stderr logging level; CLI > REID_LOG_LEVEL > INFO",
    )
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
    parser.add_argument(
        "--rerank-weight",
        type=float,
        default=None,
        help="artifact rerank_weight, or legacy 0.2 if unspecified",
    )
    parser.add_argument(
        "--rerank-mode",
        choices=("raw", "standardized"),
        default=None,
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
    if args.log_level not in LEVELS:
        _parser().error("REID_LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR")
    configure_logging(args.log_level)
    if args.batch_size <= 0:
        LOGGER.error("--batch-size must be positive")
        return 2
    if args.twins == "on":
        LOGGER.warning("--twins=on is ignored; twin filtering was cancelled by D-024")
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
            d082_manifest_sha256=args.d082_manifest_sha256,
            d082_plate_mask_control=args.d082_plate_mask_control,
            verifier_gallery_cache=args.verifier_gallery_cache,
            d082_production_manifest_sha256=args.d082_production_manifest_sha256,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error(
            "pipeline_failed query=%s gallery=%s model=%s out=%s error=%s: %s",
            args.query,
            args.gallery,
            args.model,
            args.out,
            type(exc).__name__,
            exc,
        )
        return 1
    LOGGER.info(
        "wrote and validated submission.csv, embeddings.npy, candidates.csv in %s",
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
