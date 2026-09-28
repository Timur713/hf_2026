"""Prepare a self-contained D-052 artifact using fixed gallery-only scales.

No query input, labels, fitting objective or parameter search is accepted. This
prepares statistics for one specified static gallery; it does not certify transfer
to a different gallery, five-fold quality, GPU speed or release readiness.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml

from reid.infer.extractor import Extractor
from reid.infer.run import extract_rows, read_inputs
from reid.post.policy import rerank_weight_for
from scripts.audit_release import MAX_WEIGHT_BYTES, file_record
from scripts.install_fusion_bundle import VERIFIER_PROVENANCE, validate_verifier_provenance


def gallery_similarity_std(embeddings: np.ndarray) -> float:
    """Population std INCLUDING the diagonal, matching A's D-052 concat path.

    D-057's confidence std is a DIFFERENT statistic, excluding the diagonal.
    Blocks cap temporary matrix memory without introducing any query dependency.
    """
    g = np.asarray(embeddings, dtype=np.float32)
    if g.ndim != 2 or len(g) < 2 or g.shape[1] == 0 or not np.isfinite(g).all():
        raise ValueError("at least two finite gallery embeddings are required")
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("gallery contains a zero embedding")
    g = g / norms
    count, mean, m2 = 0, 0.0, 0.0
    for start in range(0, len(g), 256):
        block = (g[start:start + 256] @ g.T).astype(np.float64)
        n = block.size
        block_mean = float(block.mean())
        block_m2 = float(np.square(block - block_mean).sum())
        delta = block_mean - mean
        total = count + n
        m2 += block_m2 + delta * delta * count * n / total
        mean += delta * n / total
        count = total
    std = float(np.sqrt(m2 / count))
    if not np.isfinite(std) or std <= 1e-12:
        raise ValueError("degenerate gallery similarity scale")
    return std


def prepare_bundle(
    components: list[Path], images: Path, gallery: Path, out: Path,
    *, name: str = "fusion", device: str = "cpu", batch_size: int = 32,
    verifier: Path | None = None, require_all_data: bool = False,
    verifier_provenance: Path | None = None,
    rerank_weight: float = 0.2,
    candidate_threshold: float = 6.0,
    threshold_source: str | None = None,
) -> dict:
    rerank_weight = rerank_weight_for({}, rerank_weight)
    if (isinstance(candidate_threshold, bool) or not math.isfinite(candidate_threshold)
            or candidate_threshold <= 0):
        raise ValueError("candidate threshold must be finite and positive")
    if threshold_source is None:
        if candidate_threshold != 6.0:
            raise ValueError("a changed candidate threshold requires its OOF calibration source")
        threshold_source = "D-057, two-fold provisional calibration; not retuned here"
    if not threshold_source.strip():
        raise ValueError("threshold source must not be empty")
    if len(components) < 2 or batch_size <= 0 or not name.strip():
        raise ValueError("need at least two components, positive batch size and nonempty name")
    if verifier_provenance is not None and verifier is None:
        raise ValueError("verifier provenance requires a verifier checkpoint")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {out}")
    sources, configs = [], []
    total_weight_bytes = 0
    for index, directory in enumerate(components):
        if (directory / "fusion_config.json").exists():
            raise ValueError("nested fusion artifacts are not supported")
        cfg_path, weights_path = directory / "model_config.yaml", directory / "weights.safetensors"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"invalid model config: {cfg_path}")
        if require_all_data and cfg.get("train", {}).get("folds") != "all":
            raise ValueError(f"component {index} is not trained on all folds")
        total_weight_bytes += weights_path.stat().st_size
        sources.extend(((cfg_path, f"components/{index}/model_config.yaml"),
                        (weights_path, f"components/{index}/weights.safetensors")))
        configs.append(cfg)
    if verifier is not None:
        total_weight_bytes += verifier.stat().st_size
        if require_all_data:
            import torch
            payload = torch.load(verifier, map_location="cpu", weights_only=True)
            if payload.get("trained_on") != "folds [0, 1, 2, 3, 4]":
                raise ValueError("verifier is not trained on all folds")
        sources.append((verifier, "verifier.pt"))
        if verifier_provenance is not None:
            sources.append((verifier_provenance, VERIFIER_PROVENANCE))
    if total_weight_bytes > MAX_WEIGHT_BYTES:
        raise ValueError("combined inference weights exceed 2 GB")
    rows = read_inputs(gallery, images)
    if len(rows) < 10:
        raise ValueError("gallery needs at least ten images for the C6 top-10")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temp:
        staged = Path(temp) / "artifact"
        staged.mkdir()
        records = {}
        for source, relative in sources:
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            records[relative] = file_record(destination)
        if verifier_provenance is not None:
            validate_verifier_provenance(staged / VERIFIER_PROVENANCE, records["verifier.pt"]["sha256"])

        # Validate the bytes that will actually be published, not just the
        # source snapshot from preflight (an upstream training job may write).
        total_weight_bytes = sum(record["bytes"] for relative, record in records.items()
                                 if relative.endswith((".safetensors", ".pt")))
        if total_weight_bytes > MAX_WEIGHT_BYTES:
            raise ValueError("copied inference weights exceed 2 GB")
        configs = []
        for index in range(len(components)):
            cfg = yaml.safe_load((staged / f"components/{index}/model_config.yaml").read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                raise ValueError(f"invalid copied model config: component {index}")
            if require_all_data and cfg.get("train", {}).get("folds") != "all":
                raise ValueError(f"copied component {index} is not trained on all folds")
            configs.append(cfg)
        if require_all_data and verifier is not None:
            payload = torch.load(staged / "verifier.pt", map_location="cpu", weights_only=True)
            if payload.get("trained_on") != "folds [0, 1, 2, 3, 4]":
                raise ValueError("copied verifier is not trained on all folds")

        manifest = {
            "schema_version": 1, "name": name, "components": [],
            "candidate_policy": "gallery_z", "candidate_threshold": candidate_threshold,
            "rerank_mode": "standardized", "rerank_weight": rerank_weight,
        }
        for index in range(len(components)):
            # Extract from the COPIED bytes, so calibration and saved hashes
            # refer to the same snapshot even if an upstream artifact changes.
            extractor = Extractor(staged / f"components/{index}", device)
            try:
                emb = extract_rows(extractor, rows, batch_size)
                std = gallery_similarity_std(emb)
            finally:
                pool = getattr(extractor, "_preprocess_pool", None)
                if pool is not None:
                    pool.shutdown(wait=True)
            manifest["components"].append({"path": f"components/{index}", "similarity_std": std})
            del extractor, emb
            gc.collect()
        (staged / "fusion_config.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        records["fusion_config.json"] = file_record(staged / "fusion_config.json")
        audit = {
            "kind": "fusion_artifact_preparation", "artifacts": records,
            "total_weight_bytes": total_weight_bytes,
            "require_all_data": require_all_data,
            "component_training_folds": [cfg.get("train", {}).get("folds") for cfg in configs],
            "gallery_csv": file_record(gallery), "gallery_count": len(rows),
            "scale_definition": "std of normalized gallery-gallery cosine, ddof=0, including diagonal",
            "candidate_threshold_source": threshold_source,
            "scope": "gallery statistics and packaging only; no query, metric, GPU acceptance or final release claim",
        }
        (staged / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        if out.exists():
            raise FileExistsError(f"artifact appeared during preparation: {out}")
        staged.rename(out)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", type=Path, action="append", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="fusion")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verifier", type=Path)
    parser.add_argument("--verifier-provenance", type=Path,
                        help="optional YAML sidecar with artifact.sha256 matching --verifier")
    parser.add_argument("--require-all-data", action="store_true")
    parser.add_argument("--rerank-weight", type=float, default=0.2,
                        help="previously selected on train OOF; never tuned here")
    parser.add_argument("--candidate-threshold", type=float, default=6.0,
                        help="previously calibrated on train OOF; never tuned here")
    parser.add_argument("--threshold-source",
                        help="calibration decision/report; required when changing the legacy threshold")
    args = parser.parse_args()
    try:
        result = prepare_bundle(args.component, args.images, args.gallery, args.out,
                                name=args.name, device=args.device, batch_size=args.batch_size,
                                verifier=args.verifier, require_all_data=args.require_all_data,
                                verifier_provenance=args.verifier_provenance,
                                rerank_weight=args.rerank_weight,
                                candidate_threshold=args.candidate_threshold,
                                threshold_source=args.threshold_source)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"Prepared {args.out}: {len(args.component)} components, {result['total_weight_bytes']} weight bytes")


if __name__ == "__main__":
    main()
