"""Exact projection-precast parity on train fold 0, never a quality evaluation.

All-data checkpoints are allowed: both paths use identical weights and inputs;
no metric, calibration, label fitting or test CSV is involved. CUDA is required
to certify the actual experimental AMP path. Any numeric difference fails.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from reid.infer.extractor import Extractor
from reid.infer.fusion import load_fusion_config
from scripts.audit_release import file_record
from scripts.bench_extract import _driver_version, _git_commit, _positive_int
from scripts.diagnose_c3_gap import train_rows


def uniform_indices(length: int, count: int = 32) -> list[int]:
    if length < count or count < 1:
        raise ValueError("need at least 32 distinct train-fold inputs for the batch=1 sample")
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


def compare_embeddings(baseline: np.ndarray, precast: np.ndarray) -> dict:
    if (baseline.ndim != 2 or baseline.shape != precast.shape or min(baseline.shape) < 1 or
            baseline.dtype != np.float32 or precast.dtype != np.float32):
        raise ValueError("both outputs must be matching nonempty float32 embedding matrices")
    if not np.isfinite(baseline).all() or not np.isfinite(precast).all():
        raise ValueError("nonfinite embeddings cannot pass parity")
    a, b = baseline.astype(np.float64), precast.astype(np.float64)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if np.any(na <= 0) or np.any(nb <= 0):
        raise ValueError("zero-norm embeddings cannot pass parity")
    cosine = np.clip(np.einsum("ij,ij->i", a, b) / (na * nb), -1, 1)
    return {
        "array_equal": bool(np.array_equal(baseline, precast)),
        "different_rows": int(np.count_nonzero(np.any(baseline != precast, axis=1))),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "max_norm_diff": float(np.max(np.abs(na - nb))),
        "baseline_max_unit_norm_error": float(np.max(np.abs(na - 1))),
        "precast_max_unit_norm_error": float(np.max(np.abs(nb - 1))),
        "min_cosine": float(np.min(cosine)),
    }


def model_sources(model: Path) -> dict:
    cfg = load_fusion_config(model)
    paths = [model / "fusion_config.json"]
    for member in cfg["components"]:
        paths.extend(model / member["path"] / name for name in ("model_config.yaml", "weights.safetensors"))
    if (model / "manifest.json").is_file():
        paths.append(model / "manifest.json")
    return {path.relative_to(model).as_posix(): file_record(path) for path in paths}


def compare_stage(baseline, precast, rows, batch: int, out: Path, name: str, save_arrays: bool) -> dict:
    batches, arrays_a, arrays_b = [], [], []
    digest_a, digest_b = hashlib.sha256(), hashlib.sha256()
    start = time.monotonic()
    for offset in range(0, len(rows), batch):
        part = rows[offset:offset + batch]
        items = [(row.image_path, row.bbox) for row in part]
        # Identical image paths, BBoxes, order and batch shape for each pair.
        a = baseline.extract_batch(items)
        b = precast.extract_batch(items)
        if a.shape != (len(part), int(baseline.cfg["embed_dim"])) or b.shape != a.shape:
            raise ValueError("extractor output shape disagrees with the input batch/C3 dimension")
        comparison = compare_embeddings(a, b)
        batches.append({"offset": offset, "rows": len(part), **comparison})
        digest_a.update(a.tobytes(order="C"))
        digest_b.update(b.tobytes(order="C"))
        if save_arrays:
            arrays_a.append(a.copy())
            arrays_b.append(b.copy())
        if len(batches) % 10 == 0 or offset + batch >= len(rows):
            print(f"{name}: {min(offset + batch, len(rows))}/{len(rows)} rows compared", flush=True)
    result = {
        "batch_size": batch, "rows": len(rows), "embedding_dim": int(baseline.cfg["embed_dim"]),
        "image_ids": [row.image_id for row in rows], "seconds": time.monotonic() - start,
        "array_equal": all(record["array_equal"] for record in batches),
        "different_rows": sum(record["different_rows"] for record in batches),
        "baseline_array_bytes_sha256": digest_a.hexdigest(), "precast_array_bytes_sha256": digest_b.hexdigest(),
        "batches": batches,
    }
    for key in ("max_abs_diff", "max_norm_diff", "baseline_max_unit_norm_error", "precast_max_unit_norm_error"):
        result[key] = max(record[key] for record in batches)
    result["min_cosine"] = min(record["min_cosine"] for record in batches)
    if save_arrays:
        path = out / f"{name}.npz"
        np.savez_compressed(path, image_ids=np.array(result["image_ids"]),
                            baseline=np.concatenate(arrays_a), precast=np.concatenate(arrays_b))
        result["arrays"] = {"file": path.name, **file_record(path)}
    return result


def run(model: Path, data_root: Path, out: Path, *, splits: Path = Path("splits"),
        device: str = "cuda", batch: int = 32, torch_threads: int = 8,
        workers: int = 8, save_arrays: bool = False) -> dict:
    if out.exists() or out.is_symlink():
        raise FileExistsError("refusing to overwrite parity output")
    if min(batch, workers, torch_threads) <= 0:
        raise ValueError("batch, workers and torch_threads must be positive")
    cuda = torch.device(device)
    if cuda.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("real CUDA AMP is required; CPU cannot certify projection precast")
    # set_device requires an index; torch.device('cuda') alone has index=None.
    # Preserve an already-selected GPU instead of silently forcing cuda:0.
    if cuda.index is None:
        cuda = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(torch_threads)
    torch.cuda.set_device(cuda)
    query, gallery = train_rows(data_root, splits, 0)
    rows = query + gallery
    sample_indices = uniform_indices(len(rows))
    sources = model_sources(model)
    inputs = {str(path): file_record(path) for path in (
        data_root / "train.csv", splits / "folds.csv", splits / "fold0_query.csv", splits / "fold0_gallery.csv")}
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "train_fold0_projection_precast_numeric_parity", "fold": 0,
        "scope": "Exact numeric parity only, not mAP, calibration, speed or training-provenance acceptance; all-data weights allowed",
        "commit": _git_commit(), "script": file_record(Path(__file__)), "model": str(model),
        "device": {"requested": device, "resolved": str(cuda), "name": torch.cuda.get_device_name(cuda),
                   "driver": _driver_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda},
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "workers": workers, "n_query": len(query), "n_gallery": len(gallery),
        "model_sources": sources, "input_sources": inputs, "sample_indices": sample_indices,
        "backend": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "stages": {}, "passed": False, "failures": [],
    }
    extractors = []
    try:
        for enabled in (False, True):
            extractors.append(Extractor(model, device, preprocess_workers=workers, precast_projections=enabled))
        baseline, precast = extractors
        report["baseline_cast_stats"] = dict(baseline.precast_projection_stats)
        report["precast_cast_stats"] = dict(precast.precast_projection_stats)
        if not report["precast_cast_stats"]["applied"]:
            raise ValueError("no projection parameters were precast; comparison cannot certify the experiment")
        report["stages"]["full_fold0"] = compare_stage(
            baseline, precast, rows, batch, out, "full_fold0", save_arrays)
        report["stages"]["batch1_sample"] = compare_stage(
            baseline, precast, [rows[i] for i in sample_indices], 1, out, "batch1_sample", save_arrays)
        for name, result in report["stages"].items():
            if not result["array_equal"]:
                report["failures"].append(f"{name}: embeddings differ")
        report["model_sources_unchanged"] = model_sources(model) == sources
        report["input_sources_unchanged"] = all(file_record(Path(path)) == record for path, record in inputs.items())
        if not report["model_sources_unchanged"] or not report["input_sources_unchanged"]:
            report["failures"].append("source files changed during comparison")
        report["passed"] = not report["failures"]
    except Exception as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for extractor in extractors:
            if extractor._preprocess_pool is not None:
                extractor._preprocess_pool.shutdown()
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        extractors.clear()
        gc.collect()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=_positive_int, default=32)
    parser.add_argument("--torch-threads", type=_positive_int, default=8)
    parser.add_argument("--workers", type=_positive_int, default=8)
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(**vars(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"PRECAST PARITY FAILED: {exc}\n")
    print(json.dumps({"passed": report["passed"], "failures": report["failures"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
