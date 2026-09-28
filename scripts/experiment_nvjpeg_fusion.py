"""EXPERIMENT ONLY: batched nvJPEG -> CPU full frames -> unchanged C3/fusion.

Train fold0 only. Bench includes disk reads, decoding, CPU copies and full C5;
quality requires OOF metadata and uses production ranking without verifier.
API source: https://docs.pytorch.org/vision/0.20/generated/torchvision.io.decode_jpeg.html
No production default, shared preprocessing or checkpoint is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.io import ImageReadMode, decode_jpeg, read_file

from reid.eval.eval_local import evaluate, read_candidates, read_submission
from reid.eval.validate_outputs import validate_outputs
from reid.infer.extractor import Extractor
from reid.infer.fusion import load_fusion_config
from reid.infer.run import extract_rows, rank_embeddings, write_outputs
from reid.model import load_config
from reid.post.policy import candidate_threshold_for
from scripts.audit_release import file_record
from scripts.bench_extract import _driver_version, _git_commit, _positive_int, _weights_mb, measure_latency, measure_throughput
from scripts.check_precast_equivalence import compare_embeddings, model_sources, uniform_indices
from scripts.diagnose_c3_gap import train_rows


class NvJpegFusionExtractor(Extractor):
    """Call-local decoded frames only; no image survives an extract_batch call."""
    def __init__(self, model_dir, device="cuda:0", preprocess_workers=8, precast_projections=False):
        if precast_projections:
            raise ValueError("this isolated decoder experiment requires projection precast off")
        if torch.device(device).type != "cuda":
            raise ValueError("nvJPEG experiment requires CUDA")
        super().__init__(model_dir, device, preprocess_workers=preprocess_workers, precast_projections=False)
        self._nvjpeg_frames = None
        self._call_lock = threading.Lock()

    def decoded_frames(self, items):
        def encoded(item):
            path = Path(item[0])
            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                raise ValueError("train-only nvJPEG experiment accepts JPEG inputs only")
            data = read_file(str(path))  # Fresh CPU bytes on EVERY measured call.
            if data.numel() < 2 or data[:2].tolist() != [255, 216]:
                raise ValueError(f"not a JPEG: {path}")
            return data
        encoded_images = (list(self._preprocess_pool.map(encoded, items)) if self._preprocess_pool
                          else [encoded(item) for item in items])
        frames = decode_jpeg(encoded_images, mode=ImageReadMode.RGB, device=self.device,
                             apply_exif_orientation=False)
        if not isinstance(frames, list) or len(frames) != len(items):
            raise ValueError("batched nvJPEG returned an unexpected number of frames")
        cpu_frames = [frame.cpu() for frame in frames]  # Blocking D2H, inside C5.
        if any(f.dtype != torch.uint8 or f.ndim != 3 or f.shape[0] != 3 or min(f.shape[-2:]) < 1
               for f in cpu_frames):
            raise ValueError("nvJPEG must produce RGB uint8 CHW frames")
        return cpu_frames

    def _decode_frame(self, item):
        if self._nvjpeg_frames is None:
            raise RuntimeError("decoded-frame lookup is available only inside the current batch")
        return self._nvjpeg_frames[str(item[0])]

    def extract_batch(self, items):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("concurrent calls on one experimental extractor are unsupported")
        try:
            if items:
                self._nvjpeg_frames = dict(zip((str(item[0]) for item in items), self.decoded_frames(items)))
            return super().extract_batch(items)
        finally:
            self._nvjpeg_frames = None
            self._call_lock.release()


def require_oof(model: Path, cfg: dict) -> None:
    for member in cfg["components"]:
        if load_config(str(model / member["path"])).get("train", {}).get("folds") != "all except 0":
            raise ValueError("quality requires every component to declare train.folds='all except 0'; all-data forbidden")


def pixel_drift(baseline, nvjpeg, rows):
    chosen = [rows[i] for i in uniform_indices(len(rows))]
    items = [(row.image_path, row.bbox) for row in chosen]
    decoded = nvjpeg.decoded_frames(items)
    records = []
    for row, item, other in zip(chosen, items, decoded):
        reference = baseline._decode_frame(item)
        if reference.shape != other.shape:
            raise ValueError("CPU/nvJPEG pixel shapes differ")
        delta = (reference.to(torch.int16) - other.to(torch.int16)).abs()
        records.append({"image_id": row.image_id, "max_abs": int(delta.max()),
                        "mean_abs": float(delta.float().mean()), "changed_fraction": float((delta != 0).float().mean())})
    return records


def quality_run(model, cfg, query, gallery, splits, out, device, workers):
    require_oof(model, cfg)
    models, arrays, metrics = [], [], {}
    try:
        for name, cls in (("baseline", Extractor), ("nvjpeg", NvJpegFusionExtractor)):
            extractor = cls(model, device, preprocess_workers=workers, precast_projections=False)
            models.append(extractor)
            q, g = extract_rows(extractor, query, 32), extract_rows(extractor, gallery, 32)
            arrays.append(np.concatenate([q, g]))
            ranked = rank_embeddings(query, gallery, q, g, cfg, verifier=None)
            dest = out / name
            write_outputs(dest, query, gallery, q, g, candidate_threshold_for(cfg), ranked)
            validate_outputs(splits / "fold0_query.csv", splits / "fold0_gallery.csv",
                             dest / "submission.csv", dest / "embeddings.npy", dest / "candidates.csv")
            metrics[name] = evaluate(0, read_submission(str(dest / "submission.csv")),
                                     read_candidates(str(dest / "candidates.csv")), folds_path=str(splits / "folds.csv"))
            print(f"{name}: map10={metrics[name]['map10']:.8f}", flush=True)
        delta = metrics["nvjpeg"]["map10"] - metrics["baseline"]["map10"]
        return {"metrics": metrics, "map10_delta": delta, "within_c3_tolerance": delta >= -0.002,
                "embedding_drift": compare_embeddings(*arrays),
                "pixel_drift_sample32": pixel_drift(*models, query + gallery),
                "same_fixed_config_and_scales": cfg, "verifier_enabled": False,
                "training_provenance": "OOF metadata checked, not independently proven"}
    finally:
        for extractor in models:
            if extractor._preprocess_pool is not None:
                extractor._preprocess_pool.shutdown()


def run(model, data_root, out, mode="bench", device="cuda:0", workers=8, torch_threads=8, splits=Path("splits")):
    if out.exists() or out.is_symlink():
        raise FileExistsError("refusing to overwrite experimental output")
    if out.resolve().is_relative_to(data_root.resolve()) or out.resolve().is_relative_to(model.resolve()):
        raise ValueError("output must be outside data and model directories")
    if mode not in {"bench", "quality"} or min(workers, torch_threads) <= 0:
        raise ValueError("invalid mode or nonpositive thread count")
    cfg = load_fusion_config(model)
    if mode == "quality":
        require_oof(model, cfg)  # Reject all-data before allocating GPU resources.
    cuda = torch.device(device)
    if cuda.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("CUDA with nvJPEG is required")
    if cuda.index is None:
        cuda = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(cuda)
    torch.set_num_threads(torch_threads)
    query, gallery = train_rows(data_root, splits, 0)
    sources = model_sources(model)
    out.mkdir(parents=True)
    report = {"kind": "experimental_batched_nvjpeg_fusion", "mode": mode, "commit": _git_commit(),
              "script": file_record(Path(__file__)), "model_sources": sources, "status": "running",
              "scope": "Train fold0 only; unchanged C3, no verifier, no precast, no production promotion",
              "input_sources": {str(p): file_record(p) for p in (data_root / "train.csv", splits / "folds.csv",
                                splits / "fold0_query.csv", splits / "fold0_gallery.csv")},
              "device": {"requested": device, "resolved": str(cuda), "name": torch.cuda.get_device_name(cuda),
                         "driver": _driver_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                         "torchvision": torchvision.__version__}, "model": str(model), "run_label": "nvjpeg-fusion",
              "preprocess_workers": workers, "cpu_count": os.cpu_count(), "torch_num_threads": torch.get_num_threads(),
              "torch_threads_requested": torch_threads, "input_csv": str(splits / "fold0_query.csv"),
              "precast_projections": {"requested": False, "applied": False},
              "torch_num_interop_threads": torch.get_num_interop_threads(), "weights_mb": _weights_mb(model)}
    extractor = None
    try:
        if mode == "bench":
            start = time.perf_counter()
            extractor = NvJpegFusionExtractor(model, str(cuda), preprocess_workers=workers)
            torch.cuda.synchronize(cuda)
            report["model_load_seconds"] = time.perf_counter() - start
            report["precast_projections"] = extractor.precast_projection_stats
            torch.cuda.reset_peak_memory_stats(cuda)
            sync = lambda: torch.cuda.synchronize(cuda)
            report["latency_batch1"] = measure_latency(extractor, query, 50, 300, sync)
            report["throughput"] = [measure_throughput(extractor, query, n, 10.0, sync) for n in (1, 8, 16, 32)]
            report["peak_vram_mb"] = torch.cuda.max_memory_allocated(cuda) / 1024**2
        else:
            report["quality"] = quality_run(model, cfg, query, gallery, splits, out, str(cuda), workers)
        report["model_sources_unchanged"] = model_sources(model) == sources
        if not report["model_sources_unchanged"]:
            raise ValueError("model sources changed during experiment")
        report["status"] = "completed; not production acceptance"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if extractor is not None and extractor._preprocess_pool is not None:
            extractor._preprocess_pool.shutdown()
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("bench", "quality"), default="bench")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=_positive_int, default=8)
    parser.add_argument("--torch-threads", type=_positive_int, default=8)
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    try:
        report = run(**vars(parser.parse_args(argv)))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"NVJPEG EXPERIMENT FAILED: {exc}\n")
    print(json.dumps({"mode": report["mode"], "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
