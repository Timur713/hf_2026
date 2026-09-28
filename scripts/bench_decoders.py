#!/usr/bin/env python3
"""Benchmark the disk-read -> JPEG-decode -> BBox-crop part of inference.

The benchmark deliberately stops before resize/normalization/model inference.  It
uses rows from an input CSV, keeps their order (or applies a seeded shuffle), and
cycles through them when more samples are requested than the CSV contains.

Optional backends are imported lazily.  An unavailable backend is represented in
the report instead of making the other measurements unusable.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
DEFAULT_METHODS = ("pil_full", "pil_draft", "torchvision_cuda", "dali_roi")


class BackendUnavailable(RuntimeError):
    """Raised when an optional decoder cannot run in the current environment."""


@dataclass(frozen=True)
class Sample:
    image_id: str
    path: Path
    bbox: Tuple[int, int, int, int]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a value in (0, 1]")
    return parsed


def _resolve_image(images_dir: Path, image_id: str) -> Path:
    direct = images_dir / image_id
    candidates = [direct]
    if not direct.suffix:
        candidates.extend(images_dir / (image_id + suffix) for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("image for {!r} not found under {}".format(image_id, images_dir))


def load_samples(csv_path: Path, images_dir: Path, limit: Optional[int] = None) -> List[Sample]:
    """Load benchmark rows without inspecting labels or other test-only data."""
    required = ("image_id", "x", "y", "w", "h")
    samples: List[Sample] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("CSV is missing columns: {}".format(", ".join(missing)))
        for line_number, row in enumerate(reader, start=2):
            try:
                image_id = row["image_id"].strip()
                bbox = tuple(int(row[name]) for name in ("x", "y", "w", "h"))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("invalid row {} in {}: {}".format(line_number, csv_path, exc)) from exc
            if not image_id:
                raise ValueError("empty image_id at row {}".format(line_number))
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError("non-positive BBox at row {}".format(line_number))
            samples.append(Sample(image_id, _resolve_image(images_dir, image_id), bbox))
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError("CSV contains no data rows: {}".format(csv_path))
    return samples


def clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    """Return a non-empty (left, top, right, bottom) crop inside the image."""
    if width <= 0 or height <= 0:
        raise ValueError("invalid image size {}x{}".format(width, height))
    x, y, w, h = bbox
    left = max(0, min(width - 1, x))
    top = max(0, min(height - 1, y))
    right = max(left + 1, min(width, x + w))
    bottom = max(top + 1, min(height, y + h))
    return left, top, right, bottom


class DecoderBackend:
    name = "base"
    device = "unknown"

    def decode_batch(self, samples: Sequence[Sample]) -> Any:
        raise NotImplementedError

    def synchronize(self) -> None:
        return None

    def close(self) -> None:
        return None


class PillowBackend(DecoderBackend):
    device = "cpu"

    def __init__(self, draft_scale: Optional[float]) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise BackendUnavailable("Pillow is not installed") from exc
        self._Image = Image
        self._draft_scale = draft_scale
        self.name = "pil_draft" if draft_scale is not None else "pil_full"

    def decode_batch(self, samples: Sequence[Sample]) -> List[Any]:
        crops = []
        for sample in samples:
            # read_bytes is intentionally inside the measured call: disk I/O is
            # part of the organizer's latency zone.
            encoded = sample.path.read_bytes()
            import io

            with self._Image.open(io.BytesIO(encoded)) as image:
                original_width, original_height = image.size
                if self._draft_scale is not None:
                    target = (
                        max(1, round(original_width * self._draft_scale)),
                        max(1, round(original_height * self._draft_scale)),
                    )
                    image.draft("RGB", target)
                decoded = image.convert("RGB")
                decoded.load()
                width, height = decoded.size
                left, top, right, bottom = clamp_bbox(sample.bbox, original_width, original_height)
                if (width, height) != (original_width, original_height):
                    scale_x = width / original_width
                    scale_y = height / original_height
                    left = max(0, min(width - 1, int(left * scale_x)))
                    top = max(0, min(height - 1, int(top * scale_y)))
                    right = max(left + 1, min(width, int(round(right * scale_x))))
                    bottom = max(top + 1, min(height, int(round(bottom * scale_y))))
                crop = decoded.crop((left, top, right, bottom))
                crop.load()
                crops.append(crop)
        return crops


class TorchvisionCudaBackend(DecoderBackend):
    name = "torchvision_cuda"

    def __init__(self, device_id: int) -> None:
        try:
            import torch
            import torchvision
            from torchvision.io import ImageReadMode, decode_jpeg, read_file
        except (ImportError, OSError, RuntimeError) as exc:
            raise BackendUnavailable("PyTorch/torchvision with image I/O is unavailable: {}".format(exc)) from exc
        if not torch.cuda.is_available():
            raise BackendUnavailable("torch.cuda.is_available() is false")
        if device_id < 0 or device_id >= torch.cuda.device_count():
            raise BackendUnavailable("CUDA device {} does not exist".format(device_id))
        torch.cuda.set_device(device_id)
        self._torch = torch
        self._torchvision = torchvision
        self._read_file = read_file
        self._decode_jpeg = decode_jpeg
        self._rgb = ImageReadMode.RGB
        self._device = torch.device("cuda", device_id)
        self.device = "cuda:{} ({})".format(device_id, torch.cuda.get_device_name(device_id))

    def decode_batch(self, samples: Sequence[Sample]) -> List[Any]:
        encoded = [self._read_file(str(sample.path)) for sample in samples]
        decoder_input: Any = encoded if len(encoded) > 1 else encoded[0]
        decoded = self._decode_jpeg(decoder_input, mode=self._rgb, device=self._device)
        images = decoded if isinstance(decoded, list) else [decoded]
        crops = []
        for sample, image in zip(samples, images):
            height, width = int(image.shape[-2]), int(image.shape[-1])
            left, top, right, bottom = clamp_bbox(sample.bbox, width, height)
            crops.append(image[..., top:bottom, left:right])
        return crops

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self._device)


class DaliRoiBackend(DecoderBackend):
    name = "dali_roi"

    def __init__(self, device_id: int, batch_size: int, num_threads: int) -> None:
        try:
            import numpy as np
            import torch
            from nvidia.dali import fn, types
            from nvidia.dali.pipeline import Pipeline
        except (ImportError, OSError, RuntimeError) as exc:
            raise BackendUnavailable("NVIDIA DALI (plus NumPy/PyTorch) is unavailable: {}".format(exc)) from exc
        if not torch.cuda.is_available():
            raise BackendUnavailable("torch.cuda.is_available() is false")
        if device_id < 0 or device_id >= torch.cuda.device_count():
            raise BackendUnavailable("CUDA device {} does not exist".format(device_id))
        torch.cuda.set_device(device_id)
        # feed_input supplies exactly one batch before each measured run. Disable
        # pipelined prefetching so DALI does not request a second, not-yet-fed
        # batch while scheduling the first one.
        pipeline = Pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=2026,
            exec_pipelined=False,
            exec_async=False,
            prefetch_queue_depth=1,
        )
        with pipeline:
            # DALI 2.x cannot infer metadata by peeking a named external source
            # before the first feed_input call. Supplying it explicitly also
            # keeps the pipeline build independent of the first sample.
            encoded = fn.external_source(
                name="encoded", device="cpu", batch=True, dtype=types.UINT8, ndim=1
            )
            anchors = fn.external_source(
                name="anchors", device="cpu", batch=True, dtype=types.FLOAT, ndim=1
            )
            shapes = fn.external_source(
                name="shapes", device="cpu", batch=True, dtype=types.FLOAT, ndim=1
            )
            decoded = fn.decoders.image_slice(
                encoded,
                anchors,
                shapes,
                device="mixed",
                output_type=types.RGB,
                normalized_anchor=False,
                normalized_shape=False,
                axis_names="HW",
            )
            pipeline.set_outputs(decoded)
        try:
            pipeline.build()
        except Exception as exc:
            raise BackendUnavailable("DALI ROI pipeline cannot be built: {}".format(exc)) from exc
        self._np = np
        self._torch = torch
        self._pipeline = pipeline
        self._device = torch.device("cuda", device_id)
        self.device = "cuda:{} ({})".format(device_id, torch.cuda.get_device_name(device_id))

    def decode_batch(self, samples: Sequence[Sample]) -> Any:
        encoded_batch = []
        anchors = []
        shapes = []
        for sample in samples:
            encoded_batch.append(self._np.frombuffer(sample.path.read_bytes(), dtype=self._np.uint8))
            x, y, width, height = sample.bbox
            # DALI image_slice uses H,W order when axis_names="HW".  Dataset
            # boxes are valid; bounds-hardening belongs to B4.2.
            anchors.append(self._np.asarray([max(0, y), max(0, x)], dtype=self._np.float32))
            shapes.append(self._np.asarray([max(1, height), max(1, width)], dtype=self._np.float32))
        self._pipeline.feed_input("encoded", encoded_batch)
        self._pipeline.feed_input("anchors", anchors)
        self._pipeline.feed_input("shapes", shapes)
        return self._pipeline.run()[0]

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self._device)


def make_backend(
    method: str,
    *,
    batch_size: int,
    device_id: int,
    draft_scale: float,
    dali_threads: int,
) -> DecoderBackend:
    if method == "pil_full":
        return PillowBackend(None)
    if method == "pil_draft":
        return PillowBackend(draft_scale)
    if method == "torchvision_cuda":
        return TorchvisionCudaBackend(device_id)
    if method == "dali_roi":
        return DaliRoiBackend(device_id, batch_size, dali_threads)
    raise ValueError("unknown method: {}".format(method))


def _cyclic_batch(samples: Sequence[Sample], start: int, batch_size: int) -> List[Sample]:
    return [samples[(start + offset) % len(samples)] for offset in range(batch_size)]


def benchmark_backend(
    backend: DecoderBackend,
    samples: Sequence[Sample],
    *,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> Dict[str, Any]:
    """Measure one backend; all CUDA work is synchronized at timing boundaries."""
    for iteration in range(warmup):
        batch = _cyclic_batch(samples, iteration * batch_size, batch_size)
        backend.decode_batch(batch)
        backend.synchronize()

    elapsed_seconds: List[float] = []
    for iteration in range(iterations):
        batch = _cyclic_batch(samples, (warmup + iteration) * batch_size, batch_size)
        backend.synchronize()
        started = time.perf_counter_ns()
        backend.decode_batch(batch)
        backend.synchronize()
        elapsed_seconds.append((time.perf_counter_ns() - started) / 1_000_000_000.0)

    total_seconds = sum(elapsed_seconds)
    batch_ms = [value * 1000.0 for value in elapsed_seconds]
    return {
        "method": backend.name,
        "status": "ok",
        "device": backend.device,
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations": iterations,
        "samples_measured": batch_size * iterations,
        "batch_latency_p50_ms": statistics.median(batch_ms),
        "per_image_latency_p50_ms": statistics.median(batch_ms) / batch_size,
        "throughput_fps": (batch_size * iterations) / total_seconds,
        "total_measured_seconds": total_seconds,
    }


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "Benchmark label: `{}`".format(report["run_label"]),
        "",
        "Scope: `{}`".format(report["scope"]),
        "",
        "| method | status | device | batch | p50 batch, ms | p50/image, ms | throughput, FPS | note |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result in report["results"]:
        if result["status"] == "ok":
            row = (
                result["method"],
                "ok",
                result["device"],
                result["batch_size"],
                "{:.3f}".format(result["batch_latency_p50_ms"]),
                "{:.3f}".format(result["per_image_latency_p50_ms"]),
                "{:.2f}".format(result["throughput_fps"]),
                "",
            )
        else:
            row = (
                result["method"],
                result["status"],
                result.get("device", "-"),
                result["batch_size"],
                "-",
                "-",
                "-",
                result.get("reason", "").replace("|", "\\|"),
            )
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(*row))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV; use train.csv for development benchmarks")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing <image_id>.jpg")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated decoder methods")
    parser.add_argument("--batch-sizes", default="1", help="Comma-separated positive batch sizes, e.g. 1,32")
    parser.add_argument("--warmup", type=_positive_int, default=50)
    parser.add_argument("--iterations", type=_positive_int, default=300)
    parser.add_argument("--limit", type=_positive_int, default=None, help="Optional number of CSV rows to load")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shuffle", action="store_true", help="Seeded shuffle before cycling through rows")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--draft-scale", type=_unit_float, default=0.5)
    parser.add_argument("--dali-threads", type=_positive_int, default=4)
    parser.add_argument("--run-label", default="unspecified", help="Hardware/run label stored in output")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--json-out", type=Path, default=None, help="Also write the complete JSON report here")
    parser.add_argument("--fail-on-unavailable", action="store_true")
    return parser


def _parse_csv_list(raw: str, valid: Optional[Iterable[str]] = None) -> List[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("list must not be empty")
    if len(values) != len(set(values)):
        raise ValueError("list contains duplicate values")
    if valid is not None:
        invalid = sorted(set(values) - set(valid))
        if invalid:
            raise ValueError("unsupported values: {}".format(", ".join(invalid)))
    return values


def run(argv: Optional[Sequence[str]] = None) -> Tuple[int, Dict[str, Any]]:
    args = build_parser().parse_args(argv)
    try:
        methods = _parse_csv_list(args.methods, DEFAULT_METHODS)
        batch_sizes = [int(value) for value in _parse_csv_list(args.batch_sizes)]
        if any(value <= 0 for value in batch_sizes):
            raise ValueError("batch sizes must be positive")
        samples = load_samples(args.csv, args.images, args.limit)
    except (OSError, ValueError) as exc:
        raise SystemExit("input error: {}".format(exc))

    if args.shuffle:
        random.Random(args.seed).shuffle(samples)

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "jpeg_decoder_crop",
        "scope": "disk_read+jpeg_decode+bbox_crop",
        "run_label": args.run_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "input": {
            "csv": str(args.csv),
            "images": str(args.images),
            "rows_loaded": len(samples),
            "shuffle": args.shuffle,
            "seed": args.seed,
        },
        "protocol": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "batch_sizes": batch_sizes,
            "cuda_synchronize": True,
            "page_cache": "os_managed_warm_after_warmup",
            "draft_scale": args.draft_scale,
        },
        "results": [],
        "notes": [
            "Timing only; decoder pixel changes must be checked on fold 0 before adoption (C3).",
            "Only A10 measurements may be copied into the Day-1 decoder table for B1.5.",
        ],
    }

    saw_error = False
    saw_unavailable = False
    for batch_size in batch_sizes:
        for method in methods:
            backend: Optional[DecoderBackend] = None
            try:
                backend = make_backend(
                    method,
                    batch_size=batch_size,
                    device_id=args.device_id,
                    draft_scale=args.draft_scale,
                    dali_threads=args.dali_threads,
                )
                report["results"].append(
                    benchmark_backend(
                        backend,
                        samples,
                        batch_size=batch_size,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                )
            except BackendUnavailable as exc:
                saw_unavailable = True
                report["results"].append(
                    {
                        "method": method,
                        "status": "unavailable",
                        "batch_size": batch_size,
                        "reason": str(exc),
                    }
                )
            except Exception as exc:
                saw_error = True
                report["results"].append(
                    {
                        "method": method,
                        "status": "error",
                        "batch_size": batch_size,
                        "reason": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
            finally:
                if backend is not None:
                    backend.close()

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_markdown(report))

    if saw_error or (args.fail_on_unavailable and saw_unavailable):
        return 1, report
    if not any(result["status"] == "ok" for result in report["results"]):
        return 2, report
    return 0, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    status, _ = run(argv)
    return status


if __name__ == "__main__":
    sys.exit(main())
