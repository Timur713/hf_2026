#!/usr/bin/env python3
"""Benchmark the complete C5 extraction zone on real CSV/image inputs.

The measured call includes disk read, JPEG decode, crop, preprocessing,
forward, postprocessing and L2 normalization. Model loading is reported
separately and never included in latency or throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

import torch

from reid.infer.extractor import Extractor
from reid.infer.run import InputRow, read_inputs


Sync = Callable[[], None]
Clock = Callable[[], float]


def batch_at(rows: Sequence[InputRow], start: int, batch_size: int):
    if not rows:
        raise ValueError("benchmark input is empty")
    return [
        (rows[(start + offset) % len(rows)].image_path, rows[(start + offset) % len(rows)].bbox)
        for offset in range(batch_size)
    ]


def measure_latency(
    extractor,
    rows: Sequence[InputRow],
    warmup: int,
    iterations: int,
    sync: Sync,
    clock: Clock = time.perf_counter,
) -> dict:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")
    for index in range(warmup):
        extractor.extract_batch(batch_at(rows, index, 1))
    sync()

    latencies_ms = []
    for index in range(iterations):
        sync()
        started = clock()
        extractor.extract_batch(batch_at(rows, warmup + index, 1))
        sync()
        latencies_ms.append((clock() - started) * 1000.0)
    ordered = sorted(latencies_ms)
    return {
        "warmup": warmup,
        "iterations": iterations,
        "p50_ms": statistics.median(ordered),
        "p90_ms": ordered[max(0, int(0.90 * len(ordered)) - 1)],
        "mean_ms": statistics.fmean(ordered),
    }


def measure_throughput(
    extractor,
    rows: Sequence[InputRow],
    batch_size: int,
    min_seconds: float,
    sync: Sync,
    clock: Clock = time.perf_counter,
) -> dict:
    if batch_size <= 0 or min_seconds <= 0:
        raise ValueError("batch_size and min_seconds must be positive")
    # One batch makes allocator/kernel startup effects happen before measurement.
    extractor.extract_batch(batch_at(rows, 0, batch_size))
    sync()

    batches = 0
    sync()
    started = clock()
    while True:
        extractor.extract_batch(batch_at(rows, batches * batch_size, batch_size))
        batches += 1
        sync()
        elapsed = clock() - started
        if elapsed >= min_seconds:
            break
    images = batches * batch_size
    return {
        "batch_size": batch_size,
        "seconds": elapsed,
        "batches": batches,
        "images": images,
        "fps": images / elapsed,
    }


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
    except OSError:
        # The production image intentionally contains no git binary. The
        # session wrapper records the full commit separately on the host, so
        # the benchmark must remain runnable in the minimal offline image.
        return None
    return result.stdout.strip() or None


def _weights_mb(model_dir: Path) -> float:
    return sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file()) / (1024**2)


def _markdown(report: dict) -> str:
    latency = report["latency_batch1"]
    lines = [
        f"Run `{report['run_label']}` — {report['device']['name']}",
        "",
        "| batch | duration, s | images | FPS |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["throughput"]:
        lines.append(
            f"| {row['batch_size']} | {row['seconds']:.3f} | {row['images']} | {row['fps']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Latency batch=1: p50 {latency['p50_ms']:.3f} ms, "
            f"p90 {latency['p90_ms']:.3f} ms ({latency['iterations']} measurements).",
            f"Preprocessing workers: {report['preprocess_workers']}.",
            f"CPU count: {report['cpu_count']}. PyTorch threads: {report['torch_num_threads']} "
            f"intra-op / {report['torch_num_interop_threads']} inter-op.",
            f"Experimental projection precast: {report['precast_projections']}.",
            f"Model load: {report['model_load_seconds']:.3f} s. "
            f"Peak VRAM: {report['peak_vram_mb']:.1f} MiB. Weights: {report['weights_mb']:.1f} MiB.",
        ]
    )
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--csv", type=Path, default=Path("dataset/test_query.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--batch-sizes", default="1,8,16,32")
    parser.add_argument("--throughput-seconds", type=float, default=10.0)
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--torch-threads", type=_positive_int, default=None,
        help="PyTorch intra-op CPU threads before model loading (default: leave unchanged)",
    )
    parser.add_argument(
        "--precast-projections", action="store_true",
        help="EXPERIMENTAL: precast CUDA AMP Linear/Conv2d weights only; needs quality acceptance",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        help="parallel JPEG/preprocessing workers (default: auto, or REID_PREPROCESS_WORKERS)",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    device = torch.device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("--require-cuda was set, but --device is not CUDA")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("--require-cuda was set, but CUDA is unavailable")
    if device.type == "cuda":
        # set_device rejects torch.device('cuda') without an index. Respect the
        # process's current CUDA device rather than silently forcing GPU zero.
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)

    rows = read_inputs(args.csv, args.images)
    load_started = time.perf_counter()
    extractor = Extractor(args.model, str(device), preprocess_workers=args.preprocess_workers,
                          precast_projections=args.precast_projections)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started

    if device.type == "cuda":
        sync = lambda: torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        device_name = torch.cuda.get_device_name(device)
    else:
        sync = lambda: None
        device_name = str(args.device)

    latency = measure_latency(extractor, rows, args.warmup, args.iterations, sync)
    throughput = [
        measure_throughput(extractor, rows, batch, args.throughput_seconds, sync)
        for batch in batch_sizes
    ]
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    report = {
        "run_label": args.run_label,
        "commit": _git_commit(),
        "model": str(args.model),
        "weights_mb": _weights_mb(args.model),
        "input_csv": str(args.csv),
        "model_load_seconds": model_load_seconds,
        "preprocess_workers": extractor.preprocess_workers,
        "cpu_count": os.cpu_count(),
        "torch_threads_requested": args.torch_threads,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "precast_projections": extractor.precast_projection_stats,
        "device": {
            "requested": args.device,
            "resolved": str(device),
            "name": device_name,
            "driver": _driver_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "latency_batch1": latency,
        "throughput": throughput,
        "peak_vram_mb": peak_vram_mb,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
