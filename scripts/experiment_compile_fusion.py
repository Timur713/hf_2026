"""EXPERIMENT ONLY: CUDA Inductor/default on fusion members' actual features().

Train fold0 inputs only; all-data weights allow numerical/speed probing, NOT mAP.
Full C5 stays unchanged. No checkpoint, preprocessing, threshold or default edits.
Public callable API: https://github.com/pytorch/pytorch/blob/v2.5.1/torch/__init__.py
Supervisor kills only its own worker process group after at most 870 + 5 seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from reid.infer.extractor import Extractor
from reid.infer.fusion import load_fusion_config
from scripts.audit_release import file_record
from scripts.bench_extract import _driver_version, _git_commit, _positive_int, measure_latency, measure_throughput
from scripts.check_precast_equivalence import compare_embeddings, model_sources, uniform_indices
from scripts.diagnose_c3_gap import train_rows

COMPILE_OPTIONS = dict(backend="inductor", mode="default", fullgraph=True, dynamic=True)
BATCHES = (1, 8, 16, 32)


def compile_features(extractor):
    """forward() is NOT called by Extractor; compile only the bound features API."""
    members = extractor._fusion_components
    if not members or extractor.device.type != "cuda":
        raise ValueError("this experiment requires a CUDA fusion extractor")
    if any(model.training or any(p.requires_grad for p in model.parameters())
           for model, _, _, _ in members):
        raise ValueError("compile probe requires frozen eval models")
    records = []
    for model, cfg, dtype, _ in members:
        model.features = torch.compile(model.features, **COMPILE_OPTIONS)
        records.append({"class": type(model).__name__, "input_size": cfg.get("input_size"),
                        "amp_dtype": str(dtype), "callable": "features"})
    return records


def sample_embeddings(extractor, rows, batch):
    parts = []
    for offset in range(0, len(rows), batch):
        items = [(r.image_path, r.bbox) for r in rows[offset:offset + batch]]
        parts.append(extractor.extract_batch(items))
    result = np.concatenate(parts)
    if result.shape != (len(rows), int(extractor.cfg["embed_dim"])):
        raise ValueError("incorrect embedding shape")
    return result


def parity_record(reference, compiled):
    record = compare_embeddings(reference, compiled)
    record["compiled_max_unit_norm_error"] = record.pop("precast_max_unit_norm_error")
    record["eager_array_sha256"] = hashlib.sha256(reference.tobytes()).hexdigest()
    record["compiled_array_sha256"] = hashlib.sha256(compiled.tobytes()).hexdigest()
    return record


def save_report(out, report):
    pending = out / "report.pending.json"
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pending.replace(out / "report.json")
    print(f"compile probe: {report.get('phase', report['status'])}", flush=True)


def benchmark(extractor, rows, cuda, progress):
    torch.cuda.reset_peak_memory_stats(cuda)
    sync = lambda: torch.cuda.synchronize(cuda)
    progress("latency batch1 (50 warmup, 300 timed)")
    result = {"latency_batch1": measure_latency(extractor, rows, 50, 300, sync), "throughput": []}
    for batch in BATCHES:
        progress(f"throughput batch{batch} (>=10 seconds)")
        result["throughput"].append(measure_throughput(extractor, rows, batch, 10., sync))
    result["peak_vram_mb"] = torch.cuda.max_memory_allocated(cuda) / 1024**2
    return result


def worker(args):
    out, model = args.out, args.model
    report = {"kind": "experimental_torch_compile_fusion", "status": "running", "phase": "validation",
              "compile_options": COMPILE_OPTIONS, "commit": _git_commit(), "script": file_record(Path(__file__)),
              "scope": "train fold0 numerical/speed probe only; no mAP, verifier, precast or production promotion",
              "protocol": {"latency_warmup": 50, "latency_iterations": 300, "throughput_min_seconds": 10,
                           "batch_sizes": BATCHES, "zone": "complete C5 including disk/JPEG/C3"},
              "model": str(model), "runtime_limit_seconds": args.max_seconds,
              "production_acceptance": False, "equivalence_scope": "32 fixed train images at each batch size only"}
    extractor = None
    started = time.monotonic()
    try:
        save_report(out, report)
        load_fusion_config(model)
        sources = model_sources(model)
        report["model_sources"] = sources
        query, gallery = train_rows(args.data_root, args.splits, 0)
        chosen = [ (query + gallery)[i] for i in uniform_indices(len(query + gallery)) ]
        report["sample_image_ids"] = [r.image_id for r in chosen]
        report["input_sources"] = {str(p): file_record(p) for p in
            (args.data_root / "train.csv", args.splits / "folds.csv", args.splits / "fold0_query.csv", args.splits / "fold0_gallery.csv")}
        cuda = torch.device(args.device)
        if cuda.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("compile experiment requires real CUDA")
        if cuda.index is None:
            cuda = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(cuda)
        torch.set_num_threads(args.torch_threads)
        report["device"] = {"requested": args.device, "resolved": str(cuda), "name": torch.cuda.get_device_name(cuda),
                            "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "driver": _driver_version()}
        report.update(preprocess_workers=args.workers, cpu_count=os.cpu_count(),
                      torch_num_threads=torch.get_num_threads(), torch_num_interop_threads=torch.get_num_interop_threads())

        def progress(phase):
            report["phase"] = phase
            report["elapsed_seconds"] = time.monotonic() - started
            save_report(out, report)

        progress("load eager fusion")
        load_start = time.monotonic()
        extractor = Extractor(model, str(cuda), preprocess_workers=args.workers, precast_projections=False)
        torch.cuda.synchronize(cuda)
        report["model_load_seconds"] = time.monotonic() - load_start
        report["precast_projections"] = extractor.precast_projection_stats
        reference = {}
        for batch in BATCHES:
            progress(f"eager sample batch{batch}")
            reference[batch] = sample_embeddings(extractor, chosen, batch)
        report["eager"] = benchmark(extractor, query, cuda, lambda s: progress("eager " + s))
        progress("wrap features with torch.compile")
        report["compiled_members"] = compile_features(extractor)
        report["parity"] = {}
        for batch in BATCHES:
            progress(f"compile/probe sample batch{batch}")
            begin = time.monotonic()
            compiled = sample_embeddings(extractor, chosen, batch)
            torch.cuda.synchronize(cuda)
            report["parity"][str(batch)] = {**parity_record(reference[batch], compiled),
                                           "first_probe_seconds_including_compile": time.monotonic() - begin}
        report["sample_exact_equal"] = all(p["array_equal"] for p in report["parity"].values())
        # Numerical drift does not suppress useful timing, but MUST NOT imply acceptance.
        report["compiled"] = benchmark(extractor, query, cuda, lambda s: progress("compiled " + s))
        report["model_sources_unchanged"] = model_sources(model) == sources
        if not report["model_sources_unchanged"]:
            raise ValueError("model source hashes changed during probe")
        report["status"] = "completed_exact_sample" if report["sample_exact_equal"] else "completed_with_drift"
        report["phase"] = "finished; not production acceptance"
        return 0 if report["sample_exact_equal"] else 2
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        if extractor is not None and extractor._preprocess_pool is not None:
            extractor._preprocess_pool.shutdown()
        report["elapsed_seconds"] = time.monotonic() - started
        save_report(out, report)


def stop_group(process):
    """Only the isolated child session, including its compiler subprocesses."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def supervise(args):
    out = args.out
    if out.exists() or out.is_symlink():
        raise FileExistsError("refusing to overwrite experimental output")
    if any(out.resolve().is_relative_to(p.resolve()) for p in (args.model, args.data_root)):
        raise ValueError("output must be outside model/data directories")
    if not 1 <= args.max_seconds <= 870:
        raise ValueError("max-seconds must be in 1..870 (hard maximum below 15 minutes)")
    if min(args.workers, args.torch_threads) < 1:
        raise ValueError("thread counts must be positive")
    if torch.device(args.device).type != "cuda":
        raise ValueError("compile experiment is CUDA-only")
    out.mkdir(parents=True)
    save_report(out, {"status": "starting", "compile_options": COMPILE_OPTIONS, "production_acceptance": False})
    env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(out.resolve() / "inductor_cache"),
               TRITON_CACHE_DIR=str(out.resolve() / "triton_cache"), TORCH_LOGS="recompiles,graph_breaks")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker"]
    for key, value in vars(args).items():
        if key != "_worker":
            cmd += ["--" + key.replace("_", "-"), str(value)]
    with (out / "worker.log").open("x") as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.max_seconds)
        except subprocess.TimeoutExpired:
            stop_group(process)
            report = json.loads((out / "report.json").read_text())
            report.update(status="timeout", error="worker process group stopped at runtime limit", production_acceptance=False)
            save_report(out, report)
            return 124
        except BaseException:
            stop_group(process)
            raise
    if code not in (0, 2) and json.loads((out / "report.json").read_text())["status"] in ("starting", "running"):
        report = json.loads((out / "report.json").read_text())
        report.update(status="failed", error=f"worker exited {code}; see worker.log")
        save_report(out, report)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=_positive_int, default=8)
    parser.add_argument("--torch-threads", type=_positive_int, default=8)
    parser.add_argument("--max-seconds", type=_positive_int, default=840)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        return worker(args) if args._worker else supervise(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"COMPILE EXPERIMENT FAILED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
