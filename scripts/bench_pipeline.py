"""B-F6: warmed JPEG-to-candidate batch-1 profile with a fixed prepared gallery.

Train inputs only; no quality evaluation or tuning. Model load and gallery prep
are outside request latency. No verifier-token caching/approximation is enabled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import numpy as np
import torch

from reid.infer.extractor import Extractor
from reid.infer.run import CANDIDATES_PER_QUERY, extract_rows, read_inputs
from reid.infer.verifier import PairVerifier
from reid.post.pipeline import PreparedGallery
from reid.post.policy import candidate_threshold_for
from scripts.audit_release import file_record
from scripts.audit_sprint_runtime import audit


def summary_ms(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("timing samples must be finite, nonnegative and nonempty")
    return {
        "n": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(values.mean()),
        "samples_ms": values.tolist(),
    }


class TimedVerifier:
    def __init__(self, verifier):
        self.verifier = verifier
        self.elapsed_ms = 0.0

    def score_pairs(self, query, candidates):
        start = perf_counter()
        result = self.verifier.score_pairs(query, candidates)
        # The production adapter returns CPU numpy logits, synchronizing GPU.
        self.elapsed_ms += (perf_counter() - start) * 1000
        return result


def profile_requests(
    extractor,
    prepared,
    verifier,
    rows,
    threshold,
    warmup,
    iterations,
    sync,
    component_mode=False,
):
    if not rows or warmup < 0 or iterations < 1:
        raise ValueError("nonempty rows, warmup >=0 and iterations >=1 required")
    scorer = TimedVerifier(verifier) if verifier is not None else None
    samples = {
        key: []
        for key in (
            "extraction",
            "post_total",
            "verifier",
            "search_and_refusal",
            "full",
        )
    }
    for index in range(warmup + iterations):
        row = rows[index % len(rows)]
        if scorer is not None:
            scorer.elapsed_ms = 0
        sync()
        start = perf_counter()
        if component_mode:
            from reid.infer.d082 import extract_component_rows

            embedding = extract_component_rows(extractor, [row], 1)
        else:
            embedding = extract_rows(extractor, [row], 1)
        sync()
        extracted = perf_counter()
        result = prepared.rank(embedding, [(row.image_path, row.bbox)], scorer)
        # The request ends with the candidate/abstention decision; no request
        # batching, CSV/disk serialization, transport or queue waiting is timed.
        accepted = bool(result.top_scores[0] >= threshold)
        candidate_ids = result.ranking[0][:CANDIDATES_PER_QUERY] if accepted else []
        assert len(candidate_ids) <= CANDIDATES_PER_QUERY
        sync()
        end = perf_counter()
        if index >= warmup:
            post = (end - extracted) * 1000
            verifier_ms = scorer.elapsed_ms if scorer is not None else 0.0
            for key, value in {
                "extraction": (extracted - start) * 1000,
                "post_total": post,
                "verifier": verifier_ms,
                "search_and_refusal": max(0.0, post - verifier_ms),
                "full": (end - start) * 1000,
            }.items():
                samples[key].append(value)
    return {key: summary_ms(value) for key, value in samples.items()}


def profile_batch32(extractor, rows, iterations, sync, component_mode=False):
    items = [
        (rows[i % len(rows)].image_path, rows[i % len(rows)].bbox) for i in range(32)
    ]

    def extract():
        if component_mode:
            from reid.infer.d082 import extract_component_rows

            return extract_component_rows(
                extractor, [rows[i % len(rows)] for i in range(32)], 32
            )
        return extractor.extract_batch(items)

    extract()  # separate throughput warmup
    sync()
    samples = []
    for _ in range(iterations):
        sync()
        start = perf_counter()
        extract()
        sync()
        samples.append((perf_counter() - start) * 1000)
    result = summary_ms(samples)
    result["fps"] = 32 * 1000 / result["mean_ms"]
    result["scope"] = (
        "D082 raw components + export normalization"
        if component_mode
        else "Extractor.extract_batch"
    ) + " only; no search, verifier, refusal or queue wait"
    return result


def load_profile_runtime(
    model, verifier, rows, galleries, development_pin=None, production_pin=None
):
    """Select a pinned graph runtime; never profile a graph bundle as fusion."""
    if development_pin is not None and production_pin is not None:
        raise ValueError("choose either a development or a production D082 pin")
    if production_pin is not None:
        from reid.infer.d082_production import load_production_runtime

        return load_production_runtime(model, production_pin, verifier)
    if development_pin is not None:
        from reid.infer.d082 import load_d082_runtime

        return load_d082_runtime(
            model,
            development_pin,
            [r.image_id for r in rows],
            [r.image_id for r in galleries],
            verifier,
        )
    if any(
        (model / name).exists() or (model / name).is_symlink()
        for name in (
            "d082_production.json",
            "ranker_runtime.json",
            "head",
            "development_ids.json",
        )
    ):
        raise ValueError("graph profile requires an explicit D082 manifest pin")
    return None


def run(a):
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.device.startswith("cuda"):
        device = torch.device(a.device)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats(device)

        def sync():
            torch.cuda.synchronize(device)
    else:
        device = torch.device("cpu")

        def sync():
            return None

    qcsv = Path(f"splits/fold{a.fold}_query.csv")
    gcsv = Path(f"splits/fold{a.fold}_gallery.csv")
    qrows, grows = read_inputs(qcsv, a.images), read_inputs(gcsv, a.images)
    if a.query_limit:
        qrows = qrows[: a.query_limit]
    if a.gallery_limit:
        grows = grows[: a.gallery_limit]
    if len(grows) < 10:
        raise ValueError("at least 10 gallery images required")
    validation_start = perf_counter()
    d082 = load_profile_runtime(
        a.model,
        a.verifier,
        qrows,
        grows,
        getattr(a, "d082_manifest_sha256", None),
        getattr(a, "d082_production_manifest_sha256", None),
    )
    validated = perf_counter()
    start = perf_counter()
    if getattr(a, "d082_production_manifest_sha256", None) is not None:
        from reid.infer.d082_extractor import D082Extractor

        extractor = D082Extractor(a.model, str(device))
    else:
        extractor = (
            Extractor(a.model, str(device))
            if d082 is None
            else Extractor(a.model, str(device), fusion_internal_batch_size=4)
        )

    verifier = PairVerifier(
        a.verifier if d082 is None else d082.verifier_path, str(device)
    )
    sync()
    loaded = perf_counter()
    if d082 is None:
        gallery_emb = extract_rows(extractor, grows, a.gallery_batch_size)
    else:
        from reid.infer.d082 import extract_component_rows

        gallery_components = extract_component_rows(
            extractor, grows, a.gallery_batch_size
        )
    sync()
    extracted = perf_counter()
    prepared = (
        PreparedGallery(
            gallery_emb,
            [r.image_id for r in grows],
            [(r.image_path, r.bbox) for r in grows],
            extractor.cfg,
        )
        if d082 is None
        else d082.prepare(gallery_components, grows)
    )
    ready = perf_counter()
    threshold = candidate_threshold_for(extractor.cfg)
    request = profile_requests(
        extractor,
        prepared,
        verifier,
        qrows,
        threshold,
        a.warmup,
        a.iterations,
        sync,
        component_mode=d082 is not None,
    )
    batch32 = profile_batch32(
        extractor, qrows, a.batch32_iterations, sync, component_mode=d082 is not None
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "kind": "full_request_profile_not_quality",
        "argv": sys.argv,
        "model": audit(a.model, a.verifier) if d082 is None else d082.identity,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "workers_env": {
            k: os.environ.get(k)
            for k in ("REID_PREPROCESS_WORKERS", "REID_VERIFIER_WORKERS")
        },
        "fold": a.fold,
        "input_sha": {"query_csv": file_record(qcsv), "gallery_csv": file_record(gcsv)},
        "query_ids": [r.image_id for r in qrows],
        "gallery_ids": [r.image_id for r in grows],
        "warmup_requests": a.warmup,
        "warmup_batch32": 1,
        "startup_ms": {
            "runtime_validation": (validated - validation_start) * 1000,
            "model_load": (loaded - start) * 1000,
            "gallery_extraction": (extracted - loaded) * 1000,
            "gallery_statistics": (ready - extracted) * 1000,
        },
        "batch1": request,
        "batch32_extraction": batch32,
        "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
        "limitations": [
            "CPU/H100 is not A5000 acceptance",
            "No quality metrics; all-data weights allowed for engineering only",
            "Model-load is in-process cold; OS page cache is NOT flushed",
            "Warm p95 is empirical over reported sample count, not an SLA",
            "Full request excludes file serialization, network and queue time",
            "Verifier timing combines decode/preprocess and network; no token cache enabled",
            "Batch32 may repeat selected train frames; inspect sample counts and IDs",
            "RSS and VRAM peaks cover entire benchmark, including batch32",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--verifier", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fold", type=int, choices=(0, 1, 3, 4), default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--batch32-iterations", type=int, default=10)
    p.add_argument("--gallery-batch-size", type=int, default=32)
    p.add_argument("--query-limit", type=int)
    p.add_argument("--gallery-limit", type=int)
    p.add_argument(
        "--d082-manifest-sha256",
        help="opt-in D082 OOF development profile; not release acceptance",
    )
    p.add_argument(
        "--d082-production-manifest-sha256",
        help="all-data D082 profile with its pinned graph/verifier and fixed internal4",
    )
    a = p.parse_args()
    if (
        a.warmup < 0
        or min(a.iterations, a.batch32_iterations, a.gallery_batch_size) < 1
        or (a.query_limit is not None and a.query_limit < 1)
        or (a.gallery_limit is not None and a.gallery_limit < 10)
    ):
        p.error("invalid limits, iterations, warmup or batch size")
    result = run(a)
    print(
        json.dumps(
            {
                "device": result["device"],
                "full_p50_ms": result["batch1"]["full"]["p50_ms"],
                "full_p95_ms": result["batch1"]["full"]["p95_ms"],
                "report": str(a.out),
            }
        )
    )


if __name__ == "__main__":
    main()
