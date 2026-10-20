#!/usr/bin/env python3
"""Pinned D082 embedding-zone timing only; completion is separate from A5000 target."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

APP = Path(__file__).resolve().parents[1]
BAKED = APP / "artifacts/models/d082"
RUNTIME_REVISION = "hf_2026-d082-singleton-r3"
EXPECTED_SOURCE = {
    "Dockerfile": "0fdc13f14a0d3f6f940cd670c70eda5458049d5f3f56e7c645361b26dc2d068d",
    "docker-compose.yml": "5b3598a5106d71c96713d3598a2f0a0b1f81f7722d98d24900de4c4ece7a2be4",
    "reid/__init__.py": "c11117afad82d01197d80a72ec82de27e9bd2deffe32d31cae6c63ecc75edef6",
    "reid/data/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "reid/data/augment.py": "15dd10f780273db981c390204457e2922349f8d842e248b68912db4779de53f0",
    "reid/data/crop_cache.py": "c5f5c89ee8cf3442d63124f85c7b26460cf699c92fd05e06e22c106ad65296b2",
    "reid/data/dataset.py": "28f89cdc8946809d26a1c3698580316548bb29db7d9c10c8671dc663e8cde8fd",
    "reid/data/external.py": "4558bfefc61d2084296cd2a286c78aab4b48b03389d8b9562bc765ab6987c463",
    "reid/data/preprocess.py": "2e33c981ed842b76d73ed5f0916519ed96277a78502cdbdae14796d5a85c8777",
    "reid/data/sampler.py": "6ed644454edf044cb2ac15af4b742da8bf502d208842e0620ce2c359f61c823d",
    "reid/data/vehicleid.py": "5a12159477d2c407538ee9915ed1856770bf85283f992604c85d7adb32b227c6",
    "reid/eval/__init__.py": "21606e5f75d5840be65fbae8b6aad4d42c132a4766602402ed3c513b0938e384",
    "reid/eval/eval_local.py": "444454de2bfcdd46c63de74a580ef1f2bf353e1d13a1b48ea6cfa352f4d226be",
    "reid/eval/validate_outputs.py": "7582dde1bd6dce1c9b3390603cdc3dad4111394bd8e565c120ab5415185ad690",
    "reid/infer/__init__.py": "e7952ff33c6253be8c5869fefb70aa9a63a2eb9a837b3506e07c1a28f4f1ca0d",
    "reid/infer/d082.py": "b3a79ede6f319538fedb5862e1fb67e662bb0b609abe4a329d7566bc9ac797bc",
    "reid/infer/d082_entrypoint.py": "7ea3473d451d5c0b033aede11a0120ae27733806f852916551112d5d74b6b9e2",
    "reid/infer/d082_extractor.py": "2b1e009ed8e1b237f93ac59744aa552713efb90c3506a362a5c296d498e58dc7",
    "reid/infer/d082_production.py": "578c93f18e37ffb0308e271df0b82bd8e6f35bae208b54c92f56bf056dd82b98",
    "reid/infer/extractor.py": "3b2d7a39a798d303a72ad861a75f56b5a32f9d4b81627c24d6853e23c3c086c3",
    "reid/infer/fusion.py": "c7fa9c7813d43898012d31536391c4af20c54f5b50a80482235cd6276f746ad3",
    "reid/infer/list_ranker.py": "a5fe9fd109a4301961a0d831c352222667ba0a37355445b01bf9c931bbd69f26",
    "reid/infer/observability.py": "e5bbc3c64c0227c88feee9f2d4b834ab8b870abc283fabbba100981389088767",
    "reid/infer/plate_mask.py": "503bfe70843814f5124abb19a530c00330e9a55f719cabfcd85ccf0243b8b791",
    "reid/infer/run.py": "4b18ae91a7b30ed3547795f9f80fc29df7da191d394549d05351a14b7c8e6204",
    "reid/infer/smoke.py": "19fa3f7f3607122340744c7dc8a81b9b53e192353a2121c0200c7bdd4e9fc1fe",
    "reid/infer/verifier.py": "5ae5c868b5c104e60ae7761dc81e90a911cdf15161e0d8b0deac27a3910e4665",
    "reid/infer/verifier_cache.py": "ba01f3c1c892795e71baa2a5c4cf7fa405ff59b29d9a3b14f5dc8a0606ce4b39",
    "reid/model/__init__.py": "c041a1e671cfb4214d73a8da528198303a50d4fd4195b15e64ee807678482848",
    "reid/model/corr_verifier.py": "15b12211ccfbb24ba75549bd42a0b97f4e3e6533b509a964e4b7feb04c4a9a11",
    "reid/model/interpret.py": "3961b1e0a3e623f77a1b39e14aec577db84ddb3a1e7ee58427ccc6dca4ced11c",
    "reid/model/list_ranker.py": "e1d4eea8c01c30767fe3f3a87157bdad3705f69bb332e6d6b968eb441286eb3b",
    "reid/model/losses.py": "244498f3926939047c91f9e7a512e112215c6879076e950c85d579b62cf6269c",
    "reid/model/reranker.py": "12671954251681b092f470f048732811659bfc7bfa0a1c4589819f6290a4b423",
    "reid/model/verifier.py": "0e6c87a0d4c0ce5d40dd14be700869e77c27383624d53819ad981892b87951fd",
    "reid/post/__init__.py": "d06e08aa1e3ab6a2ed77d7c03afeea3231f184285bf41d1acb32274c33f6b624",
    "reid/post/gallery_policy.py": "c42cb69610db235895e09bede4d194fd0a23726b23d595d5e2c779735f6babce",
    "reid/post/pipeline.py": "0f5c9dbcdb8775c1775616ed49b7a54b75f414408bec6fba5cbd030f689d6380",
    "reid/post/policy.py": "60e7f50fedc2f210fb045f0e1234adca6eb66d249da3ddaed4545cc7f165cc47",
    "reid/post/ranker.py": "1312b86a501f7c28e8ea34907e4d061dce96c9059e700872c206d787c7576a07",
    "reid/post/search.py": "2a4859605b5c3ce7b00a93005218b095a6e095b75c255d47c7e946eba2eecc6d",
    "reid/post/verifier.py": "3217988fe811236a989d8f7b1c84f6aa7f62db69bb95e6cdc2b4a84a2e8b6100",
    "requirements-lock-cu121.txt": "3a23bae7b7e0297b79896a455b70c7ee2dd970bf0f98477a772301b8973ce288",
    "requirements.txt": "60a6fde9fe47e049b7cd90fbbafe8c9cc494d2450b87be23e2f7c371d46a91d9",
    "scripts/bench_extract.py": "addd99fcc7b80f018d400cad7ceab37ba4a906670dbadfa7718d29667b30a20e",
    "scripts/bench_pipeline.py": "cf5b4c693c16b3268f37f6fc7ec0d64b6a24a9bfbdc31691dbb870618095ca42",
}
BATCH_SIZES = (1, 8, 16, 32)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_binding(app=APP):
    actual = {}
    for relative, expected in EXPECTED_SOURCE.items():
        path = app / relative
        if path.is_symlink() or sha(path) != expected:
            raise ValueError(
                "runtime source differs from pinned integration: " + relative
            )
        actual[relative] = expected
    return dict(
        declared_revision=RUNTIME_REVISION,
        verified_files_sha256=actual,
        scope="Actual listed reid sources, original bench and package-lock bytes; not Git history/image signature verification",
    )


def baked_validation(root, loader):
    pin_file = root / "manifest.sha256"
    if pin_file.is_symlink() or not pin_file.is_file():
        raise ValueError("actual baked manifest.sha256 must be a regular file")
    pin = pin_file.read_text().strip()
    loaded = loader(root / "bundle", pin)
    if loaded.identity["recipe"]["candidate_threshold"] != 6.6:
        raise ValueError("not the frozen D082 refusal recipe")
    return pin, loaded


def target_gate(report):
    latency_report = report["latency_batch1"]
    if latency_report["warmup"] < 50 or latency_report["iterations"] < 300:
        raise ValueError("latency requires at least 50 warmups and 300 measurements")
    latency = float(latency_report["p50_ms"])
    measured = report["throughput"]
    if len(measured) != len(BATCH_SIZES) or [r["batch_size"] for r in measured] != list(
        BATCH_SIZES
    ):
        raise ValueError("unexpected throughput batch plan")
    if not math.isfinite(latency) or latency < 0:
        raise ValueError("invalid measured latency")
    if any(not math.isfinite(float(r["fps"])) or r["fps"] <= 0 for r in measured):
        raise ValueError("invalid measured throughput")
    if any(r["seconds"] < 10 for r in measured):
        raise ValueError("throughput requires at least 10 seconds per batch size")
    fastest = max(measured, key=lambda r: r["fps"])
    name = report["device"]["name"]
    actual_a5000 = name in ("NVIDIA RTX A5000", "RTX A5000")
    latency_ok, fps_ok = latency <= 40.0, fastest["fps"] >= 100.0
    return dict(
        hardware_required="RTX A5000",
        measured_device_name=name,
        actual_a5000=actual_a5000,
        latency_statistic="p50_ms at external batch1",
        latency_limit_ms=40.0,
        latency_pass=latency_ok,
        throughput_statistic="maximum measured FPS over fixed external batches1,8,16,32",
        throughput_min_fps=100.0,
        fastest_batch=fastest["batch_size"],
        measured_max_fps=fastest["fps"],
        throughput_pass=fps_ok,
        measured_numeric_targets_pass=latency_ok and fps_ok,
        actual_a5000_target_pass=actual_a5000 and latency_ok and fps_ok,
        scope="Embedding-zone engineering target only; not full pipeline, official benchmark or final acceptance",
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, default=Path("/data/images"))
    p.add_argument("--csv", type=Path, default=Path("/data/test_query.csv"))
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--throughput-seconds", type=float, default=10.0)
    p.add_argument("--run-label", required=True)
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--preprocess-workers", type=int, default=8)
    p.add_argument("--json-out", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.json_out.exists():
        raise FileExistsError("refuse existing final benchmark report")
    source = source_binding()
    wrapper_sha256 = sha(Path(__file__))
    sys.path.insert(0, str(APP))
    import torch
    from scripts import bench_extract as original
    from reid.infer.d082_production import load_production_runtime, inspect_bundle
    from reid.infer.d082_extractor import D082Extractor

    # Always validate actual baked bytes on CPU, before allocating neural CUDA models.
    started = time.perf_counter()
    pin, loaded = baked_validation(BAKED, load_production_runtime)
    cpu_identity = loaded.identity
    cpu_validation_seconds = time.perf_counter() - started
    del loaded
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "one visible CUDA GPU is required; CPU benchmark is forbidden"
        )

    arguments = [
        "--model",
        str(BAKED / "bundle"),
        "--device",
        "cuda:0",
        "--require-cuda",
        "--images",
        str(args.images),
        "--csv",
        str(args.csv),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--throughput-seconds",
        str(args.throughput_seconds),
        "--batch-sizes",
        ",".join(map(str, BATCH_SIZES)),
        "--run-label",
        args.run_label,
    ]
    if args.torch_threads is not None:
        arguments += ["--torch-threads", str(args.torch_threads)]
    if args.preprocess_workers is not None:
        arguments += ["--preprocess-workers", str(args.preprocess_workers)]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    previous = original.Extractor
    try:
        original.Extractor = D082Extractor
        with tempfile.TemporaryDirectory(
            prefix="fixed4-bench-", dir=args.json_out.parent
        ) as tmp:
            raw = Path(tmp) / "original_bench.json"
            result = original.main([*arguments, "--json-out", str(raw)])
            if result != 0:
                return result
            report = json.loads(raw.read_text())
    finally:
        original.Extractor = previous

    # No completion receipt is published if the baked bundle/source changed.
    if (BAKED / "manifest.sha256").read_text().strip() != pin:
        raise ValueError("baked pin changed during benchmark")
    inspect_bundle(BAKED / "bundle", pin)
    if source_binding() != source:
        raise ValueError("runtime source changed during benchmark")
    if sha(Path(__file__)) != wrapper_sha256:
        raise ValueError("benchmark wrapper changed during execution")
    report.update(
        kind="final_d082_embedding_benchmark",
        status="BENCHMARK_COMPLETED_NOT_FINAL_ACCEPTANCE",
        benchmark_completed=True,
        runtime_source=source,
        wrapper_sha256=wrapper_sha256,
        manifest_sha256=pin,
        baked_root=str(BAKED),
        production_identity=cpu_identity,
        cpu_validation_seconds=cpu_validation_seconds,
        input_csv_sha256=sha(args.csv),
        fusion_internal_batch_size=4,
        padding="repeat last prepared tensor to4; decode real JPEG once",
        embedding_arithmetic="Original extract_component_rows: Torch L2 + two NumPy L2; original PreparedRankerGallery._fuse with fixed bundle scales",
        measured_zone="disk JPEG read/decode -> bbox crop -> per-component preprocessing -> fixed4 encoder forward -> production L2/fusion -> CPU float32 embedding",
        excludes=[
            "CPU validation",
            "model loading",
            "gallery search/statistics",
            "graph",
            "pair verifier",
            "confidence",
            "CSV/output writing",
        ],
        verifier_gallery_cache=False,
        projection_precast=False,
        inputs_scope="Original bench cycles the supplied CSV rows; warm OS page cache after warmup, no explicit JPEG tensor/embedding cache or filesystem cache flush",
        final_acceptance=False,
    )
    from importlib.metadata import version

    report["installed_package_versions"] = {
        line.split("==", 1)[0]: version(line.split("==", 1)[0])
        for line in (APP / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    properties = torch.cuda.get_device_properties(0)
    report["device"].update(
        visible_device_count=1,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        total_memory_bytes=properties.total_memory,
        compute_capability=[properties.major, properties.minor],
        uuid=str(getattr(properties, "uuid", "unavailable")),
    )
    report["a5000_target"] = target_gate(report)
    with args.json_out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                status=report["status"],
                manifest_sha256=pin,
                a5000_target=report["a5000_target"],
                final_acceptance=False,
            )
        )
    )
    # A target miss is a completed measurement. Consumers inspect the separate gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
