from pathlib import Path
import json

import numpy as np
import pytest

from reid.infer.run import InputRow
from scripts import bench_extract
from scripts.bench_extract import batch_at, measure_latency, measure_throughput


class FakeExtractor:
    def __init__(self):
        self.batch_sizes = []

    def extract_batch(self, items):
        self.batch_sizes.append(len(items))
        return np.ones((len(items), 4), dtype=np.float32)


class StepClock:
    def __init__(self, step):
        self.value = 0.0
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


def _rows(count=3):
    return [InputRow(f"i{i}", Path(f"i{i}.jpg"), (0, 0, 1, 1)) for i in range(count)]


def test_batch_at_wraps_in_input_order():
    batch = batch_at(_rows(), start=2, batch_size=4)
    assert [path.stem for path, _ in batch] == ["i2", "i0", "i1", "i2"]


def test_latency_uses_exact_warmup_and_iteration_counts():
    extractor = FakeExtractor()
    result = measure_latency(
        extractor,
        _rows(),
        warmup=2,
        iterations=3,
        sync=lambda: None,
        clock=StepClock(0.001),
    )
    assert extractor.batch_sizes == [1, 1, 1, 1, 1]
    assert result["iterations"] == 3
    assert abs(result["p50_ms"] - 1.0) < 1e-9


def test_throughput_runs_for_at_least_requested_duration():
    extractor = FakeExtractor()
    result = measure_throughput(
        extractor,
        _rows(),
        batch_size=8,
        min_seconds=0.025,
        sync=lambda: None,
        clock=StepClock(0.01),
    )
    assert result["seconds"] >= 0.025
    assert result["images"] == result["batches"] * 8
    assert all(size == 8 for size in extractor.batch_sizes)


def test_git_commit_is_optional_in_minimal_container(monkeypatch):
    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(bench_extract.subprocess, "run", missing_git)
    assert bench_extract._git_commit() is None


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_torch_threads_rejects_nonpositive_or_noninteger_values(value):
    with pytest.raises(SystemExit) as error:
        bench_extract._parser().parse_args(["--model", "unused", "--run-label", "test",
                                           "--torch-threads", value])
    assert error.value.code == 2


def test_torch_threads_default_is_unset():
    args = bench_extract._parser().parse_args(["--model", "unused", "--run-label", "test"])
    assert args.torch_threads is None


@pytest.mark.parametrize("requested", [None, 1, 2, 4])
def test_benchmark_sets_threads_before_loading_and_records_actual_counts(tmp_path, monkeypatch, requested):
    calls = []
    thread_count = [8]
    def set_threads(value):
        calls.append(("threads", value))
        thread_count[0] = value
    def read_inputs(*_args):
        calls.append(("read_inputs", thread_count[0]))
        return _rows()
    class FakeModel(FakeExtractor):
        def __init__(self, *_args, preprocess_workers=None, precast_projections=False):
            super().__init__()
            calls.append(("load", thread_count[0]))
            self.preprocess_workers = 1 if preprocess_workers is None else preprocess_workers
            self.precast_projection_stats = {"requested": precast_projections, "applied": False}

    monkeypatch.setattr(bench_extract.torch, "set_num_threads", set_threads)
    monkeypatch.setattr(bench_extract.torch, "get_num_threads", lambda: thread_count[0])
    monkeypatch.setattr(bench_extract.torch, "get_num_interop_threads", lambda: 3)
    monkeypatch.setattr(bench_extract.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(bench_extract, "read_inputs", read_inputs)
    monkeypatch.setattr(bench_extract, "Extractor", FakeModel)
    monkeypatch.setattr(bench_extract, "_git_commit", lambda: "fixture")
    monkeypatch.setattr(bench_extract, "_driver_version", lambda: None)
    output = tmp_path / "report.json"
    args = ["--model", str(tmp_path), "--device", "cpu", "--run-label", "test",
            "--warmup", "0", "--iterations", "1", "--batch-sizes", "1",
            "--throughput-seconds", "0.000001", "--json-out", str(output)]
    if requested is not None:
        args.extend(["--torch-threads", str(requested)])
    assert bench_extract.main(args) == 0
    expected = 8 if requested is None else requested
    assert calls == ([] if requested is None else [("threads", requested)]) + [
        ("read_inputs", expected), ("load", expected),
    ]
    report = json.loads(output.read_text())
    assert report["torch_threads_requested"] == requested
    assert report["torch_num_threads"] == expected
    assert report["torch_num_interop_threads"] == 3
    assert report["cpu_count"] == 8
    assert report["latency_batch1"]["iterations"] == 1
    assert report["throughput"][0]["batch_size"] == 1
    assert report["precast_projections"] == {"requested": False, "applied": False}


def test_projection_precast_is_explicit_opt_in():
    args = ["--model", "unused", "--run-label", "test"]
    assert bench_extract._parser().parse_args(args).precast_projections is False
    assert bench_extract._parser().parse_args(args + ["--precast-projections"]).precast_projections is True


@pytest.mark.parametrize("requested,resolved", [(None, "cuda:2"), ("cuda:4", "cuda:4"), ("cpu", "cpu")])
def test_benchmark_resolves_device_for_extractor_and_cuda_calls(tmp_path, monkeypatch, requested, resolved):
    calls = []

    def current_device():
        calls.append(("current", None))
        return 2

    def cuda_call(name, result=None):
        def call(device):
            # Matches the requirement that originally failed in set_device.
            assert device.type == "cuda" and device.index is not None
            calls.append((name, str(device)))
            return result
        return call

    class FakeModel(FakeExtractor):
        def __init__(self, model, device, preprocess_workers=None, precast_projections=False):
            super().__init__()
            calls.append(("load", device))
            self.preprocess_workers = 1
            self.precast_projection_stats = {"requested": precast_projections, "applied": False}

    monkeypatch.setattr(bench_extract.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bench_extract.torch.cuda, "current_device", current_device)
    for name in ("set_device", "synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(bench_extract.torch.cuda, name, cuda_call(name))
    monkeypatch.setattr(bench_extract.torch.cuda, "get_device_name", cuda_call("get_device_name", "mock GPU"))
    monkeypatch.setattr(bench_extract.torch.cuda, "max_memory_allocated", cuda_call("max_memory_allocated", 1024 ** 2))
    monkeypatch.setattr(bench_extract, "Extractor", FakeModel)
    monkeypatch.setattr(bench_extract, "read_inputs", lambda *_: _rows())
    monkeypatch.setattr(bench_extract, "_driver_version", lambda: None)
    monkeypatch.setattr(bench_extract, "_git_commit", lambda: "fixture")
    output = tmp_path / "report.json"
    args = ["--model", str(tmp_path), "--run-label", "device-test", "--warmup", "0",
            "--iterations", "1", "--batch-sizes", "1", "--throughput-seconds", "0.000001",
            "--json-out", str(output)]
    if requested is not None:
        args += ["--device", requested]
    assert bench_extract.main(args) == 0
    report = json.loads(output.read_text())
    assert report["device"]["requested"] == ("cuda" if requested is None else requested)
    assert report["device"]["resolved"] == resolved
    assert ("load", resolved) in calls
    if resolved == "cpu":
        assert calls == [("load", "cpu")]
        assert report["peak_vram_mb"] == 0
    else:
        assert (("current", None) in calls) == (requested is None)
        assert calls.index(("set_device", resolved)) < calls.index(("load", resolved))
        assert all(value == resolved for name, value in calls if name != "current")
        assert ("synchronize", resolved) in calls
        assert report["peak_vram_mb"] == 1
