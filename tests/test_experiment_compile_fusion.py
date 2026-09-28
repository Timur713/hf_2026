"""CPU orchestration tests only; no claim about CUDA compile speed or accuracy."""
import argparse
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reid.infer.run import InputRow
from scripts import experiment_compile_fusion as exp


def rows(n=64):
    return [InputRow(str(i), Path(f"{i}.jpg"), (0, 0, 2, 2)) for i in range(n)]


def arguments(tmp_path, **changes):
    return argparse.Namespace(model=tmp_path / "model", data_root=tmp_path / "data", splits=tmp_path / "splits",
                              out=tmp_path / "out", device="cuda:0", workers=8, torch_threads=8,
                              max_seconds=840, _worker=False, **changes)


def test_wraps_actual_features_not_forward_and_preserves_weights(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2)
        def features(self, x):
            return self.projection(x)
        def forward(self, x):
            pytest.fail("Extractor should not invoke forward")
    members = [Model().eval().requires_grad_(False) for _ in range(3)]
    weights = [{key: val.clone() for key, val in m.state_dict().items()} for m in members]
    compiled, calls = [], []
    def compile(fn, **kwargs):
        assert kwargs == exp.COMPILE_OPTIONS
        compiled.append(fn.__self__)
        def wrapper(x):
            calls.append(fn.__self__)
            return fn(x)
        return wrapper
    monkeypatch.setattr(torch, "compile", compile)
    extractor = SimpleNamespace(device=torch.device("cuda:0"), _fusion_components=[
        (m, {"input_size": [392, 392]}, torch.float16, 1.) for m in members])
    record = exp.compile_features(extractor)
    assert compiled == members and len(record) == 3
    for member, state in zip(members, weights):
        member.features(torch.ones(1, 2))
        assert all(torch.equal(v, member.state_dict()[k]) for k, v in state.items())
    assert calls == members
    assert all(r["callable"] == "features" for r in record)


@pytest.mark.parametrize("invalid", ["cpu", "single", "training", "gradient"])
def test_rejects_unsupported_or_mutable_models(invalid):
    model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    if invalid == "training":
        model.train()
    if invalid == "gradient":
        model.requires_grad_(True)
    extractor = SimpleNamespace(device=torch.device("cpu" if invalid == "cpu" else "cuda:0"),
                                _fusion_components=None if invalid == "single" else [(model, {}, None, 1)])
    with pytest.raises(ValueError):
        exp.compile_features(extractor)


def test_parity_reports_exact_drift_and_hashes():
    a = np.array([[1, 0]], np.float32)
    assert exp.parity_record(a, a.copy())["array_equal"] is True
    b = np.array([[1, .001]], np.float32)
    result = exp.parity_record(a, b)
    assert result["array_equal"] is False and result["max_abs_diff"] == pytest.approx(.001)
    assert result["eager_array_sha256"] != result["compiled_array_sha256"]
    assert "compiled_max_unit_norm_error" in result and "precast_max_unit_norm_error" not in result


@pytest.fixture
def worker_setup(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.out.mkdir()
    trace = []
    state = {"compiled": False, "drift": False, "fail_compile": False}
    monkeypatch.setattr(exp, "load_fusion_config", lambda _: {"components": [1, 2, 3]})
    monkeypatch.setattr(exp, "model_sources", lambda _: {"weights": {"sha256": "a" * 64}})
    monkeypatch.setattr(exp, "file_record", lambda _: {"sha256": "b" * 64, "bytes": 1})
    monkeypatch.setattr(exp, "_git_commit", lambda: None)
    monkeypatch.setattr(exp, "_driver_version", lambda: None)
    monkeypatch.setattr(exp, "train_rows", lambda _data, _splits, fold: (rows(), []))
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: trace.append(("device", str(device))))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "MOCK GPU")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 1024**2)
    class MockExtractor:
        _preprocess_pool = None
        precast_projection_stats = {"requested": False, "applied": False}
        cfg = {"embed_dim": 2}
        def __init__(self, model, device, **kwargs):
            assert kwargs == {"preprocess_workers": 8, "precast_projections": False}
        def extract_batch(self, items):
            trace.append(("extract", state["compiled"], tuple(str(x[0]) for x in items)))
            return np.tile(np.array([[1., .001 if state["compiled"] and state["drift"] else 0.]],
                                    np.float32), (len(items), 1))
    monkeypatch.setattr(exp, "Extractor", MockExtractor)
    def compile(_e):
        if state["fail_compile"]:
            raise RuntimeError("graph break fixture")
        state["compiled"] = True
        return [{"callable": "features"}]
    monkeypatch.setattr(exp, "compile_features", compile)
    def latency(_e, _rows, warmup, iterations, sync):
        trace.append(("latency", state["compiled"], warmup, iterations))
        return {"p50_ms": 1., "p90_ms": 2.}
    def throughput(_e, _rows, batch, seconds, sync):
        trace.append(("throughput", state["compiled"], batch, seconds))
        return {"batch_size": batch, "fps": 90.}
    monkeypatch.setattr(exp, "measure_latency", latency)
    monkeypatch.setattr(exp, "measure_throughput", throughput)
    return args, trace, state


@pytest.mark.parametrize("drift", [False, True])
def test_paired_train_samples_and_complete_c5_for_both_paths(worker_setup, drift):
    args, trace, state = worker_setup
    state["drift"] = drift
    args.device = "cuda"
    assert exp.worker(args) == (2 if drift else 0)
    result = json.loads((args.out / "report.json").read_text())
    assert result["sample_exact_equal"] is not drift and result["production_acceptance"] is False
    assert result["commit"] is None and result["model_sources_unchanged"] is True
    assert result["device"]["resolved"] == "cuda:3"
    assert len(result["sample_image_ids"]) == 32 and len(set(result["sample_image_ids"])) == 32
    eager = [x[2] for x in trace if x[:2] == ("extract", False)]
    compiled = [x[2] for x in trace if x[:2] == ("extract", True)]
    assert eager == compiled
    for variant in (False, True):
        assert ("latency", variant, 50, 300) in trace
        assert [("throughput", variant, b, 10.) for b in exp.BATCHES] == [
            t for t in trace if t[:2] == ("throughput", variant)]


def test_compiler_failure_keeps_honest_eager_report(worker_setup):
    args, trace, state = worker_setup
    state["fail_compile"] = True
    assert exp.worker(args) == 1
    result = json.loads((args.out / "report.json").read_text())
    assert result["status"] == "failed" and "graph break" in result["error"]
    assert "eager" in result and "compiled" not in result and result["production_acceptance"] is False


@pytest.mark.parametrize("invalid", ["existing", "model_child", "data_child", "cpu", "long", "zero"])
def test_supervisor_rejects_unsafe_setup_before_spawn(tmp_path, monkeypatch, invalid):
    args = arguments(tmp_path)
    if invalid == "existing":
        args.out.mkdir()
    elif invalid.endswith("child"):
        args.out = (args.model if invalid == "model_child" else args.data_root) / "out"
    elif invalid == "cpu":
        args.device = "cpu"
    else:
        args.max_seconds = 900 if invalid == "long" else 0
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("must reject before spawn"))
    with pytest.raises((ValueError, FileExistsError)):
        exp.supervise(args)


def test_timeout_stops_only_own_process_group_and_records_failure(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    seen = []
    class Process:
        pid = 54321
        def wait(self, timeout=None):
            if timeout == args.max_seconds:
                raise subprocess.TimeoutExpired("compile", timeout)
            return -15
    def spawn(cmd, **kwargs):
        assert kwargs["start_new_session"] is True and "--_worker" in cmd
        assert kwargs["env"]["TORCHINDUCTOR_CACHE_DIR"] == str(args.out / "inductor_cache")
        return Process()
    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(exp.os, "killpg", lambda pid, sig: seen.append((pid, sig)))
    assert exp.supervise(args) == 124
    assert seen == [(54321, signal.SIGTERM), (54321, signal.SIGKILL)]
    result = json.loads((args.out / "report.json").read_text())
    assert result["status"] == "timeout" and result["production_acceptance"] is False


def test_abnormal_worker_exit_never_leaves_running_report(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: SimpleNamespace(wait=lambda **_k: -9))
    assert exp.supervise(args) == -9
    result = json.loads((args.out / "report.json").read_text())
    assert result["status"] == "failed" and "-9" in result["error"]
