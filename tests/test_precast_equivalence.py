"""Numeric gate and train-only orchestration; all extractors/CUDA calls are fake."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from reid.infer.run import InputRow
from scripts import check_precast_equivalence as parity


def rows(count):
    return [InputRow(f"train{i}", Path(f"train{i}.jpg"), (1, 2, 3, 4)) for i in range(count)]


class FakeExtractor:
    instances = []
    perturb = False
    apply_cast = True

    def __init__(self, _model=None, _device=None, preprocess_workers=8, precast_projections=False):
        self.cfg = {"embed_dim": 3}
        self.enabled = precast_projections
        self.precast_projection_stats = {"requested": precast_projections,
                                         "applied": precast_projections and self.apply_cast}
        self._preprocess_pool = None
        self.calls = []
        self.instances.append(self)

    def extract_batch(self, items):
        self.calls.append(items)
        values = np.array([[int(Path(p).stem.removeprefix("train")) + 1, 2, 3]
                           for p, _ in items], dtype=np.float64)
        values = (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float32)
        if self.enabled and self.perturb:
            values[0, 0] += np.float32(1e-7)
        return values


@pytest.fixture
def fake_run(monkeypatch):
    FakeExtractor.instances = []
    FakeExtractor.perturb = False
    FakeExtractor.apply_cast = True
    inputs = rows(65)
    def train_rows(data_root, splits, fold):
        assert fold == 0
        return inputs[:33], inputs[33:]
    monkeypatch.setattr(parity, "train_rows", train_rows)
    monkeypatch.setattr(parity, "Extractor", FakeExtractor)
    monkeypatch.setattr(parity, "model_sources", lambda *_: {"component/weights": {"sha256": "a" * 64, "bytes": 4}})
    monkeypatch.setattr(parity, "file_record", lambda path: {
        "sha256": hashlib.sha256(str(path).encode()).hexdigest(), "bytes": 1,
    })
    monkeypatch.setattr(parity, "_git_commit", lambda: "fixture")
    monkeypatch.setattr(parity, "_driver_version", lambda: None)
    monkeypatch.setattr(parity.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(parity.torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(parity.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(parity.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(parity.torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(parity.torch.cuda, "get_device_name", lambda *_: "MOCK CUDA; no actual GPU measurement")
    return inputs


def test_exact_comparison_and_one_float_ulp_change():
    a = np.eye(3, dtype=np.float32)
    same = parity.compare_embeddings(a, a.copy())
    assert same["array_equal"] and same["max_abs_diff"] == 0 and same["min_cosine"] == 1
    b = a.copy()
    b[0, 0] = np.nextafter(b[0, 0], np.float32(2))
    different = parity.compare_embeddings(a, b)
    assert not different["array_equal"] and different["different_rows"] == 1
    assert different["max_abs_diff"] > 0 and different["max_norm_diff"] > 0


@pytest.mark.parametrize("case", ["shape", "dtype", "nan", "zero"])
def test_invalid_embeddings_cannot_pass(case):
    a = np.eye(3, dtype=np.float32)
    b = a.copy()
    if case == "shape":
        b = b[:2]
    elif case == "dtype":
        b = b.astype(np.float64)
    elif case == "nan":
        b[0, 0] = np.nan
    else:
        b[0] = 0
    with pytest.raises(ValueError):
        parity.compare_embeddings(a, b)


def test_sample_is_exactly_32_distinct_uniform_train_indices():
    indices = parity.uniform_indices(65)
    assert len(indices) == len(set(indices)) == 32
    assert indices[0] == 0 and indices[-1] == 64
    assert max(np.diff(indices)) - min(np.diff(indices)) <= 1
    with pytest.raises(ValueError):
        parity.uniform_indices(31)


def test_fusion_source_manifest_hashes_every_loaded_config_and_weight(tmp_path):
    manifest = {"schema_version": 1, "name": "fixture", "candidate_policy": "gallery_z",
                "candidate_threshold": 6.3, "rerank_mode": "standardized", "components": []}
    for i in range(2):
        path = tmp_path / "components" / str(i)
        path.mkdir(parents=True)
        (path / "weights.safetensors").write_bytes(b"weight fixture")
        (path / "model_config.yaml").write_text("infer_dtype: fp16\n")
        manifest["components"].append({"path": f"components/{i}", "similarity_std": .1})
    (tmp_path / "fusion_config.json").write_text(json.dumps(manifest))
    before = parity.model_sources(tmp_path)
    assert len(before) == 5
    assert all(len(record["sha256"]) == 64 and record["bytes"] > 0 for record in before.values())
    (tmp_path / "components/1/weights.safetensors").write_bytes(b"other weights")
    after = parity.model_sources(tmp_path)
    assert before["fusion_config.json"] == after["fusion_config.json"]
    assert before["components/1/weights.safetensors"] != after["components/1/weights.safetensors"]


@pytest.mark.parametrize("save_arrays", [False, True])
def test_run_covers_every_fold_row_and_32_batch1_with_identical_paired_inputs(tmp_path, fake_run, save_arrays):
    out = tmp_path / "output"
    result = parity.run(tmp_path / "model", tmp_path / "dataset", out, save_arrays=save_arrays)
    a, b = FakeExtractor.instances
    assert a.calls == b.calls
    assert [len(batch) for batch in a.calls] == [32, 32, 1] + [1] * 32
    assert [str(p) for batch in a.calls[:3] for p, _ in batch] == [str(row.image_path) for row in fake_run]
    assert result["passed"] is True and result["fold"] == 0
    assert result["n_query"] == 33 and result["n_gallery"] == 32
    assert result["stages"]["full_fold0"]["rows"] == 65
    assert result["stages"]["batch1_sample"]["rows"] == 32
    assert result["stages"]["full_fold0"]["max_abs_diff"] == 0
    assert result["precast_cast_stats"]["applied"] is True
    assert result["model_sources_unchanged"] and result["input_sources_unchanged"]
    assert not any("test_query" in name or "test_gallery" in name for name in result["input_sources"])
    assert json.loads((out / "report.json").read_text()) == result
    assert len(list(out.glob("*.npz"))) == (2 if save_arrays else 0)
    if save_arrays:
        with np.load(out / "full_fold0.npz", allow_pickle=False) as saved:
            assert saved["baseline"].shape == (65, 3)
            assert np.array_equal(saved["baseline"], saved["precast"])


def test_any_difference_is_failure_even_if_cosine_is_nearly_one(tmp_path, fake_run):
    FakeExtractor.perturb = True
    result = parity.run(tmp_path / "model", tmp_path / "dataset", tmp_path / "output")
    assert result["passed"] is False and len(result["failures"]) == 2
    assert result["stages"]["full_fold0"]["max_abs_diff"] > 0


@pytest.mark.parametrize("requested,resolved,current_calls", [("cuda", "cuda:3", 1), ("cuda:1", "cuda:1", 0)])
def test_cuda_without_index_resolves_current_device_before_set_device(
        tmp_path, fake_run, monkeypatch, requested, resolved, current_calls):
    current, selected = [], []
    def current_device():
        current.append(True)
        return 3
    def set_device(device):
        assert device.type == "cuda" and device.index is not None
        selected.append(str(device))
    monkeypatch.setattr(parity.torch.cuda, "current_device", current_device)
    monkeypatch.setattr(parity.torch.cuda, "set_device", set_device)
    result = parity.run(tmp_path / "model", tmp_path / "dataset", tmp_path / "output", device=requested)
    assert result["passed"] is True
    assert selected == [resolved] and len(current) == current_calls
    assert result["device"]["requested"] == requested
    assert result["device"]["resolved"] == resolved


def test_sources_changed_during_run_fail_gate(tmp_path, fake_run, monkeypatch):
    calls = []
    def changed(_model):
        calls.append(True)
        return {"hash": str(len(calls))}
    monkeypatch.setattr(parity, "model_sources", changed)
    result = parity.run(tmp_path / "model", tmp_path / "dataset", tmp_path / "output")
    assert result["passed"] is False
    assert result["failures"] == ["source files changed during comparison"]


def test_unapplied_precast_never_reports_success(tmp_path, fake_run):
    FakeExtractor.apply_cast = False
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="no projection"):
        parity.run(tmp_path / "model", tmp_path / "dataset", out)
    report = json.loads((out / "report.json").read_text())
    assert report["passed"] is False and report["failures"]


def test_cpu_and_existing_output_are_rejected_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(parity, "Extractor", lambda *_a, **_k: pytest.fail("must reject before model load"))
    with pytest.raises(ValueError, match="CUDA"):
        parity.run(tmp_path / "model", tmp_path / "dataset", tmp_path / "new", device="cpu")
    with pytest.raises(FileExistsError):
        parity.run(tmp_path / "model", tmp_path / "dataset", tmp_path)


@pytest.mark.parametrize("passed", [False, True])
def test_cli_exit_code_reflects_exact_gate(monkeypatch, passed):
    monkeypatch.setattr(parity, "run", lambda **_: {"passed": passed, "failures": [] if passed else ["different"]})
    assert parity.main(["--model", "unused", "--out", "unused"]) == (0 if passed else 1)
