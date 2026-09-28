"""CPU mocks only: transport/C5 orchestration, never claims nvJPEG GPU parity."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reid.infer.run import InputRow
from scripts import experiment_nvjpeg_fusion as exp


def rows(n=64):
    return [InputRow(str(i), Path(f"{i}.jpg"), (0, 0, 2, 2)) for i in range(n)]


@pytest.fixture
def decoder(monkeypatch):
    reads, decodes = [], []
    def read(path):
        reads.append(path)
        return torch.tensor([255, 216, int(Path(path).stem)], dtype=torch.uint8)
    def decode(encoded, mode, device, apply_exif_orientation):
        assert isinstance(encoded, list) and all(x.device.type == "cpu" and x.dtype == torch.uint8 for x in encoded)
        assert mode == exp.ImageReadMode.RGB and str(device) == "cuda:0"
        assert apply_exif_orientation is False
        decodes.append(len(encoded))
        return [torch.full((3, 2, 2), int(x[-1]), dtype=torch.uint8) for x in encoded]
    def base(self, items):
        return np.array([[int(self._decode_frame(item)[0, 0, 0]), 1, 2] for item in items], dtype=np.float32)
    monkeypatch.setattr(exp, "read_file", read)
    monkeypatch.setattr(exp, "decode_jpeg", decode)
    monkeypatch.setattr(exp.Extractor, "extract_batch", base)
    obj = exp.NvJpegFusionExtractor.__new__(exp.NvJpegFusionExtractor)
    obj.device = torch.device("cuda:0")
    obj._preprocess_pool = None
    obj._nvjpeg_frames = None
    obj._call_lock = threading.Lock()
    return obj, reads, decodes


@pytest.mark.parametrize("workers", [0, 2])
def test_reads_every_jpeg_on_every_call_and_clears_call_local_frames(decoder, workers):
    obj, reads, decodes = decoder
    obj._preprocess_pool = ThreadPoolExecutor(workers) if workers else None
    items = [(Path(f"{i}.jpg"), (0, 0, 2, 2)) for i in (3, 1, 3)]
    try:
        for _ in range(2):
            assert obj.extract_batch(items)[:, 0].tolist() == [3, 1, 3]
            assert obj._nvjpeg_frames is None
        assert sorted(reads) == ["1.jpg", "1.jpg", "3.jpg", "3.jpg", "3.jpg", "3.jpg"]
        assert decodes == [3, 3]
        with pytest.raises(RuntimeError, match="current batch"):
            obj._decode_frame(items[0])
    finally:
        if obj._preprocess_pool:
            obj._preprocess_pool.shutdown()


@pytest.mark.parametrize("failure", ["decode", "forward", "magic", "png"])
def test_failed_calls_never_retain_frames_or_lock(decoder, monkeypatch, failure):
    obj, _, _ = decoder
    def fail(*_a, **_k):
        raise RuntimeError("fixture failure")
    path = Path("0.jpg")
    if failure == "decode":
        monkeypatch.setattr(exp, "decode_jpeg", fail)
    elif failure == "forward":
        monkeypatch.setattr(exp.Extractor, "extract_batch", fail)
    elif failure == "magic":
        monkeypatch.setattr(exp, "read_file", lambda _: torch.zeros(3, dtype=torch.uint8))
    else:
        path = Path("0.png")
    with pytest.raises((RuntimeError, ValueError)):
        obj.extract_batch([(path, (0, 0, 2, 2))])
    assert obj._nvjpeg_frames is None
    assert obj._call_lock.acquire(blocking=False)
    obj._call_lock.release()


def test_concurrent_call_rejected_without_mutating_active_cache(decoder):
    obj, reads, _ = decoder
    obj._call_lock.acquire()
    sentinel = {"owned_by_first_call": object()}
    obj._nvjpeg_frames = sentinel
    with pytest.raises(RuntimeError, match="concurrent"):
        obj.extract_batch([(Path("0.jpg"), (0, 0, 2, 2))])
    assert obj._nvjpeg_frames is sentinel and reads == []
    obj._call_lock.release()


def test_quality_rejects_all_data_before_gpu_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "load_fusion_config", lambda _: {"components": [{"path": "0"}]})
    monkeypatch.setattr(exp, "load_config", lambda _: {"train": {"folds": "all"}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("must reject before GPU"))
    with pytest.raises(ValueError, match="all-data forbidden"):
        exp.run(tmp_path / "model", tmp_path / "data", tmp_path / "out", mode="quality")
    assert not (tmp_path / "out").exists()


def test_pixel_drift_uses_32_train_rows_and_signed_difference():
    base = SimpleNamespace(_decode_frame=lambda _: torch.zeros(3, 2, 2, dtype=torch.uint8))
    nv = SimpleNamespace(decoded_frames=lambda items: [torch.full((3, 2, 2), 2, dtype=torch.uint8) for _ in items])
    result = exp.pixel_drift(base, nv, rows())
    assert len(result) == 32 and result[0]["image_id"] == "0" and result[-1]["image_id"] == "63"
    assert all(r["max_abs"] == r["mean_abs"] == 2 and r["changed_fraction"] == 1 for r in result)


def test_quality_uses_same_production_rank_config_without_verifier(tmp_path, monkeypatch):
    seen, evaluations = [], []
    cfg = {"components": [{"path": "0"}], "candidate_threshold": 6.3, "candidate_policy": "gallery_z"}
    monkeypatch.setattr(exp, "load_config", lambda _: {"train": {"folds": "all except 0"}})
    factory = lambda *_a, **_k: SimpleNamespace(_preprocess_pool=None)
    monkeypatch.setattr(exp, "Extractor", factory)
    monkeypatch.setattr(exp, "NvJpegFusionExtractor", factory)
    monkeypatch.setattr(exp, "extract_rows", lambda _e, r, batch: np.tile(np.array([[1, 0]], np.float32), (len(r), 1)))
    def rank(query, gallery, q, g, passed_cfg, verifier=None):
        assert passed_cfg is cfg and verifier is None
        seen.append((len(query), len(gallery)))
        return "ranking"
    monkeypatch.setattr(exp, "rank_embeddings", rank)
    monkeypatch.setattr(exp, "write_outputs", lambda *_: None)
    monkeypatch.setattr(exp, "validate_outputs", lambda *_: None)
    monkeypatch.setattr(exp, "read_submission", lambda _: {})
    monkeypatch.setattr(exp, "read_candidates", lambda _: [])
    def evaluate(fold, ranking, candidates, folds_path):
        assert fold == 0 and folds_path.endswith("folds.csv")
        evaluations.append(True)
        return {"map10": .85 + len(evaluations) * .001}
    monkeypatch.setattr(exp, "evaluate", evaluate)
    monkeypatch.setattr(exp, "pixel_drift", lambda *_: [])
    result = exp.quality_run(tmp_path, cfg, rows(32), rows(32), tmp_path, tmp_path, "cuda:0", 8)
    assert seen == [(32, 32), (32, 32)] and len(evaluations) == 2
    assert result["map10_delta"] == pytest.approx(.001)
    assert result["embedding_drift"]["array_equal"] is True and result["verifier_enabled"] is False


def test_bench_keeps_official_c5_protocol_and_no_git_is_allowed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(exp, "load_fusion_config", lambda _: {"components": []})
    monkeypatch.setattr(exp, "model_sources", lambda _: {"weights": {"sha256": "a" * 64}})
    monkeypatch.setattr(exp, "file_record", lambda _: {"sha256": "b" * 64, "bytes": 1})
    monkeypatch.setattr(exp, "_git_commit", lambda: None)
    monkeypatch.setattr(exp, "_driver_version", lambda: None)
    monkeypatch.setattr(exp, "_weights_mb", lambda _: 1.)
    monkeypatch.setattr(exp, "train_rows", lambda _data, _splits, fold: (rows(), rows()))
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "MOCK GPU")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 1024**2)
    monkeypatch.setattr(exp, "NvJpegFusionExtractor", lambda *_a, **_k: SimpleNamespace(
        _preprocess_pool=None, precast_projection_stats={"requested": False, "applied": False}))
    def latency(_e, _r, warmup, iterations, _sync):
        calls.append(("latency", warmup, iterations))
        return {"p50_ms": 1., "p90_ms": 1., "iterations": iterations}
    def throughput(_e, _r, batch, seconds, _sync):
        calls.append(("throughput", batch, seconds))
        return {"batch_size": batch, "fps": 1.}
    monkeypatch.setattr(exp, "measure_latency", latency)
    monkeypatch.setattr(exp, "measure_throughput", throughput)
    result = exp.run(tmp_path / "model", tmp_path / "data", tmp_path / "out")
    assert calls == [("latency", 50, 300)] + [("throughput", batch, 10.) for batch in (1, 8, 16, 32)]
    assert result["commit"] is None and result["precast_projections"]["applied"] is False
    assert json.loads((tmp_path / "out/report.json").read_text())["status"].startswith("completed")
