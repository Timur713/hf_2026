import csv
import hashlib

import numpy as np

from scripts.check_determinism import compare_outputs, output_hashes, subset_csv
from scripts.check_determinism import prepare_cuda_measurement


def _write_outputs(root, value):
    root.mkdir()
    (root / "submission.csv").write_bytes(value)
    (root / "candidates.csv").write_bytes(value + b"-candidates")
    np.save(root / "embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))


def test_compare_outputs_checks_all_three_artifacts(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_outputs(first, b"same")
    _write_outputs(second, b"same")

    comparison = compare_outputs(first, second)

    assert set(comparison) == {"submission.csv", "embeddings.npy", "candidates.csv"}
    assert all(result["identical"] for result in comparison.values())
    assert output_hashes(first)["submission.csv"] == hashlib.sha256(b"same").hexdigest()


def test_compare_outputs_detects_one_changed_byte(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_outputs(first, b"same")
    _write_outputs(second, b"different")

    comparison = compare_outputs(first, second)

    assert comparison["submission.csv"]["identical"] is False
    assert comparison["candidates.csv"]["identical"] is False
    assert comparison["embeddings.npy"]["identical"] is True


def test_subset_csv_copies_deterministic_prefix(tmp_path):
    source, destination = tmp_path / "source.csv", tmp_path / "subset.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "x", "y", "w", "h"])
        for index in range(4):
            writer.writerow([f"image-{index}", 1, 2, 3, 4])

    assert subset_csv(source, destination, 2) == 2
    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["image_id"] for row in rows] == ["image-0", "image-1"]


def test_cuda_counters_are_reset_only_after_lazy_initialization(monkeypatch):
    import torch

    events = []
    monkeypatch.setattr(torch.cuda, "init", lambda: events.append("init"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: events.append("reset"))
    assert prepare_cuda_measurement(torch.device("cpu")) == 0.0
    assert events == []
    assert prepare_cuda_measurement(torch.device("cuda:0")) >= 0
    assert events == ["init", "sync", "reset"]
