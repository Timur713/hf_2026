import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.prepare_fusion_artifact as prepare


def test_gallery_std_matches_research_formula():
    g = np.random.default_rng(12).normal(size=(260, 9)).astype(np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    expected = float((g @ g.T).std())
    assert prepare.gallery_similarity_std(g) == pytest.approx(expected, rel=1e-6)
    assert prepare.gallery_similarity_std(g[::-1]) == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError, match="degenerate"):
        prepare.gallery_similarity_std(np.ones((12, 2), dtype=np.float32))


def fixture_bundle(tmp_path, monkeypatch):
    sources = []
    for i in range(2):
        root = tmp_path / f"model{i}"
        root.mkdir()
        (root / "model_config.yaml").write_text(yaml.safe_dump({"name": f"m{i}", "train": {"folds": "all"}}))
        (root / "weights.safetensors").write_bytes(bytes([i + 1]) * 20)
        (root / "not-an-inference-file.txt").write_text("must not enter artifact")
        sources.append(root)
    gallery = tmp_path / "gallery.csv"
    with gallery.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "x", "y", "w", "h"])
        for i in range(12):
            writer.writerow([f"g{i}", 0, 0, 10, 10])
            (tmp_path / f"g{i}.jpg").touch()

    class FakeExtractor:
        def __init__(self, model_dir, device):
            # Calibration uses the copied artifact, not the mutable source.
            assert Path(model_dir).parent.name == "components"
            self.emb = np.random.default_rng(int(Path(model_dir).name)).normal(size=(12, 8)).astype(np.float32)
        def extract_batch(self, items):
            return np.stack([self.emb[int(Path(path).stem[1:])] for path, _ in items])

    monkeypatch.setattr(prepare, "Extractor", FakeExtractor)
    return sources, gallery


def test_prepares_only_selected_files_and_records_real_hashes(tmp_path, monkeypatch):
    sources, gallery = fixture_bundle(tmp_path, monkeypatch)
    out = tmp_path / "bundle"
    audit = prepare.prepare_bundle(sources, tmp_path, gallery, out, require_all_data=True)
    manifest = json.loads((out / "fusion_config.json").read_text())
    assert manifest["candidate_threshold"] == 6.0
    assert len(manifest["components"]) == 2
    assert all(c["similarity_std"] > 0 for c in manifest["components"])
    assert audit["total_weight_bytes"] == 40
    assert audit["gallery_count"] == 12
    for relative, record in audit["artifacts"].items():
        assert hashlib.sha256((out / relative).read_bytes()).hexdigest() == record["sha256"]
    assert not list(out.rglob("not-an-inference-file.txt"))
    with pytest.raises(FileExistsError):
        prepare.prepare_bundle(sources, tmp_path, gallery, out)


def test_preparation_guards_final_metadata_and_size(tmp_path, monkeypatch):
    sources, gallery = fixture_bundle(tmp_path, monkeypatch)
    cfg = sources[0] / "model_config.yaml"
    cfg.write_text(yaml.safe_dump({"train": {"folds": "all except 0"}}))
    with pytest.raises(ValueError, match="not trained on all"):
        prepare.prepare_bundle(sources, tmp_path, gallery, tmp_path / "out", require_all_data=True)
    monkeypatch.setattr(prepare, "MAX_WEIGHT_BYTES", 39)
    with pytest.raises(ValueError, match="exceed 2 GB"):
        prepare.prepare_bundle(sources, tmp_path, gallery, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_explicit_threshold_is_packaged_with_calibration_source(tmp_path, monkeypatch):
    sources, gallery = fixture_bundle(tmp_path, monkeypatch)
    out = tmp_path / "bundle"
    audit = prepare.prepare_bundle(sources, tmp_path, gallery, out,
        candidate_threshold=6.3, threshold_source="D-063 / A-103, OOF window", rerank_weight=0.15)
    cfg = json.loads((out / "fusion_config.json").read_text())
    assert cfg["candidate_threshold"] == 6.3
    assert cfg["rerank_weight"] == 0.15
    assert audit["candidate_threshold_source"] == "D-063 / A-103, OOF window"


@pytest.mark.parametrize("threshold,source", [
    (6.3, None), (6.3, " "), (0, "D-063"), (-1, "D-063"),
    (float("nan"), "D-063"), (float("inf"), "D-063"), (True, "D-063"),
])
def test_threshold_fails_closed_before_preparation(tmp_path, threshold, source):
    with pytest.raises(ValueError):
        prepare.prepare_bundle([], tmp_path, tmp_path / "missing", tmp_path / "out",
                               candidate_threshold=threshold, threshold_source=source)
    assert not (tmp_path / "out").exists()


def test_failed_calibration_does_not_publish_partial_artifact(tmp_path, monkeypatch):
    sources, gallery = fixture_bundle(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError("decode failed")
    monkeypatch.setattr(prepare, "extract_rows", fail)
    with pytest.raises(RuntimeError, match="decode failed"):
        prepare.prepare_bundle(sources, tmp_path, gallery, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["metadata", "size"])
def test_revalidates_copied_snapshot(tmp_path, monkeypatch, mutation):
    sources, gallery = fixture_bundle(tmp_path, monkeypatch)
    copyfile = prepare.shutil.copyfile
    def changing_copy(source, destination):
        result = copyfile(source, destination)
        if Path(source).parent == sources[0]:
            if mutation == "metadata" and Path(source).name == "model_config.yaml":
                Path(destination).write_text(yaml.safe_dump({"train": {"folds": "all except 0"}}))
            if mutation == "size" and Path(source).name == "weights.safetensors":
                Path(destination).write_bytes(b"x" * 60)
        return result
    monkeypatch.setattr(prepare.shutil, "copyfile", changing_copy)
    monkeypatch.setattr(prepare, "MAX_WEIGHT_BYTES", 50)
    with pytest.raises(ValueError, match="copied"):
        prepare.prepare_bundle(sources, tmp_path, gallery, tmp_path / "out", require_all_data=True)
    assert not (tmp_path / "out").exists()
