import csv
import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from scripts.diagnose_c3_gap import run, train_rows, validate_components


@pytest.fixture
def fixture(tmp_path):
    data, splits = tmp_path / "data", tmp_path / "splits"
    (data / "images").mkdir(parents=True)
    splits.mkdir()
    qids, gids = ["q0", "q1", "q_open"], [f"g{i}" for i in range(12)]
    for role, ids in (("query", qids), ("gallery", gids)):
        with (splits / f"fold0_{role}.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            writer.writerows((iid, 1, 1, 10, 10) for iid in ids)
    with (splits / "folds.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "vehicle_id", "camera_id", "fold", "role"])
        writer.writerows((iid, vid, 0, 0, "query") for iid, vid in zip(qids, [0, 1, 99]))
        writer.writerows((iid, i, 1, 0, "gallery") for i, iid in enumerate(gids))
    with (data / "train.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"])
        writer.writerows((iid, 1, 1, 10, 10, 0, 0) for iid in qids + gids)
    for i, iid in enumerate(qids + gids):
        pixels = np.random.default_rng(i).integers(0, 256, size=(16, 20, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(data / "images" / f"{iid}.jpg")
    components = []
    for i in range(2):
        component = tmp_path / f"model{i}"
        component.mkdir()
        cfg = {"arch": "stub", "embed_dim": 3, "pool": "avg", "train": {"folds": "all except 0"},
               "input_size": [8, 8], "mean": [0, 0, 0], "std": [1, 1, 1], "bbox_pad": 0,
               "resize": "stretch", "infer_dtype": "fp16"}
        (component / "model_config.yaml").write_text(yaml.safe_dump(cfg))
        (component / "weights.safetensors").write_bytes(b"fixture-not-real-weights")
        components.append(component)
    return data, splits, components


def test_all_data_and_wrong_fold_rejected(fixture):
    _, _, components = fixture
    assert len(validate_components(components, 0)) == 2
    path = components[0] / "model_config.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["train"]["folds"] = "all"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="all except"):
        validate_components(components, 0)
    with pytest.raises(ValueError, match="train folds"):
        validate_components(components, 5)


def test_fold_ids_must_be_in_train(fixture):
    data, splits, _ = fixture
    assert len(train_rows(data, splits, 0)[0]) == 3
    path = data / "train.csv"
    path.write_text("\n".join(line for line in path.read_text().splitlines() if not line.startswith("q_open,")) + "\n")
    with pytest.raises(ValueError, match="exact train.csv"):
        train_rows(data, splits, 0)


def test_cpu_diagnostic_cache_real_images_and_replay(fixture, tmp_path, monkeypatch):
    data, splits, components = fixture

    class Fake(torch.nn.Module):
        def features(self, batch):
            return batch.mean(dim=(2, 3))

        def forward(self, batch):
            raise AssertionError("Must bypass neck and use features")

    monkeypatch.setattr("reid.infer.extractor.load", lambda *args: Fake())
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git is absent in the production image")
    monkeypatch.setattr("scripts.diagnose_c3_gap.subprocess.check_output", no_git)
    out = tmp_path / "diagnostic"
    variants = ["full_fp32", "cache512_fp32", "cache1024_fp32"]
    report = run(components, variants, 0, data, splits, out, "cpu", batch=4, workers=2)
    assert set(report["results"]) == set(variants)
    assert len(report["extraction"]) == 6
    assert not report["training_provenance_verified"]
    assert report["code_commit"].startswith("unavailable")
    assert len(report["diagnostic_script"]["sha256"]) == 64
    for variant in variants:
        with np.load(out / variant / "component0_fold0.npz", allow_pickle=False) as dump:
            assert dump["query_ids"].tolist() == ["q0", "q1", "q_open"]
            assert dump["gallery_emb"].dtype == np.float32
            np.testing.assert_allclose(np.linalg.norm(dump["gallery_emb"], axis=1), 1, atol=1e-6)
        replay = json.loads((out / variant / "replay" / "report.json").read_text())
        assert replay["similarity_std"] == report["similarity_std"]
        assert set(replay["stages"]) == {"fusion", "hub"}
        assert replay["logits"] is None
    with pytest.raises(FileExistsError):
        run(components, variants, 0, data, splits, out, "cpu")
    with pytest.raises(ValueError, match="CUDA"):
        run(components, ["full_fp16"], 0, data, splits, tmp_path / "fp16", "cpu")
