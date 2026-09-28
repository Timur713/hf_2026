"""CPU-only synthetic checks for the archived diagnostic helper, no real data."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "fullframe_replay_helper", Path(__file__).with_name("replay_fullframe_embeddings.py"))
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)


@pytest.fixture
def inputs(tmp_path):
    splits, images = tmp_path / "splits", tmp_path / "images"
    splits.mkdir()
    images.mkdir()
    qids, gids = ["q0", "q1", "q_open"], [f"g{i}" for i in range(12)]
    for role, ids in (("query", qids), ("gallery", gids)):
        with (splits / f"fold0_{role}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            writer.writerows((image_id, 0, 0, 10, 10) for image_id in ids)
        for image_id in ids:
            # Empty files prove that this helper does not decode any image.
            (images / f"{image_id}.jpg").touch()
    with (splits / "folds.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "vehicle_id", "camera_id", "fold", "role"])
        writer.writerows((q, vid, 0, 0, "query") for q, vid in zip(qids, [0, 1, 99]))
        writer.writerows((g, i, 1, 0, "gallery") for i, g in enumerate(gids))
    rng = np.random.default_rng(2026)
    g = rng.normal(size=(12, 64)).astype(np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    q = np.stack([g[0], g[1], rng.normal(size=64)]).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    emb = tmp_path / "embeddings.npy"
    np.save(emb, np.concatenate([q, g]))
    cfg = tmp_path / "fusion_config.json"
    cfg.write_text(json.dumps({
        "schema_version": 1, "name": "synthetic_no_weights",
        "components": [{"path": f"components/{i}", "similarity_std": scale}
                       for i, scale in enumerate([0.1, 0.2])],
        "candidate_policy": "gallery_z", "candidate_threshold": 6.,
        "rerank_mode": "standardized", "rerank_weight": 0.15}))
    return emb, cfg, images, splits


def test_cpu_replay_without_weights_or_decoding(inputs, tmp_path):
    emb, cfg, images, splits = inputs
    out = tmp_path / "out"
    report = helper.replay(emb, cfg, images, out, splits=splits)
    assert report["metrics"]["map10"] == 1.0
    assert report["verifier_applied"] is False
    assert report["component_scales_reapplied"] is False
    assert report["csv_order_verified_against_fold_truth"] is True
    assert report["embedding_row_identity_verified"] is False
    assert report["extraction_provenance_verified"] is False
    assert report["files"]["embeddings.npy"]["sha256"] == report["input_embeddings"]["sha256"]
    with pytest.raises(FileExistsError):
        helper.replay(emb, cfg, images, out, splits=splits)


@pytest.mark.parametrize("corruption", ["dtype", "count", "empty_dim", "norm", "nan"])
def test_malformed_embeddings_rejected(inputs, tmp_path, corruption):
    emb, cfg, images, splits = inputs
    matrix = np.load(emb, allow_pickle=False)
    bad = {"dtype": lambda: matrix.astype(np.float64), "count": lambda: matrix[:-1],
           "empty_dim": lambda: matrix[:, :0], "norm": lambda: matrix * 2,
           "nan": lambda: matrix * np.float32(np.nan)}[corruption]()
    np.save(emb, bad)
    with pytest.raises(ValueError):
        helper.replay(emb, cfg, images, tmp_path / "out", splits=splits)
    assert not (tmp_path / "out").exists()


def test_csv_identity_order_rejected(inputs, tmp_path):
    emb, cfg, images, splits = inputs
    path = splits / "fold0_query.csv"
    with path.open() as handle:
        rows = list(csv.reader(handle))
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows([rows[0], *rows[1:][::-1]])
    with pytest.raises(ValueError, match="IDs/order"):
        helper.replay(emb, cfg, images, tmp_path / "out", splits=splits)
    assert not (tmp_path / "out").exists()


def test_duplicate_json_metadata_rejected(inputs):
    _, cfg, _, _ = inputs
    cfg.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        helper.read_metadata(cfg)
