import csv
import json
from pathlib import Path

import numpy as np
import pytest

from reid.infer.run import InputRow, rank_embeddings
from reid.post.gallery_policy import GalleryPolicy
from reid.post.policy import rerank_weight_for
from scripts.replay_fusion import CachedVerifier, fuse_components, load_components, replay


@pytest.fixture
def inputs(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    qids, gids = ["q0", "q1", "q_open"], [f"g{i}" for i in range(12)]
    for name, ids in (("query", qids), ("gallery", gids)):
        with (splits / f"fold0_{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            writer.writerows((i, 0, 0, 10, 10) for i in ids)
    with (splits / "folds.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "vehicle_id", "camera_id", "fold", "role"])
        writer.writerows((q, i, 0, 0, "query") for q, i in zip(qids, [0, 1, 99]))
        writer.writerows((g, i, 1, 0, "gallery") for i, g in enumerate(gids))
    paths = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        g = rng.normal(size=(12, 64)).astype(np.float32)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        q = np.stack([g[0], g[1], rng.normal(size=64)]).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        path = tmp_path / f"component{seed}.npz"
        np.savez(path, query_ids=np.array(qids), gallery_ids=np.array(gids), query_emb=q, gallery_emb=g)
        paths.append(path)
    return splits, paths, qids, gids


def test_replay_uses_production_and_writes_valid_c6(inputs, tmp_path):
    splits, paths, qids, gids = inputs
    out = tmp_path / "out"
    report = replay(paths, 0, out, splits=splits)
    assert not report["training_provenance_verified"]
    assert set(report["stages"]) == {"fusion", "hub"}
    q, g, _ = fuse_components(load_components(paths, qids, gids))
    rows = lambda ids: [InputRow(i, Path(i), (0, 0, 1, 1)) for i in ids]
    expected = rank_embeddings(rows(qids), rows(gids), q, g, report["policy"])
    with (out / "hub/submission.csv").open() as f:
        actual = [row[1:] for row in csv.reader(f)]
    assert actual == expected.ranking
    assert report["stages"]["hub"]["metrics"]["map10"] == 1
    assert json.loads((out / "report.json").read_text())["similarity_std"] == report["similarity_std"]
    with pytest.raises(FileExistsError):
        replay(paths, 0, out, splits=splits)


@pytest.mark.parametrize("corruption", [None, "order", "window", "fold", "nan"])
def test_exact_logit_window_required(inputs, tmp_path, corruption):
    splits, paths, qids, _ = inputs
    replay(paths, 0, tmp_path / "base", splits=splits)
    with np.load(tmp_path / "base/candidate_window.npz", allow_pickle=False) as cache:
        cand = cache["candidates"].copy()
    logits = np.zeros(cand.shape, dtype=np.float32)
    fold = 0
    if corruption == "order":
        cand = cand[:, ::-1]
    elif corruption == "window":
        cand = np.repeat(cand, 3, axis=1)
        logits = np.zeros(cand.shape, dtype=np.float32)
    elif corruption == "fold":
        fold = 1
    elif corruption == "nan":
        logits[0, 0] = np.nan
    path = tmp_path / "logits.npz"
    np.savez(path, query_ids=np.array(qids), candidates=cand, logits=logits, fold=fold)
    out = tmp_path / "with_verifier"
    if corruption:
        with pytest.raises(ValueError):
            replay(paths, 0, out, splits=splits, logits=path)
        assert not out.exists()
    else:
        report = replay(paths, 0, out, splits=splits, logits=path, rerank_weight=0.15)
        assert report["policy"]["rerank_weight"] == 0.15
        assert report["stages"]["hub"]["files"] == report["stages"]["verifier"]["files"]


def test_c4_order_and_normalization_are_not_silently_repaired(inputs):
    _, paths, qids, gids = inputs
    with pytest.raises(ValueError, match="IDs/order"):
        load_components(paths, qids[::-1], gids)
    with np.load(paths[0]) as data:
        values = dict(data)
    values["query_emb"] *= 2
    np.savez(paths[0], **values)
    with pytest.raises(ValueError, match="L2-normalized"):
        load_components(paths, qids, gids)


def test_replay_fixed_scales_and_no_test_fold(inputs, tmp_path):
    splits, paths, _, _ = inputs
    result = replay(paths, 0, tmp_path / "fixed", splits=splits, scales=[0.1, 0.2])
    assert result["similarity_std"] == [0.1, 0.2]
    assert result["scale_source"] == "provided fixed constants"
    with pytest.raises(ValueError, match="train folds"):
        replay(paths, 5, tmp_path / "illegal", splits=splits)
    with pytest.raises(ValueError, match="one positive"):
        replay(paths, 0, tmp_path / "bad-scale", splits=splits, scales=[0.1])


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True])
def test_rerank_weight_policy_validates(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        rerank_weight_for({"rerank_weight": value})


def test_weight_policy_legacy_artifact_and_override():
    assert rerank_weight_for({}) == 0.2
    assert rerank_weight_for({"rerank_weight": 0.15}) == 0.15
    assert rerank_weight_for({"rerank_weight": 0.15}, 0) == 0


def test_verifier_changes_top1_and_abstains_using_new_candidate(inputs, tmp_path):
    splits, paths, qids, gids = inputs
    q, g, _ = fuse_components(load_components(paths, qids, gids))
    replay(paths, 0, tmp_path / "base", splits=splits)
    with np.load(tmp_path / "base/candidate_window.npz", allow_pickle=False) as cache:
        candidates = cache["candidates"].copy()
    logits = np.zeros(candidates.shape, dtype=np.float32)
    logits[:, -1] = 10
    cache_path = tmp_path / "logits.npz"
    np.savez(cache_path, fold=0, query_ids=np.array(qids), candidates=candidates, logits=logits)
    policy = GalleryPolicy(g, gids)
    old_conf = policy.confidence_for(q, candidates[:, 0].tolist())
    new_conf = policy.confidence_for(q, candidates[:, -1].tolist())
    assert old_conf[0] > new_conf[0]
    threshold = float((old_conf[0] + new_conf[0]) / 2)
    report = replay(paths, 0, tmp_path / "changed", splits=splits, logits=cache_path,
                    rerank_weight=100, candidate_threshold=threshold)
    rows = lambda ids: [InputRow(i, Path(i), (0, 0, 1, 1)) for i in ids]
    result = rank_embeddings(rows(qids), rows(gids), q, g, report["policy"],
                             CachedVerifier(cache_path, 0, qids, candidates.tolist()))
    assert result.top_ids == candidates[:, -1].tolist()
    np.testing.assert_allclose(result.top_scores, new_conf)
    with (tmp_path / "changed/verifier/candidates.csv").open() as handle:
        accepted = {row["query_id"] for row in csv.DictReader(handle)}
    assert "q0" not in accepted
    with (tmp_path / "changed/hub/candidates.csv").open() as handle:
        assert "q0" in {row["query_id"] for row in csv.DictReader(handle)}


def test_ranking_does_not_validate_unused_candidate_threshold(inputs):
    _, paths, qids, gids = inputs
    q, g, _ = fuse_components(load_components(paths, qids, gids))
    rows = lambda ids: [InputRow(i, Path(i), (0, 0, 1, 1)) for i in ids]
    assert rank_embeddings(rows(qids), rows(gids), q, g,
                           {"candidate_threshold": float("nan")}).ranking
