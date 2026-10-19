"""Opt-in D082 regression and exact comparison with A's research implementation."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from reid.infer.list_ranker import DevelopmentGraphScorer
from reid.infer.run import InputRow, write_outputs
from reid.model.list_ranker import CandidateGraphRanker
from reid.post.pipeline import PreparedGallery
from reid.post.ranker import PreparedRankerGallery
from scripts.analysis.top5_features import build_features, z
from scripts.replay_fusion import CachedVerifier, fuse_components, load_components


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _items(ids):
    return [(Path(i), (0, 0, 1, 1)) for i in ids]


@pytest.fixture
def data():
    rng = np.random.default_rng(20261017)
    arrays = []
    for dim in (18, 14):
        q = rng.normal(size=(4, dim)).astype(np.float32)
        g = rng.normal(size=(15, dim)).astype(np.float32)
        arrays.append(
            (
                q / np.linalg.norm(q, axis=1, keepdims=True),
                g / np.linalg.norm(g, axis=1, keepdims=True),
            )
        )
    return arrays, [f"g{i}" for i in range(15)], [f"q{i}" for i in range(4)]


@pytest.fixture
def head(tmp_path):
    torch.manual_seed(91)
    model = CandidateGraphRanker().eval()
    with torch.no_grad():
        model.head[-1].weight.normal_(std=0.5)
    save_file(model.state_dict(), tmp_path / "weights.safetensors")
    cfg = dict(
        features=13,
        hidden=64,
        depth=2,
        outer=0,
        actual_supervised_folds=[1, 3, 4],
        excluded_folds=[0, 2],
    )
    (tmp_path / "model_config.json").write_text(json.dumps(cfg))
    nested = []
    for view in (1, 3, 4):
        rec = dict(
            actual_train_folds=[f for f in (1, 3, 4) if f != view],
            excluded_folds=sorted([0, 2, view]),
            weights_sha256="a" * 64,
            config_sha256="b" * 64,
        )
        nested.append(dict(view=view, provenance=[dict(receipt=rec, export=rec)] * 2))
    receipt = dict(
        actual_train_folds=[1, 3, 4],
        excluded_folds=[0, 2],
        nested_supervision=nested,
        weights_sha256=_sha(tmp_path / "weights.safetensors"),
        config_sha256=_sha(tmp_path / "model_config.json"),
    )
    (tmp_path / "training_receipt.json").write_text(json.dumps(receipt))
    args = dict(
        outer_fold=0,
        weights_sha256=receipt["weights_sha256"],
        config_sha256=receipt["config_sha256"],
        receipt_sha256=_sha(tmp_path / "training_receipt.json"),
    )
    return tmp_path, args


class FixedVerifier:
    def score_pairs(self, query, candidates):
        # Uses only the current query's original stable gallery ID, no batch state.
        return np.array(
            [(int(str(g[0])[1:]) % 7) * 0.37 for g in candidates], np.float32
        )


def _prepared(data, ranker=None, residual=0.5):
    arrays, gids, _ = data
    q, g, scales = fuse_components(arrays)
    adapter = PreparedRankerGallery(
        [gc for _, gc in arrays],
        gids,
        _items(gids),
        scales,
        ranker=ranker,
        residual_weight=residual,
    )
    return adapter, q, g


def test_features_identical_to_research(data):
    arrays, gids, _ = data
    reference = build_features(arrays, gids)
    prepared, _, _ = _prepared(data)
    actual = prepared.features([qc for qc, _ in arrays])
    for key in ("nodes", "edges", "base", "indices", "candidates", "q"):
        np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)
    np.testing.assert_array_equal(prepared.gallery.embeddings, reference["g"])


@pytest.mark.parametrize("mode", ["disabled", "zero"])
def test_disabled_and_explicit_zero_three_files_byte_equal(data, tmp_path, mode):
    class MustNotRun:
        def score_lists(self, *args):
            raise AssertionError("zero residual should never evaluate the head")

    arrays, gids, qids = data
    prepared, q, g = _prepared(
        data,
        None if mode == "disabled" else MustNotRun(),
        0.5 if mode == "disabled" else 0,
    )
    baseline = PreparedGallery(
        g,
        gids,
        _items(gids),
        dict(
            candidate_policy="gallery_z", rerank_mode="standardized", rerank_weight=0.15
        ),
    )
    expected = baseline.rank(q, _items(qids), FixedVerifier())
    actual = prepared.rank([qc for qc, _ in arrays], _items(qids), FixedVerifier())
    rows = lambda ids: [InputRow(i, Path(i), (0, 0, 1, 1)) for i in ids]
    for name, result in (("expected", expected), ("actual", actual)):
        write_outputs(tmp_path / name, rows(qids), rows(gids), q, g, -100, result)
    for name in ("submission.csv", "embeddings.npy", "candidates.csv"):
        assert (tmp_path / "expected" / name).read_bytes() == (
            tmp_path / "actual" / name
        ).read_bytes()
    assert (tmp_path / "actual" / "candidates.csv").stat().st_size > 100


def test_head_score_rank_and_confidence_match_research(data, head):
    arrays, gids, qids = data
    scorer = DevelopmentGraphScorer(head[0], **head[1])
    prepared, q, _ = _prepared(data, scorer)
    ref = build_features(arrays, gids)
    reference_scores = scorer.model.score_lists(
        *[torch.from_numpy(ref[k]) for k in ("nodes", "edges", "base")]
    ).numpy()
    actual_scores = scorer.score_lists(ref["nodes"], ref["edges"], ref["base"])
    np.testing.assert_array_equal(reference_scores, actual_scores)
    logits = np.stack(
        [
            FixedVerifier().score_pairs(query, _items(cands))
            for query, cands in zip(_items(qids), ref["candidates"])
        ]
    )
    raw = np.take_along_axis(ref["policy"].ranking_scores(q), ref["indices"], 1)
    scores = z(raw) + 0.15 * z(logits) + 0.5 * (reference_scores - ref["base"])
    expected = np.take_along_axis(
        ref["candidates"], np.argsort(-scores, axis=1, kind="stable"), 1
    ).tolist()
    actual = prepared.rank([qc for qc, _ in arrays], _items(qids), FixedVerifier())
    assert actual.ranking == expected
    assert any(row != base.tolist() for row, base in zip(expected, ref["candidates"]))
    for row, base in zip(actual.ranking, ref["candidates"]):
        assert set(row) == set(base)
    np.testing.assert_array_equal(
        actual.top_scores, ref["policy"].confidence_for(q, [r[0] for r in expected])
    )


def test_query_independence_and_gallery_copy(data, head):
    arrays, _, qids = data
    scorer = DevelopmentGraphScorer(head[0], **head[1])
    prepared, _, _ = _prepared(data, scorer)
    original = prepared.rank([qc for qc, _ in arrays], _items(qids), FixedVerifier())
    # Caller mutation must not change retained gallery state.
    for _, gc in arrays:
        gc[:] = 0
    order = [3, 0, 2, 1]
    shuffled = prepared.rank(
        [qc[order] for qc, _ in arrays],
        _items([qids[i] for i in order]),
        FixedVerifier(),
    )
    for i, j in enumerate(order):
        single = prepared.rank(
            [qc[j : j + 1] for qc, _ in arrays], _items([qids[j]]), FixedVerifier()
        )
        assert single.ranking[0] == shuffled.ranking[i] == original.ranking[j]
        assert single.top_scores.tobytes() == original.top_scores[j : j + 1].tobytes()


def test_changed_top1_gets_its_own_gallery_confidence(data):
    class SwapHead:
        def score_lists(self, nodes, edges, base):
            scores = base.copy()
            scores[:, 1] += 100
            return scores

    arrays, _, qids = data
    prepared, q, _ = _prepared(data, SwapHead())
    feat = prepared.features([qc for qc, _ in arrays])
    result = prepared.rank([qc for qc, _ in arrays], _items(qids), FixedVerifier())
    assert result.top_ids == feat["candidates"][:, 1].tolist()
    expected = prepared.gallery.policy.confidence_for(q, result.top_ids)
    np.testing.assert_array_equal(result.top_scores, expected)
    assert not np.array_equal(
        result.top_scores,
        prepared.gallery.policy.confidence_for(q, feat["candidates"][:, 0]),
    )


@pytest.mark.parametrize("key", ["weights_sha256", "config_sha256", "receipt_sha256"])
def test_bad_sha_rejected(head, key):
    args = dict(head[1], **{key: "0" * 64})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        DevelopmentGraphScorer(head[0], **args)


@pytest.mark.parametrize("fold", [2, 5, True, "0"])
def test_no_sealed_or_wrong_head(head, fold):
    with pytest.raises(ValueError, match="only OOF"):
        DevelopmentGraphScorer(head[0], **dict(head[1], outer_fold=fold))


@pytest.mark.parametrize(
    "field,value",
    [
        ("depth", 0),
        ("features", 14),
        ("hidden", 32),
        ("actual_supervised_folds", [1, 2, 3, 4]),
        ("actual_supervised_folds", [True, 3, 4]),
        ("outer", False),
    ],
)
def test_bad_config_rejected_even_when_rehashed(head, field, value):
    root, args = head
    cfg = json.loads((root / "model_config.json").read_text())
    cfg[field] = value
    (root / "model_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        DevelopmentGraphScorer(
            root, **dict(args, config_sha256=_sha(root / "model_config.json"))
        )


def test_nested_fold_leak_rejected_even_when_rehashed(head):
    root, args = head
    receipt = json.loads((root / "training_receipt.json").read_text())
    receipt["nested_supervision"][0]["provenance"][0]["receipt"][
        "actual_train_folds"
    ] = [2, 3, 4]
    (root / "training_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="nested encoder"):
        DevelopmentGraphScorer(
            root, **dict(args, receipt_sha256=_sha(root / "training_receipt.json"))
        )


def test_configuration_and_input_guards(data):
    arrays, gids, qids = data
    _, _, scales = fuse_components(arrays)
    for bad in (-1, 0.25, 1, float("nan"), True):
        with pytest.raises(ValueError, match="residual"):
            PreparedRankerGallery(
                [g for _, g in arrays], gids, _items(gids), scales, residual_weight=bad
            )
    for scales_bad in ([0, 1], [float("nan"), 1], [True, 1]):
        with pytest.raises(ValueError, match="scale"):
            PreparedRankerGallery(
                [g for _, g in arrays], gids, _items(gids), scales_bad
            )
    prepared, _, _ = _prepared(data)
    with pytest.raises(ValueError, match="exact-window verifier"):
        prepared.rank([q for q, _ in arrays], _items(qids), None)
    with pytest.raises(ValueError, match="shape"):
        prepared.features([arrays[0][0][:1], arrays[1][0]])


@pytest.mark.skipif(
    not os.environ.get("FALCON_RANKER_HANDOFF"), reason="optional verified A package"
)
@pytest.mark.parametrize("fold", [0, 1, 3, 4])
def test_real_c4_exact_feature_score_window_confidence_parity(fold):
    root = Path(os.environ["FALCON_RANKER_HANDOFF"])
    manifest = json.loads((root / "MANIFEST.json").read_text())
    spec = manifest["folds"][str(fold)]
    # This check is adapter parity only, not quality tuning or final acceptance.
    with np.load(root / spec["dumps"][0], allow_pickle=False) as npz:
        qids, gids = npz["query_ids"].tolist(), npz["gallery_ids"].tolist()
    arrays = load_components([root / p for p in spec["dumps"]], qids, gids)
    ref = build_features(arrays, gids)
    head = spec["head"]
    pins = manifest["files"]
    scorer = DevelopmentGraphScorer(
        root / head,
        outer_fold=fold,
        weights_sha256=pins[head + "/weights.safetensors"],
        config_sha256=pins[head + "/model_config.json"],
        receipt_sha256=pins[head + "/training_receipt.json"],
    )
    prepared, _, _ = _prepared((arrays, gids, qids), scorer)
    feat = prepared.features([q for q, _ in arrays])
    for key in ("nodes", "edges", "base", "indices", "candidates", "q"):
        np.testing.assert_array_equal(feat[key], ref[key], err_msg=key)
    cache = CachedVerifier(
        root / spec["logits"], fold, qids, ref["candidates"].tolist()
    )
    with np.load(root / spec["logits"], allow_pickle=False) as npz:
        logits = npz["logits"]
    head_scores = scorer.model.score_lists(
        *[torch.from_numpy(ref[k]) for k in ("nodes", "edges", "base")]
    ).numpy()
    raw = np.take_along_axis(ref["policy"].ranking_scores(ref["q"]), ref["indices"], 1)
    scores = z(raw) + 0.15 * z(logits) + 0.5 * (head_scores - ref["base"])
    expected = np.take_along_axis(
        ref["candidates"], np.argsort(-scores, axis=1, kind="stable"), 1
    ).tolist()
    actual = prepared.rank([q for q, _ in arrays], _items(qids), cache)
    assert actual.ranking == expected
    np.testing.assert_array_equal(
        actual.top_scores,
        ref["policy"].confidence_for(ref["q"], [r[0] for r in expected]),
    )
