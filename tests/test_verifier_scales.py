"""D-053: candidate-local scale alignment without query mixing or legacy drift."""

from pathlib import Path

import numpy as np
import pytest
import torch

from reid.post.search import SearchResult
from reid.post.verifier import PairVerifier, rerank_topk


class Scorer:
    def __init__(self, values=None):
        self.values = values or {
            "q0": [-5.0, 3.0, 9.0, -1.0],
            "q1": [2.0, -4.0, 3.0, 1.0],
        }
        self.calls = []

    def score_pairs(self, query, candidates):
        query_id = Path(query[0]).stem
        ids = [Path(item[0]).stem for item in candidates]
        self.calls.append((query_id, ids))
        return np.array([self.values[query_id][int(image_id[1:])] for image_id in ids])


def inputs():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    g = np.array([[0.9, 0.1], [0.85, 0.2], [0.8, 0.3], [-0.8, -0.3]], dtype=np.float32)
    ids = [f"g{i}" for i in range(len(g))]
    items = lambda names: [
        (Path(f"{name}.jpg"), (0.0, 0.0, 10.0, 10.0)) for name in names
    ]
    ranking = [["g0", "g1", "g2"], ["g2", "g1", "g0"]]
    return dict(
        query_items=items(["q0", "q1"]),
        gallery_items=items(ids),
        query_emb=q,
        gallery_emb=g,
        gallery_ids=ids,
        base=SearchResult(ranking, [row[0] for row in ranking], np.array([0.9, 0.3])),
    )


def z(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std(ddof=0) + 1e-9)


def test_standardized_matches_analytic_formula_and_keeps_cosine_confidence():
    args = inputs()
    scores = np.array([[2.0, 1.7, -0.5, 1e6], [-0.4, 0.1, 2.8, -1e6]])
    scorer = Scorer()
    result = rerank_topk(
        **args,
        verifier=scorer,
        weight=0.8,
        score_mode="standardized",
        ranking_scores=scores,
    )
    for i, candidate_ids in enumerate(args["base"].ranking):
        selected = [int(image_id[1:]) for image_id in candidate_ids]
        logits = [scorer.values[f"q{i}"][j] for j in selected]
        order = np.argsort(-(z(scores[i, selected]) + 0.8 * z(logits)), kind="stable")
        expected = [candidate_ids[j] for j in order]
        assert result.ranking[i] == expected
        assert result.top_ids[i] == expected[0]
        expected_cosine = (
            args["query_emb"][i] @ args["gallery_emb"][int(expected[0][1:])]
        )
        assert result.top_scores[i] == expected_cosine
    assert result.top_scores.dtype == np.float32
    # Scores outside the candidate set do not contribute to its normalization.
    scores[:, 3] *= -3
    repeated = rerank_topk(
        **args,
        verifier=Scorer(),
        weight=0.8,
        score_mode="standardized",
        ranking_scores=scores,
    )
    assert repeated.ranking == result.ranking


def test_standardized_is_invariant_to_positive_affine_scales():
    args = inputs()
    scores = args["query_emb"] @ args["gallery_emb"].T
    original = rerank_topk(
        **args,
        verifier=Scorer(),
        weight=0.8,
        score_mode="standardized",
        ranking_scores=scores,
    )
    transformed = Scorer(
        {
            "q0": np.array([-5, 3, 9, -1]) * 21.0 - 110.0,
            "q1": np.array([2, -4, 3, 1]) * 0.125 + 80.0,
        }
    )
    scores = scores.astype(np.float64) * np.array([[37.0], [0.4]]) + np.array(
        [[140.0], [-70.0]]
    )
    result = rerank_topk(
        **args,
        verifier=transformed,
        weight=0.8,
        score_mode="standardized",
        ranking_scores=scores,
    )
    assert result.ranking == original.ranking
    np.testing.assert_array_equal(result.top_scores, original.top_scores)


def test_standardized_without_matrix_uses_cosine():
    args = inputs()
    implicit = rerank_topk(**args, verifier=Scorer(), score_mode="standardized")
    explicit = rerank_topk(
        **args,
        verifier=Scorer(),
        score_mode="standardized",
        ranking_scores=args["query_emb"] @ args["gallery_emb"].T,
    )
    assert implicit.ranking == explicit.ranking
    np.testing.assert_array_equal(implicit.top_scores, explicit.top_scores)


def test_constant_logits_and_ties_preserve_base_order():
    args = inputs()
    scorer = Scorer({"q0": [7.0] * 4, "q1": [-2.0] * 4})
    result = rerank_topk(**args, verifier=scorer, score_mode="standardized")
    assert result.ranking == args["base"].ranking
    tied = rerank_topk(
        **args,
        verifier=scorer,
        score_mode="standardized",
        ranking_scores=np.zeros((2, 4)),
    )
    assert tied.ranking == args["base"].ranking
    assert np.isfinite(tied.top_scores).all()


@pytest.mark.parametrize("mode", ["raw", "standardized"])
def test_zero_weight_keeps_noncosine_base_order_without_scoring(mode):
    args = inputs()
    args["base"] = SearchResult(
        [["g2", "g0", "g1"], ["g0", "g2", "g1"]], ["g2", "g0"], np.array([123.0, 456.0])
    )
    scorer = Scorer()
    result = rerank_topk(
        **args,
        verifier=scorer,
        weight=0,
        score_mode=mode,
        ranking_scores=np.ones((2, 4)),
    )
    assert not scorer.calls
    assert result.ranking == args["base"].ranking
    assert result.top_ids == args["base"].top_ids
    np.testing.assert_array_equal(
        result.top_scores, np.array([0.8, 0.1], dtype=np.float32)
    )


@pytest.mark.parametrize("mode", ["raw", "standardized"])
def test_single_query_matches_reordered_batch(mode):
    args = inputs()
    scores = np.array([[1.0, 2.0, 3.0, -1.0], [8.0, -2.0, 3.0, 0.0]])
    batch = rerank_topk(
        **args, verifier=Scorer(), score_mode=mode, ranking_scores=scores
    )
    for i in (1, 0):
        single_args = dict(
            args,
            query_items=[args["query_items"][i]],
            query_emb=args["query_emb"][[i]],
            base=SearchResult(
                [args["base"].ranking[i]],
                [args["base"].top_ids[i]],
                args["base"].top_scores[[i]],
            ),
        )
        single = rerank_topk(
            **single_args,
            verifier=Scorer(),
            score_mode=mode,
            ranking_scores=scores[[i]],
        )
        assert single.ranking[0] == batch.ranking[i]
        assert single.top_scores[0] == batch.top_scores[i]


def test_default_is_unchanged_legacy_raw_formula():
    args = inputs()
    scorer = Scorer()
    default = rerank_topk(**args, verifier=scorer)
    explicit = rerank_topk(
        **args,
        verifier=Scorer(),
        score_mode="raw",
        ranking_scores=-100 * (args["query_emb"] @ args["gallery_emb"].T),
    )
    assert (
        default.ranking == explicit.ranking
    )  # ranking_scores is standardized-mode only.
    for i, ids in enumerate(args["base"].ranking):
        selected = [int(image_id[1:]) for image_id in ids]
        cosine = args["query_emb"][i] @ args["gallery_emb"][selected].T
        logits = np.array(
            [scorer.values[f"q{i}"][j] for j in selected], dtype=np.float32
        )
        score = cosine + 0.2 / (1 + np.exp(-np.clip(logits, -80.0, 80.0)))
        order = np.argsort(-score, kind="stable")
        assert default.ranking[i] == [ids[j] for j in order]
        assert default.top_scores[i] == cosine[order[0]]


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 3)),
        np.zeros(8),
        np.full((2, 4), np.nan),
        np.full((2, 4), np.inf),
        np.full((2, 4), "bad"),
        np.ones((2, 4), dtype=complex),
    ],
)
def test_invalid_ranking_matrix_fails_before_scoring(bad):
    scorer = Scorer()
    with pytest.raises(ValueError, match="ranking_scores"):
        rerank_topk(
            **inputs(), verifier=scorer, score_mode="standardized", ranking_scores=bad
        )
    assert not scorer.calls


@pytest.mark.parametrize(
    "field,bad",
    [
        ("query_emb", np.zeros(2)),
        ("gallery_emb", np.zeros((4, 3))),
        ("query_emb", np.full((2, 2), np.nan)),
        ("gallery_emb", np.full((4, 2), np.inf)),
        ("query_emb", np.full((2, 2), "bad")),
    ],
)
def test_invalid_embeddings_fail_before_scoring(field, bad):
    args = dict(inputs(), **{field: bad})
    scorer = Scorer()
    with pytest.raises(ValueError, match="embeddings"):
        rerank_topk(**args, verifier=scorer)
    assert not scorer.calls


@pytest.mark.parametrize("ids", [["g0", "missing"], [], ["g0", "g0"]])
def test_invalid_later_row_fails_before_scoring_first_query(ids):
    args = inputs()
    args["base"].ranking[1] = ids
    scorer = Scorer()
    with pytest.raises(ValueError, match="base ranking"):
        rerank_topk(**args, verifier=scorer)
    assert not scorer.calls


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_invalid_weight_fails_before_scoring(weight):
    scorer = Scorer()
    with pytest.raises(ValueError, match="weight"):
        rerank_topk(**inputs(), verifier=scorer, weight=weight)
    assert not scorer.calls


def test_unknown_score_mode_is_rejected_even_at_zero_weight():
    scorer = Scorer()
    with pytest.raises(ValueError, match="score_mode"):
        rerank_topk(**inputs(), verifier=scorer, score_mode="typo", weight=0)
    assert not scorer.calls


@pytest.mark.parametrize("bad", [[1, 2], [1, 2, float("nan")]])
def test_invalid_verifier_output_is_rejected(bad):
    class BadScorer:
        def score_pairs(self, query, candidates):
            return bad

    with pytest.raises(ValueError, match="finite logit"):
        rerank_topk(**inputs(), verifier=BadScorer(), score_mode="standardized")


@pytest.mark.parametrize("head", [None, "concat", "corr"])
def test_checkpoint_dispatch_reuses_training_head(tmp_path, monkeypatch, head):
    from reid.model import corr_verifier
    from reid.infer import verifier

    calls = []

    class ToyNetwork(torch.nn.Module):
        def __init__(self, arch, input_size, pretrained):
            super().__init__()
            calls.append((type(self).__name__, arch, input_size, pretrained))

    class ToyCorr(ToyNetwork):
        pass

    monkeypatch.setattr(verifier, "VerifierNetwork", ToyNetwork)
    monkeypatch.setattr(corr_verifier, "CorrVerifier", ToyCorr)
    payload = {"arch": "test", "input": 28, "state": {}}
    if head is not None:
        payload["head"] = head
    checkpoint = tmp_path / "verifier.pt"
    torch.save(payload, checkpoint)
    loaded = PairVerifier(checkpoint, device="cpu", preprocess_workers=1)
    assert calls == [
        ("ToyCorr" if head == "corr" else "ToyNetwork", "test", (28, 28), False)
    ]
    assert not loaded.model.training
    assert loaded.cfg["input_size"] == [28, 28]


def test_unknown_checkpoint_head_fails_before_network_construction(
    tmp_path, monkeypatch
):
    from reid.infer import verifier

    def forbidden(*args, **kwargs):
        pytest.fail("unknown head must not construct the legacy architecture")

    monkeypatch.setattr(verifier, "VerifierNetwork", forbidden)
    checkpoint = tmp_path / "verifier.pt"
    torch.save(
        {"arch": "test", "input": 28, "state": {}, "head": "not_supported"}, checkpoint
    )
    with pytest.raises(ValueError, match="unsupported verifier head"):
        PairVerifier(checkpoint, device="cpu")
