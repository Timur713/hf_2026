"""Synthetic parity checks for the exact D-055/D-057 research formulas."""

import numpy as np
import pytest

from reid.post.gallery_policy import GalleryPolicy


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fixture_vectors():
    rng = np.random.default_rng(294)
    return unit(rng.normal(size=(7, 17))), unit(rng.normal(size=(12, 17)))


def research_formula(q, g, k=3, weight=0.5):
    # Intentionally spell out the research calculations rather than calling
    # production helpers: concat_parts() and fused() operate on these vectors.
    raw = q @ g.T
    z = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-9)
    gg = g @ g.T
    off = ~np.eye(len(g), dtype=bool)
    confidence = (raw - gg[off].mean()) / (gg[off].std() + 1e-12)
    np.fill_diagonal(gg, -np.inf)
    hub = np.sort(gg, axis=1)[:, -k:].mean(axis=1)
    hub = (hub - hub.mean()) / (hub.std() + 1e-9)
    return z - weight * hub, confidence, hub


def test_gallery_policy_matches_research_and_separates_confidence_from_ranking():
    q, g = fixture_vectors()
    ids = [f"g{i}" for i in range(len(g))]
    policy = GalleryPolicy(g, ids)
    expected, confidence, hub = research_formula(q, g)
    np.testing.assert_allclose(policy.ranking_scores(q), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(policy.hub, hub)
    order = np.argsort(-expected, axis=1, kind="stable")[:, :10]
    result = policy.search(q)
    assert result.ranking == [[ids[j] for j in row] for row in order]
    assert result.top_ids == [ids[row[0]] for row in order]
    selected = confidence[np.arange(len(q)), order[:, 0]]
    np.testing.assert_allclose(result.top_scores, selected, atol=1e-6)
    assert not np.allclose(result.top_scores, expected[np.arange(len(q)), order[:, 0]])
    np.testing.assert_array_equal(result.top_scores, policy.confidence_for(q, result.top_ids))


def test_query_alone_batch_reordering_and_unrelated_extreme_query_are_identical():
    q, g = fixture_vectors()
    policy = GalleryPolicy(g, [f"g{i}" for i in range(len(g))])
    batched = policy.search(q)
    scores = policy.ranking_scores(q)
    for i in range(len(q)):
        alone = policy.search(q[i:i + 1])
        assert alone.ranking == [batched.ranking[i]]
        np.testing.assert_array_equal(alone.top_scores, batched.top_scores[i:i + 1])
        np.testing.assert_array_equal(policy.ranking_scores(q[i:i + 1]), scores[i:i + 1])
    permutation = [5, 0, 6, 2, 1, 4, 3]
    reordered = policy.search(q[permutation])
    assert reordered.ranking == [batched.ranking[i] for i in permutation]
    np.testing.assert_array_equal(reordered.top_scores, batched.top_scores[permutation])
    extended = np.concatenate([q, g[:1], -g[:1]], axis=0)
    np.testing.assert_array_equal(policy.ranking_scores(extended)[:len(q)], scores)


def test_gallery_diagonal_is_excluded_for_hub_and_confidence():
    q, g = fixture_vectors()
    policy = GalleryPolicy(g, [str(i) for i in range(len(g))], hub_k=1)
    expected, confidence, hub = research_formula(q, g, k=1)
    gg = g @ g.T
    off_diagonal = gg[~np.eye(len(g), dtype=bool)]
    assert policy.gallery_mean == off_diagonal.mean()
    assert policy.gallery_std == off_diagonal.std(ddof=0)
    assert not np.isclose(policy.gallery_mean, gg.mean())
    np.testing.assert_array_equal(policy.hub, hub)
    np.testing.assert_allclose(policy.ranking_scores(q), expected, atol=1e-6)
    np.testing.assert_allclose(policy.confidence_for(q, ["2"] * len(q)), confidence[:, 2], atol=1e-6)


def test_identical_candidate_ties_keep_gallery_csv_order():
    g = unit([[1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])
    ids = ["second_lexically", "first_lexically", "c", "d", "e"]
    policy = GalleryPolicy(g, ids)
    result = policy.search(unit([[1, 0]]), top_k=5)
    assert result.ranking[0][:2] == ids[:2]


def test_constant_query_row_uses_ranking_epsilon_without_nan():
    g = unit([[1, 0, 0], [1, 1, 0], [-1, 0, 0], [0, -1, 0]])
    q = unit([[0, 0, 1]])
    policy = GalleryPolicy(g, [str(i) for i in range(len(g))])
    expected, _, _ = research_formula(q, g)
    np.testing.assert_array_equal(policy.ranking_scores(q), expected)
    assert np.isfinite(policy.search(q, top_k=4).top_scores).all()


def test_constant_hub_is_zero_and_keeps_raw_order():
    g = unit([[1, 0], [0, 1], [-1, 0], [0, -1]])
    q = unit([[2, 1]])
    policy = GalleryPolicy(g, [str(i) for i in range(len(g))])
    np.testing.assert_array_equal(policy.hub, np.zeros(len(g), dtype=np.float32))
    assert policy.search(q, top_k=4).ranking == [["0", "1", "3", "2"]]


def test_distinct_ranking_and_confidence_epsilons_match_research():
    # Almost orthogonal gallery: tiny *off-diagonal* variance makes the
    # confidence epsilon visible; substituting ranking's 1e-9 would fail.
    g = unit(np.eye(4) + 1e-10 * np.arange(16).reshape(4, 4))
    q = unit([[1, 2, 3, 4]])
    policy = GalleryPolicy(g, [str(i) for i in range(len(g))])
    _, expected, _ = research_formula(q, g)
    np.testing.assert_array_equal(policy.confidence_for(q, ["2"]), expected[:, 2])
    # A unit query almost orthogonal to the whole gallery makes row std
    # comparable to the ranking epsilon, without violating the input contract.
    g = unit([[1, 0, 0], [1, 1, 0], [-1, 0, 0], [0, -1, 0]])
    q = unit([[1e-9, 2e-9, 1]])
    policy = GalleryPolicy(g, [str(i) for i in range(len(g))])
    expected, _, _ = research_formula(q, g)
    np.testing.assert_array_equal(policy.ranking_scores(q), expected)


def test_gallery_copy_is_not_affected_by_caller_mutation():
    q, g = fixture_vectors()
    ids = [str(i) for i in range(len(g))]
    policy = GalleryPolicy(g, ids)
    before = policy.search(q)
    g[:] = 0
    ids[0] = "replaced"
    after = policy.search(q)
    assert before.ranking == after.ranking
    np.testing.assert_array_equal(before.top_scores, after.top_scores)


@pytest.mark.parametrize("g", [np.ones((4, 3)), np.eye(4), np.empty((0, 3))])
def test_degenerate_gallery_fails_clearly(g):
    with pytest.raises(ValueError, match="degenerate|hub_k"):
        GalleryPolicy(g, [str(i) for i in range(len(g))])


@pytest.mark.parametrize("hub_k", [0, -1, 12, 13, 2.5, True])
def test_invalid_hub_k(hub_k):
    _, g = fixture_vectors()
    with pytest.raises(ValueError, match="hub_k"):
        GalleryPolicy(g, [str(i) for i in range(len(g))], hub_k=hub_k)


@pytest.mark.parametrize("weight", [-0.1, np.nan, np.inf, "0.5", True, [0.5]])
def test_invalid_hub_weight(weight):
    _, g = fixture_vectors()
    with pytest.raises(ValueError, match="hub_weight"):
        GalleryPolicy(g, [str(i) for i in range(len(g))], hub_weight=weight)


@pytest.mark.parametrize("top_k", [0, -1, 13, 1.5, True])
def test_invalid_top_k(top_k):
    q, g = fixture_vectors()
    policy = GalleryPolicy(g, [str(i) for i in range(len(g))])
    with pytest.raises(ValueError, match="top_k"):
        policy.search(q, top_k=top_k)


def test_shapes_ids_and_finite_values_are_checked():
    q, g = fixture_vectors()
    ids = [str(i) for i in range(len(g))]
    for bad in (g[0], g[:, :0], np.full_like(g, np.nan), np.full_like(g, np.inf)):
        with pytest.raises(ValueError):
            GalleryPolicy(bad, ids)
    for bad_ids in (ids[:-1], ["same"] * len(g), [""] + ids[1:], [None] + ids[1:], "ids"):
        with pytest.raises(ValueError):
            GalleryPolicy(g, bad_ids)
    policy = GalleryPolicy(g, ids)
    for bad in (q[0], q[:, :-1], np.full_like(q, np.nan), np.full_like(q, np.inf)):
        for method in (policy.search, policy.ranking_scores):
            with pytest.raises(ValueError):
                method(bad)
    for bad in (ids[:2], ["missing"] * len(q), [["0"]] * len(q), "0"):
        with pytest.raises(ValueError, match="selected_ids"):
            policy.confidence_for(q, bad)


def test_empty_queries_and_nondefault_parameters():
    q, g = fixture_vectors()
    policy = GalleryPolicy(g, [str(i) for i in range(len(g))], hub_k=2, hub_weight=0.3)
    expected, _, _ = research_formula(q, g, k=2, weight=0.3)
    np.testing.assert_allclose(policy.ranking_scores(q), expected, atol=1e-6)
    empty = q[:0]
    result = policy.search(empty)
    assert result.ranking == result.top_ids == []
    assert result.top_scores.shape == (0,)
    assert policy.ranking_scores(empty).shape == (0, len(g))
    assert policy.confidence_for(empty, []).shape == (0,)
