from pathlib import Path

import numpy as np
import pytest

from reid.infer.run import InputRow, rank_embeddings
from reid.post.pipeline import PreparedGallery
from scripts.bench_pipeline import profile_requests, summary_ms
from tests.test_infer_pipeline import FakeExtractor, FakeVerifier


@pytest.mark.parametrize("policy", ["cosine", "gallery_z"])
@pytest.mark.parametrize("mode", ["raw", "standardized"])
@pytest.mark.parametrize("weight", [0, 0.15])
def test_prepared_gallery_is_identical_and_query_state_is_not_retained(
    policy, mode, weight
):
    rng = np.random.default_rng(123)
    q, g = (
        rng.normal(size=(3, 9)).astype(np.float32),
        rng.normal(size=(16, 9)).astype(np.float32),
    )
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    qr = [InputRow(f"q{i}", Path(f"q{i}.jpg"), (0, 0, 10, 10)) for i in range(len(q))]
    gr = [InputRow(f"g{i}", Path(f"g{i}.jpg"), (0, 0, 10, 10)) for i in range(len(g))]
    cfg = {"candidate_policy": policy, "rerank_mode": mode, "rerank_weight": weight}
    items = [(r.image_path, r.bbox) for r in qr]
    prepared = PreparedGallery(
        g, [r.image_id for r in gr], [(r.image_path, r.bbox) for r in gr], cfg
    )
    scorer = FakeVerifier()
    expected = rank_embeddings(qr, gr, q, g, cfg, scorer)
    actual = prepared.rank(q, items, scorer)
    assert actual.ranking == expected.ranking
    np.testing.assert_array_equal(actual.top_scores, expected.top_scores)
    alone = prepared.rank(q[:1], items[:1], scorer)
    prepared.rank(q[1:2], items[1:2], scorer)
    again = prepared.rank(q[:1], items[:1], scorer)
    assert again.ranking == alone.ranking
    np.testing.assert_array_equal(again.top_scores, alone.top_scores)
    g[:] = 0  # caller mutation must not invalidate prepared gallery statistics
    cfg["rerank_weight"] = 100
    np.testing.assert_array_equal(
        prepared.rank(q[:1], items[:1], scorer).top_scores, alone.top_scores
    )


def test_profile_covers_full_request_without_hidden_batching():
    rng = np.random.default_rng(3)
    vectors = {f"g{i}": rng.normal(size=6).astype(np.float32) for i in range(12)}
    vectors["q0"] = vectors["g0"].copy()
    extractor = FakeExtractor(vectors)
    gallery = np.stack([vectors[f"g{i}"] for i in range(12)])
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    items = [(Path(f"g{i}.jpg"), (0, 0, 10, 10)) for i in range(12)]
    prepared = PreparedGallery(
        gallery, [f"g{i}" for i in range(12)], items, {"candidate_policy": "gallery_z"}
    )
    sizes = []
    extract = extractor.extract_batch

    def counting(items):
        sizes.append(len(items))
        return extract(items)

    extractor.extract_batch = counting
    result = profile_requests(
        extractor,
        prepared,
        FakeVerifier(),
        [InputRow("q0", Path("q0.jpg"), (0, 0, 10, 10))],
        6.6,
        1,
        4,
        lambda: None,
    )
    assert sizes == [1] * 5
    for section in result.values():
        assert section["n"] == 4 and section["p95_ms"] >= section["p50_ms"]
    np.testing.assert_allclose(
        result["full"]["samples_ms"],
        np.array(result["extraction"]["samples_ms"])
        + result["post_total"]["samples_ms"],
    )


def test_summary_uses_explicit_p95_interpolation():
    assert summary_ms([1, 2, 3, 4])["p95_ms"] == pytest.approx(3.85)
    with pytest.raises(ValueError):
        summary_ms([])


def test_runtime_identity_works_without_git_in_offline_image(monkeypatch):
    from scripts import audit_sprint_runtime

    def missing(*args, **kwargs):
        raise FileNotFoundError("git is not installed")

    monkeypatch.setattr(audit_sprint_runtime.subprocess, "check_output", missing)
    identity = audit_sprint_runtime.code_identity()
    assert identity["code_commit"] is None
    assert len(identity["source_files"]["reid/infer/run.py"]["sha256"]) == 64


def test_d082_profile_uses_raw_components_per_request_and_batch32(monkeypatch):
    from reid.infer import d082
    from reid.post.search import SearchResult
    from scripts.bench_pipeline import profile_batch32

    calls = []
    qrow = InputRow("q0", Path("q0.jpg"), (0, 0, 10, 10))

    class ComponentExtractor:
        def extract_batch(self, items):
            raise AssertionError(
                "D082 must never reconstruct components from fused output"
            )

    extractor = ComponentExtractor()

    def components(active, rows, batch_size):
        assert active is extractor
        calls.append((len(rows), batch_size))
        return (np.ones((len(rows), 1), dtype=np.float32),) * 2

    monkeypatch.setattr(d082, "extract_component_rows", components)

    class Prepared:
        def rank(self, blocks, items, scorer):
            assert isinstance(blocks, tuple) and len(blocks) == 2
            assert len(items) == 1
            return SearchResult(
                [[f"g{i}" for i in range(10)]],
                ["g0"],
                np.array([7.0], dtype=np.float32),
            )

    result = profile_requests(
        extractor,
        Prepared(),
        None,
        [qrow],
        6.6,
        1,
        3,
        lambda: None,
        component_mode=True,
    )
    assert calls == [(1, 1)] * 4
    assert result["full"]["n"] == 3
    batch = profile_batch32(extractor, [qrow], 2, lambda: None, component_mode=True)
    assert calls[4:] == [(32, 32)] * 3
    assert batch["n"] == 2 and "raw components" in batch["scope"]
