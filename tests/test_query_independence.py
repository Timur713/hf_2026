from pathlib import Path

import numpy as np
import pytest

from reid.infer.run import InputRow
from scripts.check_query_independence import compare, execute


def row(iid):
    return InputRow(iid, Path(iid), (0, 0, 10, 10))


class FakeExtractor:
    def __init__(self, dependent=False):
        self.cfg = {"candidate_policy": "gallery_z", "candidate_threshold": 0.01,
                    "rerank_mode": "standardized", "rerank_weight": 0.15}
        self.dependent, self.calls = dependent, []
        rng = np.random.default_rng(2026)
        vectors = rng.normal(size=(12, 32)).astype(np.float32)
        self.gallery = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def extract_batch(self, items):
        ids = [str(item[0]) for item in items]
        self.calls.append(ids)
        output = []
        for iid in ids:
            index = int(iid[1:])
            if self.dependent and iid.startswith("q") and len(items) > 1:
                index += 1
            output.append(self.gallery[index % 12])
        return np.stack(output)


class FakeVerifier:
    def __init__(self):
        self.calls = 0

    def score_pairs(self, query, candidates):
        self.calls += 1
        assert len(candidates) == 10
        return np.zeros(10, dtype=np.float32)


def test_actual_pipeline_alignment_and_fixed_gallery(tmp_path):
    extractor, verifier = FakeExtractor(), FakeVerifier()
    fixed, extras, gallery = [row(f"q{i}") for i in range(3)], [row("q8"), row("q9")], [row(f"g{i}") for i in range(12)]
    report = execute(extractor, verifier, fixed, extras, gallery, tmp_path / "pass")
    assert report["passed"]
    assert verifier.calls == 3 + 3 + 3 + 5
    assert sum(any(iid.startswith("g") for iid in batch) for batch in extractor.calls) == 1
    assert report["variants"]["reverse"]["query_ids"] == ["q2", "q1", "q0"]
    assert report["variants"]["with_extra_queries"]["query_ids"] == ["q8", "q0", "q9", "q1", "q2"]
    assert all(v["c6"]["validated"] for v in report["variants"].values())
    with pytest.raises(FileExistsError):
        execute(extractor, verifier, fixed, extras, gallery, tmp_path / "pass")


def test_batch_dependent_features_fail_real_ranking_comparison(tmp_path):
    report = execute(FakeExtractor(dependent=True), FakeVerifier(),
                     [row(f"q{i}") for i in range(3)], [row("q8")],
                     [row(f"g{i}") for i in range(12)], tmp_path / "fail")
    assert not report["passed"]
    result = report["variants"]["batch32"]["comparison"]
    assert result["changed_top10_query_ids"]
    assert result["max_abs_embedding_drift"] > 0


def test_embedding_drift_is_diagnostic_but_confidence_change_fails():
    a = {"q": {"ranking": ("g0",), "candidates": [("g0", "0.9")], "embedding": np.array([1., 0.])}}
    b = {"extra": {}, "q": {**a["q"], "embedding": np.array([1., 1e-6])}}
    assert compare(a, b)["passed"]
    b["q"]["candidates"] = [("g0", "0.9000000001")]
    result = compare(a, b)
    assert not result["passed"]
    assert result["changed_candidate_rows_query_ids"] == ["q"]
    assert result["changed_candidate_ids_or_acceptance_query_ids"] == []
    assert result["max_abs_confidence_drift_for_same_candidates"] > 0
    assert not result["strict_byte_equivalence"]
    assert result["semantic_ranked_ids_equal"] and result["semantic_candidate_ids_equal"]
    assert result["semantic_acceptance_equal"]
    with pytest.raises(ValueError, match="missing baseline"):
        compare(a, {})


def test_disabled_or_implicit_verifier_policy_rejected(tmp_path):
    extractor = FakeExtractor()
    extractor.cfg["rerank_weight"] = 0
    with pytest.raises(ValueError, match="enabled verifier"):
        execute(extractor, FakeVerifier(), [row("q0")], [row("q1")], [row(f"g{i}") for i in range(12)], tmp_path / "out")
    assert not (tmp_path / "out").exists()
