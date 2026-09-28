from pathlib import Path
import json

import numpy as np
import pytest
import torch

from reid.infer.run import InputRow
from scripts import check_query_content_isolation as check
from scripts.check_query_content_isolation import compare_anchors, execute, main


def row(iid):
    return InputRow(iid, Path(iid), (0, 0, 10, 10))


class FakeExtractor:
    def __init__(self, leaky=False):
        self.cfg = {"candidate_policy": "gallery_z", "candidate_threshold": 0.01,
                    "rerank_mode": "standardized", "rerank_weight": 0.15}
        self.leaky, self.calls = leaky, []
        vectors = np.random.default_rng(2026).normal(size=(12, 32)).astype(np.float32)
        self.gallery = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def extract_batch(self, items):
        ids = [str(item[0]) for item in items]
        self.calls.append(ids)
        offset = sum(int(iid[1:]) for iid in ids) % 11 if self.leaky else 0
        return np.stack([self.gallery[(int(iid[1:]) + (offset if iid.startswith("q") else 0)) % 12]
                         for iid in ids])


class FakeVerifier:
    def __init__(self):
        self.calls = 0

    def score_pairs(self, query, candidates):
        self.calls += 1
        assert len(candidates) == 10
        return np.zeros(10, dtype=np.float32)


@pytest.fixture
def groups():
    query = [row(f"q{i}") for i in range(56)]
    return [query[:8], query[8:32], query[32:], [row(f"g{i}") for i in range(12)]]


def test_fixed_shape_and_fixed_gallery_safe_pipeline_passes(groups, tmp_path):
    extractor, verifier = FakeExtractor(), FakeVerifier()
    result = execute(extractor, verifier, *groups, tmp_path / "out")
    assert result["passed"]
    assert result["fixed_shape_content_isolation_passed"]
    assert result["fixed_shape_content_and_permutation_passed"]
    assert result["query_batch_count"] == 3
    assert [len(batch) for batch in extractor.calls] == [12, 32, 32, 32]
    assert sum(any(iid.startswith("g") for iid in batch) for batch in extractor.calls) == 1
    assert verifier.calls == 96
    assert result["gallery_embedding_shape"] == [12, 32]
    assert len(result["gallery_embedding_raw_bytes_sha256"]) == 64
    assert result["variants"]["baseline"]["anchor_positions"] == list(range(8))
    assert result["variants"]["replaced_neighbours"]["anchor_positions"] == list(range(8))
    assert result["variants"]["reversed_replacement"]["anchor_positions"] == list(range(31, 23, -1))
    for variant in result["variants"].values():
        assert variant["c6"]["validated"]
        assert variant["comparison"]["compared_queries"] == 8
        assert variant["comparison"]["anchor_embeddings_bitwise_equal"]
    with pytest.raises(FileExistsError):
        execute(extractor, verifier, *groups, tmp_path / "out")


def test_other_query_content_leak_is_detected_at_same_shape(groups, tmp_path):
    extractor = FakeExtractor(leaky=True)
    result = execute(extractor, FakeVerifier(), *groups, tmp_path / "out")
    assert not result["passed"]
    assert not result["fixed_shape_content_isolation_passed"]
    comparison = result["variants"]["replaced_neighbours"]["comparison"]
    assert comparison["changed_embedding_query_ids"]
    assert comparison["changed_top10_query_ids"]
    assert comparison["max_abs_embedding_drift"] > 0
    assert [len(batch) for batch in extractor.calls[1:]] == [32, 32, 32]


@pytest.mark.parametrize("problem", ["anchor-count", "neighbour-count", "replacement-count", "anchor-duplicate",
                                      "neighbour-overlap", "replacement-overlap", "gallery-overlap", "short-gallery"])
def test_groups_are_unique_disjoint_and_bounded_before_work(groups, tmp_path, problem):
    if problem == "anchor-count":
        groups[0] = groups[0][:-1]
    elif problem == "neighbour-count":
        groups[1] = groups[1][:-1]
    elif problem == "replacement-count":
        groups[2] = groups[2][:-1]
    elif problem == "anchor-duplicate":
        groups[0][-1] = groups[0][0]
    elif problem == "neighbour-overlap":
        groups[1][0] = groups[0][0]
    elif problem == "replacement-overlap":
        groups[2][0] = groups[1][0]
    elif problem == "gallery-overlap":
        groups[3][0] = groups[0][0]
    else:
        groups[3] = groups[3][:9]
    extractor = FakeExtractor()
    with pytest.raises(ValueError):
        execute(extractor, FakeVerifier(), *groups, tmp_path / "out")
    assert not extractor.calls
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["embedding", "signed-zero", "confidence", "ranking", "acceptance"])
def test_exact_anchor_comparison_hides_no_drift(mutation):
    first = {"q": {"ranking": ("g0", "g1"), "candidates": [("g0", "1.00000000")],
                   "embedding": np.array([1.0, 0.0], dtype=np.float32)}}
    second = {"q": {**first["q"], "embedding": first["q"]["embedding"].copy()}}
    if mutation == "embedding":
        second["q"]["embedding"][1] = 1e-8
    elif mutation == "signed-zero":
        second["q"]["embedding"][1] = -0.0
    elif mutation == "confidence":
        second["q"]["candidates"] = [("g0", "1.00000001")]
    elif mutation == "ranking":
        second["q"]["ranking"] = ("g1", "g0")
    else:
        second["q"]["candidates"] = []
    assert not compare_anchors(first, second)["passed"]


def test_cli_has_no_query_csv_or_test_input_argument(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--model", "model", "--verifier", "verifier.pt", "--out", str(tmp_path / "out"),
              "--query", "test_query.csv"])
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_missing_verifier_or_disabled_policy_fails_before_work(groups, tmp_path):
    extractor = FakeExtractor()
    with pytest.raises(ValueError, match="enabled verifier"):
        execute(extractor, None, *groups, tmp_path / "out")
    extractor.cfg["rerank_weight"] = 0
    with pytest.raises(ValueError, match="enabled verifier"):
        execute(extractor, FakeVerifier(), *groups, tmp_path / "out")
    assert not extractor.calls


def test_cli_uses_fixed_train_prefix_and_records_cpu_not_gpu(groups, tmp_path, monkeypatch):
    query = [*groups[0], *groups[1], *groups[2], row("q56")]
    def fixed_train(data, splits, fold):
        assert data == Path("train-data") and splits == Path("folds") and fold == 1
        return query, groups[3]
    monkeypatch.setattr(check, "train_rows", fixed_train)
    source = [{"path": "verifier.pt", "sha256": "a" * 64, "bytes": 100}]
    monkeypatch.setattr(check, "source_records", lambda *a: source)
    monkeypatch.setattr(check, "configure_determinism", lambda seed: None)
    monkeypatch.setattr(check, "git_commit", lambda: "b" * 40)
    monkeypatch.setattr(check, "file_record", lambda path: {"sha256": "c" * 64, "bytes": 1})
    extractor, verifier = FakeExtractor(), FakeVerifier()
    extractor.device, extractor._preprocess_pool = torch.device("cpu"), None
    verifier._pool = None
    monkeypatch.setattr(check, "Extractor", lambda *a: extractor)
    monkeypatch.setattr(check, "PairVerifier", lambda *a: verifier)
    output = tmp_path / "out"
    assert main(["--model", "model", "--verifier", "verifier.pt", "--out", str(output),
                 "--device", "cpu", "--data-root", "train-data", "--splits", "folds", "--fold", "1"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["sources"] == source and report["code_commit"] == "b" * 40
    assert report["seed"] == 2026 and report["fold"] == 1
    assert report["device"] == "cpu" and report["gpu_name"] is None
    assert report["gpu_check_executed"] is False
    assert not any("q56" in batch for batch in extractor.calls)
    assert len(report["inputs"]) == 4


def test_cli_rejects_insufficient_train_prefix_before_models(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "train_rows", lambda *a: ([row(f"q{i}") for i in range(55)], []))
    monkeypatch.setattr(check, "source_records", lambda *a: pytest.fail("must reject before loading sources"))
    output = tmp_path / "out"
    assert main(["--model", "model", "--verifier", "verifier.pt", "--out", str(output)]) == 2
    assert not output.exists()
