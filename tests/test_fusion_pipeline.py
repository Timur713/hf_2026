import csv
from pathlib import Path

import numpy as np
import pytest

from reid.infer.run import InputRow, run_pipeline, write_outputs
from reid.post.search import SearchResult
from reid.post.policy import candidate_threshold_for
from reid.post.gallery_policy import GalleryPolicy


class ArrayExtractor:
    def __init__(self, vectors):
        self.cfg = {"candidate_policy": "gallery_z", "rerank_mode": "standardized"}
        self.vectors = vectors

    def extract_batch(self, items):
        return np.stack([self.vectors[Path(path).stem] for path, _ in items])


class ToyVerifier:
    def score_pairs(self, query, candidates):
        return np.array([float(Path(path).stem[1:]) for path, _ in candidates], dtype=np.float32)


def csv_input(root, name, ids):
    path = root / name
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "x", "y", "w", "h"])
        writer.writerows((i, 0, 0, 10, 10) for i in ids)
    for image_id in ids:
        (root / f"{image_id}.jpg").touch()
    return path


def rows_by_query(path):
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    return {row[0]: [r for r in rows if r[0] == row[0]] for row in rows}


@pytest.mark.parametrize("with_verifier", [False, True])
def test_gallery_pipeline_matches_formula_and_is_query_independent(tmp_path, with_verifier):
    rng = np.random.default_rng(2026)
    gallery = rng.normal(size=(14, 256)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    gallery_ids = [f"g{i}" for i in range(len(gallery))]
    vectors = dict(zip(gallery_ids, gallery))
    vectors.update(q0=gallery[0], q1=gallery[1], q2=gallery[2])
    extractor = ArrayExtractor(vectors)
    q = csv_input(tmp_path, "query.csv", ["q0", "q1", "q2"])
    g = csv_input(tmp_path, "gallery.csv", gallery_ids)
    verifier = ToyVerifier() if with_verifier else None
    out = tmp_path / "all"
    run_pipeline(tmp_path, q, g, out, tmp_path, "cpu", 2, None,
                 extractor=extractor, verifier=verifier)
    q_one = csv_input(tmp_path, "single.csv", ["q1"])
    single = tmp_path / "single"
    run_pipeline(tmp_path, q_one, g, single, tmp_path, "cpu", 1, None,
                 extractor=extractor, verifier=verifier)
    for filename in ("submission.csv", "candidates.csv"):
        assert rows_by_query(out / filename).get("q1") == rows_by_query(single / filename).get("q1")
    q_rev = csv_input(tmp_path, "reversed.csv", ["q2", "q1", "q0"])
    rev = tmp_path / "rev"
    run_pipeline(tmp_path, q_rev, g, rev, tmp_path, "cpu", 3, None,
                 extractor=extractor, verifier=verifier)
    for filename in ("submission.csv", "candidates.csv"):
        assert rows_by_query(out / filename) == rows_by_query(rev / filename)

    emb = np.load(out / "embeddings.npy", allow_pickle=False)
    assert emb.shape == (17, 256) and emb.dtype == np.float32
    policy = GalleryPolicy(emb[3:], gallery_ids)
    with (out / "submission.csv").open() as f:
        chosen = [row[1] for row in csv.reader(f)]
    expected = policy.confidence_for(emb[:3], chosen)
    with (out / "candidates.csv").open() as f:
        candidates = list(csv.DictReader(f))
    for index, qid in enumerate(("q0", "q1", "q2")):
        answered = [row for row in candidates if row["query_id"] == qid]
        assert bool(answered) == (expected[index] >= 6.0)
        if answered:
            assert answered[0]["gallery_id"] == chosen[index]
            assert float(answered[0]["confidence"]) == pytest.approx(expected[index], abs=1e-6)


def test_threshold_keeps_policies_in_separate_units():
    assert candidate_threshold_for({"candidate_policy": "gallery_z"}) == 6.0
    assert candidate_threshold_for({"name": "vitb14_f0_clsavg_in280_ep100"}) == 0.6
    assert candidate_threshold_for({}) == 0.583
    assert candidate_threshold_for({"candidate_policy": "gallery_z", "candidate_threshold": 5.9}) == 5.9
    assert candidate_threshold_for({"candidate_policy": "gallery_z"}, 6.1) == 6.1
    with pytest.raises(ValueError, match="unknown candidate policy"):
        candidate_threshold_for({"candidate_policy": "typo"})


def test_large_gallery_z_keeps_candidate_order_after_serialization(tmp_path):
    q = [InputRow("q", tmp_path / "q.jpg", (0, 0, 1, 1))]
    g = [InputRow(f"g{i}", tmp_path / f"g{i}.jpg", (0, 0, 1, 1)) for i in range(10)]
    result = SearchResult(ranking=[[row.image_id for row in g]],
                          top_ids=["g0"], top_scores=np.array([1e12], dtype=np.float32))
    write_outputs(tmp_path, q, g, np.ones((1, 2)), np.ones((10, 2)), 6, result)
    with (tmp_path / "candidates.csv").open() as f:
        scores = [float(row["confidence"]) for row in csv.DictReader(f)]
    assert scores[0] > scores[1] > scores[2]
