import csv
from pathlib import Path

import numpy as np

from reid.eval.validate_outputs import validate_outputs
from reid.infer.run import DEFAULT_CANDIDATE_THRESHOLD, run_pipeline
from reid.infer.run import CANDIDATES_PER_QUERY
from reid.post.search import cosine_search
from reid.post.verifier import rerank_topk


class FakeExtractor:
    def __init__(self, vectors):
        self.vectors = vectors
        self.cfg = {}

    def extract_batch(self, items):
        return np.stack([self.vectors[Path(path).stem] for path, _ in items]).astype(np.float32)


class FakeVerifier:
    def score_pairs(self, query, candidates):
        del query
        return np.array(
            [8.0 if Path(path).stem == "g1" else -8.0 for path, _ in candidates],
            dtype=np.float32,
        )


def _write_input(path, ids):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "x", "y", "w", "h"])
        for image_id in ids:
            writer.writerow([image_id, 0, 0, 10, 10])


def test_cosine_search_is_stable_on_ties():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    g = np.tile(q, (10, 1))
    result = cosine_search(q, g, [f"g{i}" for i in range(10)])
    assert result.ranking == [[f"g{i}" for i in range(10)]]


def test_pipeline_writes_valid_outputs_and_applies_threshold(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    query_ids = ["q0", "q1"]
    gallery_ids = [f"g{i}" for i in range(10)]
    for image_id in query_ids + gallery_ids:
        (images / f"{image_id}.jpg").touch()
    query_csv, gallery_csv = tmp_path / "query.csv", tmp_path / "gallery.csv"
    _write_input(query_csv, query_ids)
    _write_input(gallery_csv, gallery_ids)

    vectors = {f"g{i}": np.array([1.0, float(i + 1), 0.0]) for i in range(10)}
    vectors["q0"] = vectors["g0"].copy()
    vectors["q1"] = np.array([-1.0, 0.0, 0.0])
    out = tmp_path / "out"
    run_pipeline(
        images,
        query_csv,
        gallery_csv,
        out,
        tmp_path / "unused-model",
        "cpu",
        3,
        0.5,
        extractor=FakeExtractor(vectors),
    )

    validate_outputs(
        query_csv,
        gallery_csv,
        out / "submission.csv",
        out / "embeddings.npy",
        out / "candidates.csv",
    )
    submission = list(csv.reader((out / "submission.csv").open()))
    assert submission[0][0] == "q0"
    assert submission[0][1] == "g0"
    candidates = list(csv.reader((out / "candidates.csv").open()))
    assert candidates[0] == ["query_id", "gallery_id", "confidence"]
    # D-048: на отвеченный запрос пишем несколько кандидатов, отказ — отсутствие строк
    answered = [row[0] for row in candidates[1:]]
    assert set(answered) == {"q0"} and 1 <= len(answered) <= CANDIDATES_PER_QUERY
    confidences = [float(row[2]) for row in candidates[1:]]
    assert confidences == sorted(confidences, reverse=True)
    embeddings = np.load(out / "embeddings.npy")
    assert embeddings.shape == (12, 3)
    assert embeddings.dtype == np.float32

    reranked_out = tmp_path / "reranked-out"
    run_pipeline(
        images,
        query_csv,
        gallery_csv,
        reranked_out,
        tmp_path / "unused-model",
        "cpu",
        3,
        0.5,
        extractor=FakeExtractor(vectors),
        verifier=FakeVerifier(),
    )
    reranked_submission = list(csv.reader((reranked_out / "submission.csv").open()))
    assert reranked_submission[0][1] == "g1"


def test_pipeline_uses_calibrated_default_threshold(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    query_ids = ["q0"]
    gallery_ids = [f"g{i}" for i in range(10)]
    for image_id in query_ids + gallery_ids:
        (images / f"{image_id}.jpg").touch()
    query_csv, gallery_csv = tmp_path / "query.csv", tmp_path / "gallery.csv"
    _write_input(query_csv, query_ids)
    _write_input(gallery_csv, gallery_ids)

    vectors = {f"g{i}": np.array([1.0, float(i + 1), 0.0]) for i in range(10)}
    vectors["q0"] = np.array([1.0, 2.0, 0.0])
    out = tmp_path / "out"
    run_pipeline(
        images,
        query_csv,
        gallery_csv,
        out,
        tmp_path / "unused-model",
        "cpu",
        3,
        None,
        extractor=FakeExtractor(vectors),
    )

    candidates = list(csv.reader((out / "candidates.csv").open()))
    assert DEFAULT_CANDIDATE_THRESHOLD == 0.583
    # D-048: на отвеченный запрос пишем несколько кандидатов, отказ — отсутствие строк
    answered = [row[0] for row in candidates[1:]]
    assert set(answered) == {"q0"} and 1 <= len(answered) <= CANDIDATES_PER_QUERY
    confidences = [float(row[2]) for row in candidates[1:]]
    assert confidences == sorted(confidences, reverse=True)


def test_verifier_reranking_is_independent_per_query():
    query_items = [(Path("q0.jpg"), (0.0, 0.0, 1.0, 1.0))]
    gallery_items = [(Path(f"g{i}.jpg"), (0.0, 0.0, 1.0, 1.0)) for i in range(10)]
    gallery_ids = [f"g{i}" for i in range(10)]
    query_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    gallery_emb = np.array(
        [[1.0, 0.0], [0.995, 0.0999], *([[-1.0, 0.0]] * 8)],
        dtype=np.float32,
    )
    gallery_emb /= np.linalg.norm(gallery_emb, axis=1, keepdims=True)
    base = cosine_search(query_emb, gallery_emb, gallery_ids)
    first = rerank_topk(
        query_items,
        gallery_items,
        query_emb,
        gallery_emb,
        gallery_ids,
        base,
        FakeVerifier(),
    )
    assert first.ranking[0][:2] == ["g1", "g0"]
    assert first.top_ids == ["g1"]
    assert np.isclose(first.top_scores[0], query_emb[0] @ gallery_emb[1])

    # Adding another query cannot change q0: no cross-query state enters reranking.
    query_items_2 = [*query_items, (Path("q1.jpg"), (0.0, 0.0, 1.0, 1.0))]
    query_emb_2 = np.concatenate([query_emb, np.array([[0.0, 1.0]], dtype=np.float32)])
    base_2 = cosine_search(query_emb_2, gallery_emb, gallery_ids)
    together = rerank_topk(
        query_items_2,
        gallery_items,
        query_emb_2,
        gallery_emb,
        gallery_ids,
        base_2,
        FakeVerifier(),
    )
    assert together.ranking[0] == first.ranking[0]
    assert together.top_scores[0] == first.top_scores[0]
