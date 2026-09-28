"""D-048: lower-ranked fallbacks must not change official top-1 decisions."""
import csv

import numpy as np
import pytest

from reid.eval.eval_local import FoldTruth, candidate_metrics
from reid.infer.run import InputRow, write_outputs
from reid.post.search import SearchResult


def _write(root, scores, ranks):
    queries = [InputRow(q, root / f"{q}.jpg", (0, 0, 1, 1)) for q in scores]
    gallery = [InputRow(g, root / f"{g}.jpg", (0, 0, 1, 1)) for g in ("twin", "positive", "wrong")]
    write_outputs(
        root, queries, gallery, np.ones((len(queries), 2)), np.ones((3, 2)),
        0.583, SearchResult(
            ranking=[ranks[q] for q in scores], top_ids=[ranks[q][0] for q in scores],
            top_scores=np.array(list(scores.values()), dtype=np.float64),
        ),
    )
    with (root / "candidates.csv").open() as handle:
        return [(r["query_id"], r["gallery_id"], float(r["confidence"])) for r in csv.DictReader(handle)]


@pytest.mark.parametrize("primary", ["twin", "positive", "wrong"])
def test_top3_preserves_official_metrics_and_refusals(tmp_path, primary):
    order = [primary] + [g for g in ("twin", "positive", "wrong") if g != primary]
    candidates = _write(tmp_path, {"q": 0.583, "open": 0.5829}, {"q": order, "open": order})
    assert len(candidates) == 3
    assert [g for _, g, _ in candidates] == order
    assert candidates[0][2] == 0.583
    assert candidates[0][2] > candidates[1][2] > candidates[2][2]
    truth = FoldTruth(
        query_ids=["q", "open"], gallery_ids=order,
        vid={"q": 1, "open": 9, "twin": 1, "positive": 1, "wrong": 2},
        cam={"q": 1, "open": 1, "twin": 1, "positive": 2, "wrong": 1},
    )
    # F1/TNR/PR-AUC and all confusion counters remain exactly as with one row.
    assert candidate_metrics(truth, candidates, "official") == candidate_metrics(truth, candidates[:1], "official")


def test_candidate_rows_are_independent_of_other_queries_and_order(tmp_path):
    ranks = {"q": ["twin", "positive", "wrong"], "other": ["wrong", "positive", "twin"]}
    alone = _write(tmp_path / "alone", {"q": 0.75}, ranks)
    first = _write(tmp_path / "first", {"q": 0.75, "other": 0.9}, ranks)
    last = _write(tmp_path / "last", {"other": 0.9, "q": 0.75}, ranks)
    assert alone == [r for r in first if r[0] == "q"] == [r for r in last if r[0] == "q"]
