import numpy as np
import pytest

from reid.infer.run import InputRow, write_outputs
from reid.post.policy import LEGACY_MODEL_V1_NAME, candidate_threshold_for
from reid.post.search import SearchResult


def test_threshold_precedence_and_legacy_release():
    assert candidate_threshold_for({"name": "final_all_data"}) == 0.583
    assert candidate_threshold_for({"name": LEGACY_MODEL_V1_NAME}) == 0.600
    cfg = {"name": LEGACY_MODEL_V1_NAME, "candidate_threshold": 0.7}
    assert candidate_threshold_for(cfg) == 0.7
    assert candidate_threshold_for(cfg, 0.8) == 0.8
    with pytest.raises(ValueError, match="finite"):
        candidate_threshold_for({}, float("nan"))


def test_final_threshold_accepts_boundary_and_rejects_below(tmp_path):
    rows = [InputRow(f"q{i}", tmp_path / f"q{i}.jpg", (0, 0, 1, 1)) for i in range(3)]
    gallery = [InputRow("g0", tmp_path / "g0.jpg", (0, 0, 1, 1))]
    result = SearchResult(
        ranking=[["g0"] for _ in rows],
        top_ids=["g0"] * 3,
        top_scores=np.array([0.5829, 0.583, 0.59], dtype=np.float64),
    )
    write_outputs(
        tmp_path, rows, gallery, np.ones((3, 2)), np.ones((1, 2)),
        candidate_threshold_for({"name": "final_all_data"}), result,
    )
    lines = (tmp_path / "candidates.csv").read_text().splitlines()
    assert lines == ["query_id,gallery_id,confidence", "q1,g0,0.58300000", "q2,g0,0.59000000"]
