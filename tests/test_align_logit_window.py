from pathlib import Path

import numpy as np
import pytest

from scripts.analysis.align_logit_window import align_logits


def write(path: Path, candidates, *, logits=None):
    values = {
        "fold": np.array(0),
        "query_ids": np.array(["q0", "q1"]),
        "candidates": np.array(candidates),
    }
    if logits is not None:
        values["logits"] = np.asarray(logits, dtype=np.float32)
    np.savez(path, **values)


def test_aligns_same_pairs_to_target_order(tmp_path):
    rows = [[f"g{i}" for i in range(10)], [f"h{i}" for i in range(10)]]
    logits = np.array([range(10), range(10, 20)], dtype=np.float32)
    source, target = tmp_path / "source.npz", tmp_path / "target.npz"
    write(source, rows, logits=logits)
    write(target, [rows[0][::-1], rows[1]])
    fold, qids, candidates, aligned, changed = align_logits(source, target)
    assert fold == 0 and qids.tolist() == ["q0", "q1"] and changed == 1
    assert candidates[0].tolist() == rows[0][::-1]
    assert aligned[0].tolist() == list(range(9, -1, -1))
    assert aligned[1].tolist() == list(range(10, 20))


def test_rejects_changed_candidate_set(tmp_path):
    rows = [[f"g{i}" for i in range(10)], [f"h{i}" for i in range(10)]]
    source, target = tmp_path / "source.npz", tmp_path / "target.npz"
    write(source, rows, logits=np.zeros((2, 10), dtype=np.float32))
    rows[0][-1] = "different"
    write(target, rows)
    with pytest.raises(ValueError, match="rescore"):
        align_logits(source, target)
