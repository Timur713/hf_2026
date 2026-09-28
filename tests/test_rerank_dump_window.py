from pathlib import Path

import numpy as np
import pytest

from scripts.analysis.rerank_dump import load_candidate_window


def write_window(path: Path, *, fold=0, n=3):
    candidates = np.array([[f"g{i}_{j}" for j in range(10)] for i in range(n)])
    np.savez(
        path,
        fold=np.array(fold),
        query_ids=np.array([f"q{i}" for i in range(n)]),
        candidates=candidates,
        sim=np.zeros((n, 10), dtype=np.float32),
        emb=np.array(["a", "b", "c"]),
    )


def test_load_exact_candidate_window(tmp_path):
    path = tmp_path / "window.npz"
    write_window(path)
    qids, candidates, sim, emb = load_candidate_window(str(path), 0)
    assert qids == ["q0", "q1", "q2"]
    assert candidates.shape == (3, 10)
    assert sim.dtype == np.float32
    assert emb.tolist() == ["a", "b", "c"]


@pytest.mark.parametrize("corruption", ["fold", "shape", "dtype", "duplicate"])
def test_candidate_window_rejects_corruption(tmp_path, corruption):
    path = tmp_path / "window.npz"
    write_window(path, fold=1 if corruption == "fold" else 0)
    if corruption != "fold":
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        if corruption == "shape":
            values["candidates"] = values["candidates"][:, :9]
            values["sim"] = values["sim"][:, :9]
        elif corruption == "dtype":
            values["sim"] = values["sim"].astype(np.float64)
        else:
            values["candidates"][0, 1] = values["candidates"][0, 0]
        np.savez(path, **values)
    with pytest.raises(ValueError):
        load_candidate_window(str(path), 0)
