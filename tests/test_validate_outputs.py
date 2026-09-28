import csv
from pathlib import Path

import numpy as np
import pytest

from reid.eval.validate_outputs import (
    CANDIDATES_HEADER,
    ValidationError,
    main,
    validate_outputs,
)


def _write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


@pytest.fixture
def valid_files(tmp_path):
    query_ids = ["q0", "q1"]
    gallery_ids = ["g{}".format(index) for index in range(12)]
    _write_csv(
        tmp_path / "query.csv",
        [["image_id", "x", "y", "w", "h"]]
        + [[query_id, 0, 0, 1, 1] for query_id in query_ids],
    )
    _write_csv(
        tmp_path / "gallery.csv",
        [["image_id", "x", "y", "w", "h"]]
        + [[gallery_id, 0, 0, 1, 1] for gallery_id in gallery_ids],
    )
    _write_csv(
        tmp_path / "submission.csv",
        [[query_id] + gallery_ids[:10] for query_id in query_ids],
    )
    _write_csv(
        tmp_path / "candidates.csv",
        [CANDIDATES_HEADER, ["q0", "g0", "0.9"]],
    )
    np.save(
        tmp_path / "embeddings.npy",
        np.zeros((len(query_ids) + len(gallery_ids), 8), dtype=np.float32),
    )
    return {
        "query": tmp_path / "query.csv",
        "gallery": tmp_path / "gallery.csv",
        "submission": tmp_path / "submission.csv",
        "embeddings": tmp_path / "embeddings.npy",
        "candidates": tmp_path / "candidates.csv",
    }


def _validate(paths):
    validate_outputs(
        paths["query"],
        paths["gallery"],
        paths["submission"],
        paths["embeddings"],
        paths["candidates"],
    )


def test_valid_outputs_and_missing_candidate_rows_are_accepted(valid_files):
    _validate(valid_files)


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [["q1"] + ["g{}".format(i) for i in range(10)]],
            "expected 2 rows",
        ),
        (
            [
                ["q1"] + ["g{}".format(i) for i in range(10)],
                ["q0"] + ["g{}".format(i) for i in range(10)],
            ],
            "expected query_id",
        ),
        (
            [
                ["q0"] + ["g0"] * 10,
                ["q1"] + ["g{}".format(i) for i in range(10)],
            ],
            "must be distinct",
        ),
        (
            [
                ["q0"] + ["g{}".format(i) for i in range(9)] + ["missing"],
                ["q1"] + ["g{}".format(i) for i in range(10)],
            ],
            "absent from input gallery",
        ),
    ],
)
def test_submission_violations_fail(valid_files, rows, message):
    _write_csv(valid_files["submission"], rows)
    with pytest.raises(ValidationError, match=message):
        _validate(valid_files)


def test_obsolete_submission_header_is_rejected(valid_files):
    obsolete_header = ["query_id"] + [
        "gallery_id_{}".format(index) for index in range(1, 11)
    ]
    valid_rows = [
        [query_id] + ["g{}".format(index) for index in range(10)]
        for query_id in ("q0", "q1")
    ]
    _write_csv(valid_files["submission"], [obsolete_header] + valid_rows)
    with pytest.raises(ValidationError, match="expected 2 rows"):
        _validate(valid_files)


@pytest.mark.parametrize(
    "array, message",
    [
        (np.zeros((14, 8), dtype=np.float64), "expected dtype float32"),
        (np.zeros((13, 8), dtype=np.float32), "expected 14 rows"),
        (np.zeros((14,), dtype=np.float32), "expected a 2D array"),
        (np.full((14, 8), np.nan, dtype=np.float32), "contain NaN or infinity"),
    ],
)
def test_embedding_violations_fail(valid_files, array, message):
    np.save(valid_files["embeddings"], array)
    with pytest.raises(ValidationError, match=message):
        _validate(valid_files)


@pytest.mark.parametrize(
    "row, message",
    [
        (["missing", "g0", "0.5"], "unknown query_id"),
        (["q0", "missing", "0.5"], "unknown gallery_id"),
        (["q0", "g0", "nan"], "confidence must be finite"),
        (["q0", "g0", "not-a-number"], "invalid confidence"),
    ],
)
def test_candidate_violations_fail(valid_files, row, message):
    _write_csv(valid_files["candidates"], [CANDIDATES_HEADER, row])
    with pytest.raises(ValidationError, match=message):
        _validate(valid_files)


def test_cli_returns_nonzero_for_invalid_output(valid_files, capsys):
    _write_csv(valid_files["candidates"], [["bad", "header"], ["q0", "g0"]])
    status = main(
        [
            "--query",
            str(valid_files["query"]),
            "--gallery",
            str(valid_files["gallery"]),
            "--submission",
            str(valid_files["submission"]),
            "--embeddings",
            str(valid_files["embeddings"]),
            "--candidates",
            str(valid_files["candidates"]),
        ]
    )
    assert status != 0
    assert "INVALID" in capsys.readouterr().err


def test_cli_self_test_passes(capsys):
    assert main(["--self-test"]) == 0
    assert "VALID" in capsys.readouterr().out
