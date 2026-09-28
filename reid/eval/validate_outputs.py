"""Validate the three submission artifacts defined by contract C6.

The validator intentionally checks only properties recoverable from the files.
In particular, it can verify the embedding row count but cannot prove that the
rows were produced in query-then-gallery order.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


SUBMISSION_COLUMN_COUNT = 11
SUBMISSION_TOP_K = 10
CANDIDATES_HEADER = ["query_id", "gallery_id", "confidence"]
INPUT_ID_COLUMN = "image_id"


class ValidationError(ValueError):
    """Raised when inputs or output artifacts violate contract C6."""


def _read_all_csv_rows(path: Path) -> List[List[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.reader(handle))
    except OSError as exc:
        raise ValidationError("{}: cannot read file: {}".format(path, exc)) from exc


def _read_headered_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    rows = _read_all_csv_rows(path)
    if not rows:
        raise ValidationError("{}: CSV file is empty".format(path))
    return rows[0], rows[1:]


def _read_input_ids(path: Path) -> List[str]:
    header, rows = _read_headered_csv_rows(path)
    if INPUT_ID_COLUMN not in header:
        raise ValidationError(
            "{}: missing required '{}' column".format(path, INPUT_ID_COLUMN)
        )

    id_index = header.index(INPUT_ID_COLUMN)
    ids: List[str] = []
    for line_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValidationError(
                "{}:{}: expected {} columns, got {}".format(
                    path, line_number, len(header), len(row)
                )
            )
        image_id = row[id_index]
        if not image_id:
            raise ValidationError(
                "{}:{}: image_id must not be empty".format(path, line_number)
            )
        ids.append(image_id)

    if len(ids) != len(set(ids)):
        raise ValidationError("{}: image_id values must be unique".format(path))
    return ids


def _require_header(path: Path, actual: Sequence[str], expected: Sequence[str]) -> None:
    if list(actual) != list(expected):
        raise ValidationError(
            "{}: expected header {!r}, got {!r}".format(path, list(expected), list(actual))
        )


def validate_submission(
    path: Path, query_ids: Sequence[str], gallery_ids: Sequence[str]
) -> None:
    # D-025: unlike candidates.csv, submission.csv deliberately has no header.
    rows = _read_all_csv_rows(path)

    if len(rows) != len(query_ids):
        raise ValidationError(
            "{}: expected {} rows (one per query), got {}".format(
                path, len(query_ids), len(rows)
            )
        )

    gallery_set = set(gallery_ids)
    for row_index, (row, expected_query_id) in enumerate(
        zip(rows, query_ids), start=1
    ):
        if len(row) != SUBMISSION_COLUMN_COUNT:
            raise ValidationError(
                "{}:{}: expected 11 columns, got {}".format(
                    path, row_index, len(row)
                )
            )
        query_id, ranked_ids = row[0], row[1:]
        if query_id != expected_query_id:
            raise ValidationError(
                "{}:{}: expected query_id {!r}, got {!r}".format(
                    path, row_index, expected_query_id, query_id
                )
            )
        if any(not gallery_id for gallery_id in ranked_ids):
            raise ValidationError(
                "{}:{}: gallery IDs must not be empty".format(path, row_index)
            )
        if len(set(ranked_ids)) != SUBMISSION_TOP_K:
            raise ValidationError(
                "{}:{}: top-10 gallery IDs must be distinct".format(path, row_index)
            )
        unknown = [gallery_id for gallery_id in ranked_ids if gallery_id not in gallery_set]
        if unknown:
            raise ValidationError(
                "{}:{}: gallery IDs are absent from input gallery: {}".format(
                    path, row_index, ", ".join(repr(item) for item in unknown)
                )
            )


def validate_embeddings(path: Path, expected_rows: int) -> None:
    try:
        embeddings = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValidationError("{}: cannot load NumPy array: {}".format(path, exc)) from exc

    if not isinstance(embeddings, np.ndarray):
        raise ValidationError("{}: expected a .npy ndarray".format(path))
    if embeddings.ndim != 2:
        raise ValidationError(
            "{}: expected a 2D array, got shape {}".format(path, embeddings.shape)
        )
    if embeddings.shape[0] != expected_rows:
        raise ValidationError(
            "{}: expected {} rows, got {}".format(
                path, expected_rows, embeddings.shape[0]
            )
        )
    if embeddings.shape[1] == 0:
        raise ValidationError("{}: embedding dimension must be positive".format(path))
    if embeddings.dtype != np.dtype(np.float32):
        raise ValidationError(
            "{}: expected dtype float32, got {}".format(path, embeddings.dtype)
        )
    if not np.isfinite(embeddings).all():
        raise ValidationError("{}: embeddings contain NaN or infinity".format(path))


def validate_candidates(
    path: Path, query_ids: Sequence[str], gallery_ids: Sequence[str]
) -> None:
    header, rows = _read_headered_csv_rows(path)
    _require_header(path, header, CANDIDATES_HEADER)

    query_set = set(query_ids)
    gallery_set = set(gallery_ids)
    for line_number, row in enumerate(rows, start=2):
        if len(row) != len(CANDIDATES_HEADER):
            raise ValidationError(
                "{}:{}: expected 3 columns, got {}".format(
                    path, line_number, len(row)
                )
            )
        query_id, gallery_id, raw_confidence = row
        if query_id not in query_set:
            raise ValidationError(
                "{}:{}: unknown query_id {!r}".format(path, line_number, query_id)
            )
        if gallery_id not in gallery_set:
            raise ValidationError(
                "{}:{}: unknown gallery_id {!r}".format(path, line_number, gallery_id)
            )
        if not raw_confidence:
            raise ValidationError(
                "{}:{}: confidence must not be empty".format(path, line_number)
            )
        try:
            confidence = float(raw_confidence)
        except ValueError as exc:
            raise ValidationError(
                "{}:{}: invalid confidence {!r}".format(
                    path, line_number, raw_confidence
                )
            ) from exc
        if not math.isfinite(confidence):
            raise ValidationError(
                "{}:{}: confidence must be finite".format(path, line_number)
            )


def validate_outputs(
    query_csv: Path,
    gallery_csv: Path,
    submission_csv: Path,
    embeddings_npy: Path,
    candidates_csv: Path,
) -> None:
    """Validate all output files, raising ``ValidationError`` on failure."""

    query_ids = _read_input_ids(Path(query_csv))
    gallery_ids = _read_input_ids(Path(gallery_csv))
    validate_submission(Path(submission_csv), query_ids, gallery_ids)
    validate_embeddings(Path(embeddings_npy), len(query_ids) + len(gallery_ids))
    validate_candidates(Path(candidates_csv), query_ids, gallery_ids)


def _write_csv(path: Path, rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _run_self_test() -> None:
    """Exercise the real validator with a minimal, valid synthetic fixture."""

    with tempfile.TemporaryDirectory(prefix="reid-validator-") as temp_dir:
        root = Path(temp_dir)
        query_ids = ["q0"]
        gallery_ids = ["g{}".format(index) for index in range(10)]
        _write_csv(root / "query.csv", [["image_id", "x", "y", "w", "h"], ["q0", "0", "0", "1", "1"]])
        _write_csv(
            root / "gallery.csv",
            [["image_id", "x", "y", "w", "h"]]
            + [[gallery_id, "0", "0", "1", "1"] for gallery_id in gallery_ids],
        )
        _write_csv(root / "submission.csv", [["q0"] + gallery_ids])
        _write_csv(root / "candidates.csv", [CANDIDATES_HEADER, ["q0", "g0", "0.5"]])
        np.save(root / "embeddings.npy", np.zeros((11, 4), dtype=np.float32))
        validate_outputs(
            root / "query.csv",
            root / "gallery.csv",
            root / "submission.csv",
            root / "embeddings.npy",
            root / "candidates.csv",
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, help="input query CSV")
    parser.add_argument("--gallery", type=Path, help="input gallery CSV")
    parser.add_argument("--submission", type=Path, help="submission.csv")
    parser.add_argument("--embeddings", type=Path, help="embeddings.npy")
    parser.add_argument("--candidates", type=Path, help="candidates.csv")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate a generated fixture (used by make check)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.self_test:
            supplied_paths = [
                args.query,
                args.gallery,
                args.submission,
                args.embeddings,
                args.candidates,
            ]
            if any(path is not None for path in supplied_paths):
                raise ValidationError("--self-test cannot be combined with file paths")
            _run_self_test()
            print("VALID: built-in output fixture satisfies contract C6")
            return 0

        values = {
            "--query": args.query,
            "--gallery": args.gallery,
            "--submission": args.submission,
            "--embeddings": args.embeddings,
            "--candidates": args.candidates,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValidationError("missing required arguments: {}".format(", ".join(missing)))
        validate_outputs(
            args.query,
            args.gallery,
            args.submission,
            args.embeddings,
            args.candidates,
        )
    except ValidationError as exc:
        print("INVALID: {}".format(exc), file=sys.stderr)
        return 1

    print("VALID: all output files satisfy contract C6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
