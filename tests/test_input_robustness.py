import csv
from pathlib import Path

import numpy as np
import pytest

from reid.infer.run import CANDIDATES_PER_QUERY
import torch
from torchvision.io import write_png

from reid.infer.extractor import Extractor
from reid.infer.run import read_inputs, run_pipeline


CFG = {
    "input_size": [16, 16],
    "bbox_pad": 0.0,
    "resize": "stretch",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "embed_dim": 4,
}


def _write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "x", "y", "w", "h"])
        writer.writerows(rows)


def _stub_extractor() -> Extractor:
    extractor = Extractor.__new__(Extractor)
    extractor.cfg = CFG
    return extractor


def test_png_is_resolved_from_bare_id_and_decoded_as_rgb(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    frame = torch.arange(3 * 7 * 5, dtype=torch.uint8).reshape(3, 7, 5)
    write_png(frame, str(images / "vehicle.png"))
    csv_path = tmp_path / "query.csv"
    _write_csv(csv_path, [["vehicle", -3, -2, 20, 20]])

    row = read_inputs(csv_path, images)[0]
    assert row.image_path.name == "vehicle.png"
    prepared = _stub_extractor()._decode_preprocess((row.image_path, row.bbox))

    assert prepared.shape == (3, 16, 16)
    assert torch.isfinite(prepared).all()


@pytest.mark.parametrize(
    "bbox",
    [
        (-100.0, -100.0, 1.0, 1.0),
        (100.0, 100.0, 1.0, 1.0),
        (2.25, 3.75, 0.01, 0.01),
    ],
)
def test_out_of_frame_and_tiny_bbox_produce_a_valid_model_input(tmp_path, bbox):
    image_path = tmp_path / "frame.PNG"
    write_png(torch.full((3, 8, 9), 127, dtype=torch.uint8), str(image_path))

    prepared = _stub_extractor()._decode_preprocess((image_path, bbox))

    assert prepared.shape == (3, 16, 16)
    assert torch.isfinite(prepared).all()


def test_corrupt_image_fails_with_path_and_decode_context(tmp_path):
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match=r"failed to decode image .*broken\.png"):
        _stub_extractor()._decode_preprocess((image_path, (0.0, 0.0, 1.0, 1.0)))


@pytest.mark.parametrize("bbox", [(0, 0, 0, 1), (0, 0, 1, -1), (0, 0, "nan", 1)])
def test_invalid_bbox_is_rejected_before_decode(tmp_path, bbox):
    images = tmp_path / "images"
    images.mkdir()
    csv_path = tmp_path / "query.csv"
    _write_csv(csv_path, [["missing", *bbox]])

    with pytest.raises(ValueError, match="bbox"):
        read_inputs(csv_path, images)


class DynamicFakeExtractor:
    cfg = {"candidate_threshold": -1.0}

    def extract_batch(self, items):
        rows = []
        for path, _bbox in items:
            value = sum(Path(path).stem.encode("utf-8"))
            rows.append([1.0, float(value % 17 + 1), float(value % 7 + 1), 0.5])
        return np.asarray(rows, dtype=np.float32)


def test_output_sizes_come_only_from_input_csvs(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    query_ids = [f"query_{index}" for index in range(3)]
    gallery_ids = [f"gallery_{index}" for index in range(12)]
    for image_id in query_ids + gallery_ids:
        (images / f"{image_id}.png").touch()
    query_csv = tmp_path / "query.csv"
    gallery_csv = tmp_path / "gallery.csv"
    _write_csv(query_csv, [[image_id, 0, 0, 1, 1] for image_id in query_ids])
    _write_csv(gallery_csv, [[image_id, 0, 0, 1, 1] for image_id in gallery_ids])

    out = tmp_path / "out"
    run_pipeline(
        images,
        query_csv,
        gallery_csv,
        out,
        tmp_path / "unused-model",
        "cpu",
        batch_size=5,
        candidate_threshold=-1.0,
        extractor=DynamicFakeExtractor(),
    )

    submission = list(csv.reader((out / "submission.csv").open()))
    candidates = list(csv.reader((out / "candidates.csv").open()))
    embeddings = np.load(out / "embeddings.npy")
    assert len(submission) == len(query_ids)
    # D-048: у отвеченного запроса может быть до CANDIDATES_PER_QUERY строк
    answered = {row[0] for row in candidates[1:]}
    assert answered <= set(query_ids)
    assert len(candidates) - 1 <= len(query_ids) * CANDIDATES_PER_QUERY
    assert embeddings.shape == (len(query_ids) + len(gallery_ids), 4)
