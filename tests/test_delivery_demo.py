import csv
from pathlib import Path

import pytest

from scripts.record_delivery_demo import prepare_demo


def test_train_demo_is_small_disjoint_and_does_not_modify_source(tmp_path):
    root, data, dest = tmp_path / "repo", tmp_path / "dataset", tmp_path / "demo"
    (root / "splits").mkdir(parents=True)
    (data / "images").mkdir(parents=True)
    for role in ("query", "gallery"):
        with (root / f"splits/fold0_{role}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            for i in range(12):
                writer.writerow([f"{role}{i}", 0, 0, 10, 10])
                (data / "images" / f"{role}{i}.jpg").write_bytes(b"fixture")
    before = {p.name: p.read_bytes() for p in (data / "images").iterdir()}
    result = prepare_demo(data, dest, root, count=10)
    assert result["images"] == 20 and not result["images_published"]
    assert before == {p.name: p.read_bytes() for p in (data / "images").iterdir()}
    assert len(list((dest / "images").iterdir())) == 20
    with pytest.raises(FileExistsError):
        prepare_demo(data, dest, root)
    with pytest.raises(ValueError, match="organizer dataset"):
        prepare_demo(data, data / "inside", root)
