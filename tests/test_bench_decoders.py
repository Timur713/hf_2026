import csv
from pathlib import Path

import pytest

from scripts import bench_decoders as bench


class FakeBackend(bench.DecoderBackend):
    name = "fake"
    device = "cpu-test"

    def __init__(self):
        self.decode_calls = 0
        self.sync_calls = 0

    def decode_batch(self, samples):
        self.decode_calls += 1
        return [sample.image_id for sample in samples]

    def synchronize(self):
        self.sync_calls += 1


def _sample(index=0):
    return bench.Sample("image{}".format(index), Path("image{}.jpg".format(index)), (0, 0, 2, 2))


def test_benchmark_result_structure_and_warmup_accounting():
    backend = FakeBackend()
    result = bench.benchmark_backend(
        backend,
        [_sample(0), _sample(1), _sample(2)],
        batch_size=2,
        warmup=2,
        iterations=3,
    )

    assert result["method"] == "fake"
    assert result["status"] == "ok"
    assert result["samples_measured"] == 6
    assert result["batch_size"] == 2
    assert result["iterations"] == 3
    assert result["batch_latency_p50_ms"] >= 0
    assert result["per_image_latency_p50_ms"] == pytest.approx(result["batch_latency_p50_ms"] / 2)
    assert result["throughput_fps"] > 0
    assert backend.decode_calls == 5
    # One sync after each warmup plus two timing-boundary syncs per iteration.
    assert backend.sync_calls == 2 + 2 * 3


def test_load_samples_reads_only_required_columns_and_resolves_jpeg(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "abc.jpg").write_bytes(b"synthetic-jpeg-placeholder")
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"])
        writer.writerow(["abc", 1, 2, 3, 4, 100, 7])

    samples = bench.load_samples(csv_path, images)

    assert samples == [bench.Sample("abc", images / "abc.jpg", (1, 2, 3, 4))]


@pytest.mark.parametrize(
    "bbox,size,expected",
    [
        ((1, 2, 3, 4), (10, 10), (1, 2, 4, 6)),
        ((-5, -7, 20, 30), (10, 8), (0, 0, 10, 8)),
        ((15, 20, 2, 2), (10, 8), (9, 7, 10, 8)),
    ],
)
def test_clamp_bbox(bbox, size, expected):
    assert bench.clamp_bbox(bbox, *size) == expected


def test_pillow_backends_decode_and_crop_a_synthetic_jpeg(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    path = tmp_path / "synthetic.jpg"
    image_module.new("RGB", (32, 24), color=(10, 20, 30)).save(path, format="JPEG")
    sample = bench.Sample("synthetic", path, (4, 3, 16, 12))

    full_crop = bench.PillowBackend(None).decode_batch([sample])[0]
    draft_crop = bench.PillowBackend(0.5).decode_batch([sample])[0]

    assert full_crop.size == (16, 12)
    assert draft_crop.size[0] > 0 and draft_crop.size[1] > 0
    assert draft_crop.size[0] <= full_crop.size[0]
    assert draft_crop.size[1] <= full_crop.size[1]


def test_markdown_contains_results_and_unavailable_reason():
    report = {
        "run_label": "cpu-smoke-not-a10",
        "scope": "disk_read+jpeg_decode+bbox_crop",
        "results": [
            {
                "method": "pil_full",
                "status": "ok",
                "device": "cpu",
                "batch_size": 1,
                "batch_latency_p50_ms": 3.0,
                "per_image_latency_p50_ms": 3.0,
                "throughput_fps": 300.0,
            },
            {
                "method": "dali_roi",
                "status": "unavailable",
                "batch_size": 1,
                "reason": "not | installed",
            },
        ],
    }

    rendered = bench._markdown(report)

    assert "cpu-smoke-not-a10" in rendered
    assert "| pil_full | ok |" in rendered
    assert "not \\| installed" in rendered
