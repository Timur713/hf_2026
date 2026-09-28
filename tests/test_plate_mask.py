import torch
from torchvision.io import write_png

from reid.data.preprocess import preprocess_frame
from reid.infer.plate_mask import (
    PlateMaskConfig,
    PlateMaskedExtractor,
    apply_plate_mask,
    plate_mask_box,
)


CFG = {
    "input_size": [12, 12],
    "bbox_pad": 0.0,
    "resize": "stretch",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "embed_dim": 4,
}


def test_plate_mask_is_inside_vehicle_bbox_and_has_plate_shape():
    config = PlateMaskConfig(width_ratio=0.4, aspect_ratio=4.0, center_x=0.5, center_y=0.75)

    x0, y0, x1, y1 = plate_mask_box((10.0, 20.0, 100.0, 50.0), 200, 100, config)

    assert 10 <= x0 < x1 <= 110
    assert 20 <= y0 < y1 <= 70
    assert x1 - x0 == 40
    assert y1 - y0 == 10


def test_apply_plate_mask_clones_input_and_uses_solid_fill():
    frame = torch.zeros((3, 60, 100), dtype=torch.uint8)
    config = PlateMaskConfig(width_ratio=0.5, aspect_ratio=5.0, fill=231)
    box = plate_mask_box((10.0, 10.0, 80.0, 40.0), 100, 60, config)

    masked = apply_plate_mask(frame, (10.0, 10.0, 80.0, 40.0), config)

    x0, y0, x1, y1 = box
    assert torch.count_nonzero(frame) == 0
    assert torch.all(masked[:, y0:y1, x0:x1] == 231)
    outside = masked.clone()
    outside[:, y0:y1, x0:x1] = 0
    assert torch.count_nonzero(outside) == 0


def test_masked_extractor_applies_fill_before_shared_c3(tmp_path):
    image_path = tmp_path / "vehicle.png"
    frame = (
        torch.arange(3 * 20 * 30, dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .reshape(3, 20, 30)
    )
    write_png(frame, str(image_path))
    bbox = (3.0, 4.0, 22.0, 12.0)
    config = PlateMaskConfig(fill=17)
    extractor = PlateMaskedExtractor.__new__(PlateMaskedExtractor)
    extractor.cfg = CFG
    extractor.plate_mask = config

    actual = extractor._decode_preprocess((image_path, bbox))
    expected = preprocess_frame(apply_plate_mask(frame, bbox, config), bbox, CFG)

    assert torch.equal(actual, expected)


def test_plate_mask_handles_tiny_out_of_frame_bbox():
    frame = torch.full((3, 5, 7), 255, dtype=torch.uint8)

    masked = apply_plate_mask(frame, (100.0, -100.0, 0.01, 0.01))

    assert masked.shape == frame.shape
    assert torch.count_nonzero(masked != frame) == 3
