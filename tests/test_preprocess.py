"""Тесты общей предобработки (reid/data/preprocess.py) и формата артефакта модели."""
import torch

from reid.data.preprocess import clip_bbox, expand_box, preprocess_frame, to_model_input

CFG = dict(input_size=[224, 224], bbox_pad=0.0, resize="stretch",
           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def test_clip_bbox_handles_out_of_frame_and_degenerate():
    assert clip_bbox(-10, -5, 50, 40, 1920, 1080) == (0, 0, 40, 35)
    assert clip_bbox(1900, 1070, 100, 100, 1920, 1080) == (1900, 1070, 1920, 1080)
    x0, y0, x1, y1 = clip_bbox(5000, 5000, 0, 0, 1920, 1080)  # BBox целиком за кадром
    assert x1 > x0 and y1 > y0 and x1 <= 1920 and y1 <= 1080


def test_expand_box_is_clipped_to_frame():
    assert expand_box((0, 0, 100, 100), 0.1, 1920, 1080) == (0, 0, 110, 110)
    assert expand_box((10, 10, 20, 20), 0.0, 1920, 1080) == (10, 10, 20, 20)


def test_to_model_input_shapes_and_letterbox_padding():
    crop = torch.full((3, 50, 200), 128, dtype=torch.uint8)
    for policy in ("stretch", "letterbox"):
        x = to_model_input(crop, [224, 224], CFG["mean"], CFG["std"], policy)
        assert x.shape == (3, 224, 224) and torch.isfinite(x).all()
    lb = to_model_input(crop, [224, 224], CFG["mean"], CFG["std"], "letterbox")
    assert lb[:, 0, :].abs().max() < 1e-5  # верхнее поле — ноль после нормализации


def test_preprocess_frame_uses_bbox():
    frame = torch.zeros((3, 1080, 1920), dtype=torch.uint8)
    frame[:, 100:200, 300:500] = 255
    x = preprocess_frame(frame, (300, 100, 200, 100), CFG)
    white = (torch.ones(3) - torch.tensor(CFG["mean"])) / torch.tensor(CFG["std"])
    assert torch.allclose(x.mean(dim=(1, 2)), white, atol=1e-4)


def test_model_artifact_roundtrip(tmp_path):
    from reid.model import Embedder, build_backbone, load, save
    torch.manual_seed(0)
    model = Embedder(build_backbone("dinov2_vits14", [56, 56]), "cls_avg").eval()
    cfg = dict(CFG, name="t", arch="dinov2_vits14", pool="cls_avg", embed_dim=768, input_size=[56, 56])
    save(model, cfg, str(tmp_path))
    loaded = load(str(tmp_path))
    x = torch.randn(2, 3, 56, 56)
    with torch.inference_mode():
        a, b = model(x), loaded(x)
    assert a.shape == (2, 768) and torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a.norm(dim=1), torch.ones(2), atol=1e-5)
