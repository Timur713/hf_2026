"""Synthetic C3 checkpoints exercise fusion without downloading any weights."""

import csv
import json
import numpy as np
import pytest
import torch
from torchvision.io import ImageReadMode, read_image, write_jpeg, write_png

from reid.data.preprocess import preprocess_frame
from reid.infer.extractor import Extractor
from reid.infer.fusion import load_fusion_config
from reid.infer.plate_mask import PlateMaskConfig, PlateMaskedExtractor, apply_plate_mask
from reid.model import Embedder, save
from reid.infer.run import run_pipeline


class TinyBackbone(torch.nn.Module):
    num_features = 2
    num_prefix_tokens = 1

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 2)

    def forward_features(self, batch):
        rgb = batch.mean(dim=(-2, -1))
        cls = self.projection(rgb)
        return torch.stack([cls, cls + 1.0, cls * 0.75 + 0.1], dim=1)


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("reid.model.build_backbone", lambda *args, **kwargs: TinyBackbone())
    root = tmp_path / "fusion"
    root.mkdir()
    manifest = {
        "schema_version": 1, "name": "synthetic-fusion",
        "components": [{"path": "components/0", "similarity_std": 0.25},
                       {"path": "components/1", "similarity_std": 0.04}],
        "candidate_policy": "gallery_z", "candidate_threshold": 6.0,
        "rerank_mode": "standardized",
    }
    for index, entry in enumerate(manifest["components"]):
        pool = "cls_avg" if index == 0 else "avg"
        model = Embedder(TinyBackbone(), pool=pool, neck="bnneck").eval()
        with torch.no_grad():
            model.backbone.projection.weight.copy_(torch.tensor([[1., .2, .3], [.1, .3, .7]]) + index)
            model.backbone.projection.bias.copy_(torch.tensor([.8, 1.2]))
            model.neck.running_mean.fill_(2.0)
            model.neck.running_var.fill_(4.0)
        cfg = {
            "name": f"tiny-{index}", "arch": "synthetic", "pool": pool,
            "neck": "bnneck", "embed_dim": 4 if index == 0 else 2,
            "input_size": [8 + index * 2, 10 + index], "bbox_pad": index * 0.1,
            "resize": "stretch" if index == 0 else "letterbox",
            "mean": [0.1, 0.2, 0.3], "std": [0.5, 0.6, 0.7],
            "infer_dtype": "fp16" if index == 0 else "fp32", "tta_flip": False,
        }
        save(model, cfg, str(root / entry["path"]))
    (root / "fusion_config.json").write_text(json.dumps(manifest))
    frames = []
    for index, extension in enumerate(("png", "jpg")):
        frame = (torch.arange(3 * 17 * 23).reshape(3, 17, 23) * (index + 1)).remainder(255).to(torch.uint8)
        path = tmp_path / f"frame-{index}.{extension}"
        (write_png if extension == "png" else write_jpeg)(frame, str(path))
        frames.append((path, (1.0, 2.0, 19.0, 12.0)))
    return root, manifest, frames


def expected_fusion(extractor, items, mask=None):
    blocks = []
    for model, cfg, _, std in extractor._fusion_components:
        frames = [read_image(str(path), mode=ImageReadMode.RGB) for path, _ in items]
        if mask is not None:
            frames = [apply_plate_mask(frame, item[1], mask) for frame, item in zip(frames, items)]
        tensors = torch.stack([preprocess_frame(frame, item[1], cfg)
                               for frame, item in zip(frames, items)])
        with torch.inference_mode():
            blocks.append(torch.nn.functional.normalize(model.features(tensors), dim=1) / np.sqrt(std))
    return torch.nn.functional.normalize(torch.cat(blocks, dim=1), dim=1).numpy()


@pytest.mark.parametrize("workers", [1, 3])
def test_fusion_formula_c3_preprocessing_and_decode_once(artifact, monkeypatch, workers):
    root, manifest, items = artifact
    reads = []

    def decode(path, **kwargs):
        reads.append(path)
        return read_image(path, **kwargs)

    monkeypatch.setattr("reid.infer.extractor.read_image", decode)
    extractor = Extractor(root, "cpu", preprocess_workers=workers)
    try:
        output = extractor.extract_batch(items)
        np.testing.assert_allclose(output, expected_fusion(extractor, items), atol=1e-7)
        assert sorted(reads) == sorted(str(item[0]) for item in items)
        assert output.shape == (2, 6)
        assert output.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0, atol=1e-7)
        assert extractor.cfg["embed_dim"] == 6
        assert extractor.cfg["candidate_policy"] == "gallery_z"
        assert extractor.cfg["candidate_threshold"] == 6.0
        assert extractor.cfg["rerank_mode"] == "standardized"
        assert extractor.cfg["components"] == manifest["components"]
        assert all(not module.training for module in extractor.model.modules())
        assert all(not parameter.requires_grad for parameter in extractor.model.parameters())
    finally:
        if extractor._preprocess_pool:
            extractor._preprocess_pool.shutdown()


def test_empty_batches_and_query_independence_have_no_fitting_state(artifact):
    root, _, items = artifact
    extractor = Extractor(root, "cpu")
    assert extractor.extract_batch([]).shape == (0, 6)
    together = extractor.extract_batch(items)
    np.testing.assert_allclose(extractor.extract(*items[0]), together[0], atol=1e-7)
    assert extractor.extract_batch([]).dtype == np.float32
    reordered = extractor.extract_batch(items[::-1] + items[:1])
    np.testing.assert_allclose(reordered, together[[1, 0, 0]], atol=1e-7)


def test_explicit_rerank_weight_survives_artifact_loading(artifact):
    root, manifest, _ = artifact
    manifest["rerank_weight"] = 0.15
    (root / "fusion_config.json").write_text(json.dumps(manifest))
    assert Extractor(root, "cpu").cfg["rerank_weight"] == 0.15


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_extreme_finite_std_common_factor_does_not_overflow(artifact, magnitude):
    root, manifest, items = artifact
    original = Extractor(root, "cpu").extract_batch(items)
    for entry in manifest["components"]:
        entry["similarity_std"] *= magnitude
    (root / "fusion_config.json").write_text(json.dumps(manifest))
    actual = Extractor(root, "cpu").extract_batch(items)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, original, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, atol=1e-7)


def test_smallest_positive_std_can_be_used_without_reciprocal_overflow(artifact):
    root, manifest, items = artifact
    for entry in manifest["components"]:
        entry["similarity_std"] = 5e-324
    (root / "fusion_config.json").write_text(json.dumps(manifest))
    actual = Extractor(root, "cpu").extract_batch(items)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(actual[:, :4], axis=1), np.sqrt(0.5), atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(actual[:, 4:], axis=1), np.sqrt(0.5), atol=1e-7)


def test_unrepresentable_scale_spread_fails_instead_of_dropping_member(artifact):
    root, manifest, _ = artifact
    manifest["components"][0]["similarity_std"] = 1e-300
    manifest["components"][1]["similarity_std"] = 1e300
    (root / "fusion_config.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dynamic range"):
        Extractor(root, "cpu")


def test_cosine_equals_fixed_standardized_similarity_ranking(artifact):
    root, _, items = artifact
    extractor = Extractor(root, "cpu")
    embedding = extractor.extract_batch(items)
    expected = 0
    total_weight = 0
    offset = 0
    for _, cfg, _, std in extractor._fusion_components:
        block = embedding[:, offset:offset + cfg["embed_dim"]]
        unit = block / np.linalg.norm(block, axis=1, keepdims=True)
        expected += (unit @ unit.T) / std
        total_weight += 1 / std
        offset += cfg["embed_dim"]
    np.testing.assert_allclose(embedding @ embedding.T, expected / total_weight, atol=2e-7)


def test_fusion_plate_mask_reaches_every_member_and_decodes_once(artifact, monkeypatch):
    root, _, items = artifact
    mask = PlateMaskConfig(width_ratio=0.9, aspect_ratio=1.0, fill=13)
    reads = []

    def decode(path, **kwargs):
        reads.append(path)
        return read_image(path, **kwargs)

    monkeypatch.setattr("reid.infer.extractor.read_image", decode)
    masked = PlateMaskedExtractor(root, "cpu", mask=mask)
    output = masked.extract_batch(items)
    assert len(reads) == len(items)
    np.testing.assert_allclose(output, expected_fusion(masked, items, mask), atol=1e-7)
    unmasked = expected_fusion(masked, items)
    assert not np.allclose(output[:, :4], unmasked[:, :4])
    assert not np.allclose(output[:, 4:], unmasked[:, 4:])


@pytest.mark.parametrize("masked", [False, True])
def test_real_decode_fusion_to_c6_with_verifier(artifact, tmp_path, monkeypatch, masked):
    root, _, _ = artifact
    generator = torch.Generator().manual_seed(24)
    for index in range(14):
        frame = torch.randint(0, 256, (3, 17, 23), dtype=torch.uint8, generator=generator)
        write_png(frame, str(tmp_path / f"img{index}.png"))
    paths = []
    for name, indices in (("query", range(2)), ("gallery", range(2, 14))):
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            writer.writerows((f"img{i}.png", 1, 2, 19, 12) for i in indices)
        paths.append(path)
    reads = []
    def decode(path, **kwargs):
        reads.append(path)
        return read_image(path, **kwargs)
    monkeypatch.setattr("reid.infer.extractor.read_image", decode)
    class Scorer:
        def score_pairs(self, query, candidates):
            return np.arange(len(candidates), dtype=np.float32)
    extractor = (PlateMaskedExtractor if masked else Extractor)(root, "cpu")
    out = tmp_path / "outputs"
    run_pipeline(tmp_path, *paths, out, root, "cpu", 3, None,
                 extractor=extractor, verifier=Scorer())
    # Real PNG decode -> C3 -> two C3 checkpoints -> gallery policy -> verifier
    # -> all three files and the actual validator, not a fake Extractor.
    assert len(reads) == 14
    assert np.load(out / "embeddings.npy", allow_pickle=False).shape == (14, 6)
    with (out / "submission.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 2 and all(len(row) == 11 for row in rows)


def test_legacy_preprocess_override_cannot_silently_bypass_mask_in_fusion(artifact):
    class LegacyMask(Extractor):
        def _decode_preprocess(self, item):
            return super()._decode_preprocess(item) * 0

    with pytest.raises(ValueError, match="_decode_frame"):
        LegacyMask(artifact[0], "cpu")


@pytest.mark.parametrize("key,value", [
    ("schema_version", 2), ("schema_version", True), ("name", ""),
    ("candidate_policy", "cosine"), ("rerank_mode", "raw"),
    ("candidate_threshold", float("nan")), ("candidate_threshold", True),
    ("candidate_threshold", 0), ("components", []), ("components", {}),
])
def test_invalid_manifest_fields_fail_closed(artifact, key, value):
    root, manifest, _ = artifact
    manifest[key] = value
    (root / "fusion_config.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_fusion_config(root)


@pytest.mark.parametrize("std", [0, -1, True, "0.5", None, float("inf"), float("nan")])
def test_invalid_fixed_scales_fail_closed(artifact, std):
    root, manifest, _ = artifact
    manifest["components"][0]["similarity_std"] = std
    (root / "fusion_config.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="similarity_std"):
        load_fusion_config(root)


def test_unknown_or_duplicate_manifest_keys_and_single_component_rejected(artifact):
    root, manifest, _ = artifact
    config = root / "fusion_config.json"
    config.write_text(json.dumps({**manifest, "fit_query_batch": True}))
    with pytest.raises(ValueError, match="exactly"):
        load_fusion_config(root)
    config.write_text('{"name": "x", "name": "y"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_fusion_config(root)
    config.write_text(json.dumps({**manifest, "components": manifest["components"][:1]}))
    with pytest.raises(ValueError, match="at least two"):
        load_fusion_config(root)


@pytest.mark.parametrize("raw", ["[]", "{", "null"])
def test_malformed_json_or_wrong_root_type_rejected(artifact, raw):
    root, _, _ = artifact
    (root / "fusion_config.json").write_text(raw)
    with pytest.raises(ValueError):
        load_fusion_config(root)


def test_unknown_component_fields_rejected(artifact):
    root, manifest, _ = artifact
    manifest["components"][0]["fit_on_batch"] = True
    (root / "fusion_config.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly path and similarity_std"):
        load_fusion_config(root)


@pytest.mark.parametrize("field,value", [("tta_flip", True), ("infer_dtype", "int8"),
                                         ("embed_dim", 0), ("embed_dim", True)])
def test_unsupported_component_settings_rejected(artifact, field, value):
    import yaml

    root, _, _ = artifact
    config = root / "components/0/model_config.yaml"
    cfg = yaml.safe_load(config.read_text())
    cfg[field] = value
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=field):
        Extractor(root, "cpu")


@pytest.mark.parametrize("value", ["../escape", "/tmp/outside-fusion", ".", "missing", ""])
def test_unsafe_or_missing_component_path_rejected(artifact, value):
    root, manifest, _ = artifact
    manifest["components"][0]["path"] = value
    (root / "fusion_config.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_fusion_config(root)


def test_symlinks_cannot_escape_and_component_paths_cannot_repeat(artifact):
    root, manifest, _ = artifact
    config = root / "fusion_config.json"
    (root / "outside").symlink_to(root.parent, target_is_directory=True)
    manifest["components"][0]["path"] = "outside"
    config.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside"):
        load_fusion_config(root)
    manifest["components"][0]["path"] = "components/1"
    config.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        load_fusion_config(root)


def test_component_weight_symlink_must_stay_inside_root(artifact):
    root, _, _ = artifact
    weights = root / "components/0/weights.safetensors"
    outside = root.parent / "outside.safetensors"
    weights.rename(outside)
    weights.symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        load_fusion_config(root)


def test_ambiguous_and_nested_artifacts_fail_closed(artifact):
    root, _, _ = artifact
    (root / "model_config.yaml").write_text("name: ambiguous")
    with pytest.raises(ValueError, match="ambiguous"):
        load_fusion_config(root)
    (root / "model_config.yaml").unlink()
    (root / "components/0/fusion_config.json").write_text("{}")
    with pytest.raises(ValueError, match="nested"):
        load_fusion_config(root)


@pytest.mark.parametrize("bad", ["zero", "nan", "shape"])
def test_invalid_component_outputs_are_not_saved(artifact, bad):
    root, _, items = artifact
    extractor = Extractor(root, "cpu")
    model = extractor._fusion_components[0][0]
    model.features = lambda batch: torch.full(
        (len(batch), 3 if bad == "shape" else 4), float("nan") if bad == "nan" else 0.0,
    )
    with pytest.raises(ValueError, match="fusion component 0"):
        extractor.extract_batch(items)


def test_corrupt_image_reports_path_in_fusion(artifact):
    root, _, items = artifact
    corrupt = root.parent / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    extractor = Extractor(root, "cpu")
    with pytest.raises(ValueError, match="failed to decode image.*corrupt.jpg"):
        extractor.extract_batch(items[:1] + [(corrupt, items[0][1])])
