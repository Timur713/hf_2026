"""Original member vectors are exposed without changing legacy C5 fusion."""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from reid.infer.extractor import Extractor
from reid.infer.plate_mask import PlateMaskConfig, PlateMaskedExtractor
from reid.infer.run import read_inputs

pytest_plugins = ("tests.test_fusion_extractor",)


def original_fusion(extractor, items):
    """Frozen arithmetic of pre-D082 Extractor._extract_fusion, CPU fixture."""
    tensors = [extractor._decode_preprocess_fusion(item) for item in items]
    reference = math.sqrt(
        min(component[3] for component in extractor._fusion_components)
    )
    blocks = []
    with torch.inference_mode():
        for index, (model, _, _, scale) in enumerate(extractor._fusion_components):
            features = model.features(
                torch.stack([row[index] for row in tensors])
            ).float()
            norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
            blocks.append((features / norms) * (reference / math.sqrt(scale)))
        return torch.nn.functional.normalize(torch.cat(blocks, dim=1), dim=1).numpy()


@pytest.mark.parametrize("workers", [1, 3])
def test_components_equal_standalone_c5_and_decode_forward_once(
    artifact, monkeypatch, workers
):
    root, manifest, items = artifact
    extractor = Extractor(root, "cpu", preprocess_workers=workers)
    expected = [
        Extractor(root / entry["path"], "cpu").extract_batch(items)
        for entry in manifest["components"]
    ]
    decoded, forwarded = [], [0, 0]
    decode = extractor._decode_frame

    def counted_decode(item):
        decoded.append(item[0])
        return decode(item)

    monkeypatch.setattr(extractor, "_decode_frame", counted_decode)
    for index, (model, _, _, _) in enumerate(extractor._fusion_components):
        forward = model.features

        def counted_forward(batch, index=index, forward=forward):
            forwarded[index] += 1
            return forward(batch)

        monkeypatch.setattr(model, "features", counted_forward)
    try:
        actual = extractor.extract_components_batch(items)
        assert isinstance(actual, tuple) and len(actual) == 2
        assert sorted(decoded) == sorted(item[0] for item in items)
        assert forwarded == [1, 1]
        for output, standalone in zip(actual, expected):
            assert output.dtype == np.float32
            np.testing.assert_array_equal(output, standalone)
            np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1, atol=1e-7)
    finally:
        if extractor._preprocess_pool:
            extractor._preprocess_pool.shutdown()


def test_additive_api_keeps_existing_fused_arithmetic_byte_identical(artifact):
    root, _, items = artifact
    extractor = Extractor(root, "cpu")
    expected = original_fusion(extractor, items)
    before = extractor.extract_batch(items)
    components = extractor.extract_components_batch(items)
    # Outputs are independent; callers cannot mutate subsequent extraction state.
    components[0].fill(0)
    after = extractor.extract_batch(items)
    assert before.tobytes() == expected.tobytes() == after.tobytes()
    np.testing.assert_array_equal(
        extractor.extract(*items[0]), original_fusion(extractor, items[:1])[0]
    )


def test_empty_components_have_member_dimensions_and_do_no_work(artifact, monkeypatch):
    root, _, _ = artifact
    extractor = Extractor(root, "cpu")

    def forbidden(*args, **kwargs):
        pytest.fail("empty extraction decoded or ran a model")

    monkeypatch.setattr(extractor, "_decode_frame", forbidden)
    for model, _, _, _ in extractor._fusion_components:
        monkeypatch.setattr(model, "features", forbidden)
    result = extractor.extract_components_batch([])
    assert isinstance(result, tuple)
    assert [array.shape for array in result] == [(0, 4), (0, 2)]
    assert all(array.dtype == np.float32 for array in result)


@pytest.mark.parametrize("empty", [False, True])
def test_single_model_component_misuse_fails(artifact, empty):
    root, manifest, items = artifact
    extractor = Extractor(root / manifest["components"][0]["path"], "cpu")
    with pytest.raises(ValueError, match="requires a fusion artifact"):
        extractor.extract_components_batch([] if empty else items)


def test_original_components_preserve_mask_hook_for_every_member(artifact):
    root, manifest, items = artifact
    mask = PlateMaskConfig(width_ratio=0.9, aspect_ratio=1, fill=13)
    masked = PlateMaskedExtractor(root, "cpu", mask=mask).extract_components_batch(
        items
    )
    clean = Extractor(root, "cpu").extract_components_batch(items)
    for index, entry in enumerate(manifest["components"]):
        expected = PlateMaskedExtractor(
            root / entry["path"], "cpu", mask=mask
        ).extract_batch(items)
        np.testing.assert_array_equal(masked[index], expected)
        assert not np.allclose(masked[index], clean[index])


def test_components_do_not_depend_on_fixed_fusion_scales_or_other_queries(artifact):
    root, _, items = artifact
    extractor = Extractor(root, "cpu")
    original = extractor.extract_components_batch(items)
    # A component-aware caller applies fixed scales downstream. They must not
    # be baked into the original embeddings exposed by this API.
    extractor._fusion_components = [
        (model, cfg, dtype, std * (index + 3))
        for index, (model, cfg, dtype, std) in enumerate(extractor._fusion_components)
    ]
    changed = extractor.extract_components_batch(items)
    reordered = extractor.extract_components_batch(items[::-1])
    for before, after, reverse in zip(original, changed, reordered):
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(before, reverse[::-1])


def test_components_use_exact_standalone_normalization_epsilon(artifact, monkeypatch):
    root, _, items = artifact
    extractor = Extractor(root, "cpu")
    # Tiny finite nonzero vectors distinguish standalone F.normalize(eps=1e-12)
    # from fusion's division by norm. Typical models have much larger norms.
    for model, cfg, _, _ in extractor._fusion_components:
        monkeypatch.setattr(
            model,
            "features",
            lambda batch, dim=cfg["embed_dim"]: torch.full(
                (len(batch), dim), 1e-14, dtype=torch.float32
            ),
        )
    actual = extractor.extract_components_batch(items)
    for output in actual:
        expected = torch.nn.functional.normalize(
            torch.full(output.shape, 1e-14), dim=1
        ).numpy()
        np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("bad", ["zero", "nan", "inf", "shape"])
def test_components_retain_fusion_validation(artifact, monkeypatch, bad):
    root, _, items = artifact
    extractor = Extractor(root, "cpu")
    value = {"zero": 0.0, "nan": float("nan"), "inf": float("inf"), "shape": 1.0}[bad]
    monkeypatch.setattr(
        extractor._fusion_components[0][0],
        "features",
        lambda batch: torch.full((len(batch), 3 if bad == "shape" else 4), value),
    )
    with pytest.raises(ValueError, match="fusion component 0"):
        extractor.extract_components_batch(items)


def test_optional_real_train_component_and_legacy_parity():
    """Explicit local engineering test; never downloads or measures quality."""
    model_dir = os.environ.get("FALCON_COMPONENTS_MODEL")
    images_dir = os.environ.get("FALCON_TRAIN_IMAGES")
    if not model_dir or not images_dir:
        pytest.skip(
            "set FALCON_COMPONENTS_MODEL and FALCON_TRAIN_IMAGES for real CPU parity"
        )
    # Only train fold0 inputs. The sealed fold2 is never read or forwarded.
    rows = read_inputs(Path("splits/fold0_query.csv"), Path(images_dir))[:2]
    items = [(row.image_path, row.bbox) for row in rows]
    extractor = Extractor(model_dir, "cpu", preprocess_workers=1)
    components = extractor.extract_components_batch(items)
    for index, entry in enumerate(extractor.cfg["components"]):
        single = Extractor(Path(model_dir) / entry["path"], "cpu", preprocess_workers=1)
        np.testing.assert_array_equal(components[index], single.extract_batch(items))
        del single
    assert (
        extractor.extract_batch(items).tobytes()
        == original_fusion(extractor, items).tobytes()
    )
