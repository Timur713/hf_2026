"""Production C5 and pipeline must share exact component/fusion arithmetic."""

import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reid.infer.d082 import extract_component_rows
from reid.infer.d082_extractor import D082Extractor, GraphFeatures
from reid.infer.extractor import Extractor
from reid.post.ranker import PreparedRankerGallery

pytest_plugins = ("tests.test_fusion_extractor",)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 17])
def test_production_api_matches_reference_components_and_fusion(
    artifact, device, count
):
    root, cfg, original = artifact
    items = (original * 9)[:count]
    reference = Extractor(
        root, device, preprocess_workers=1, fusion_internal_batch_size=4
    )
    candidate = D082Extractor(root, device, preprocess_workers=3)
    scales = [c["similarity_std"] for c in cfg["components"]]
    fusion = SimpleNamespace(
        coefficients=tuple(math.sqrt(min(scales)) / math.sqrt(s) for s in scales)
    )
    rows = [SimpleNamespace(image_path=p, bbox=b) for p, b in items]
    try:
        # First graph capture must also work on singleton broadcast input.
        candidate.extract_batch(items[:1])
        expected = extract_component_rows(reference, rows, len(rows))
        actual = candidate.extract_components_batch(items)
        for block, raw in zip(actual, reference.extract_components_batch(items)):
            np.testing.assert_array_equal(block, raw)
        fused = candidate.extract_batch(items)
        np.testing.assert_array_equal(
            fused, PreparedRankerGallery._fuse(fusion, expected)
        )
        np.testing.assert_array_equal(candidate.extract_batch(items[::-1])[::-1], fused)
        np.testing.assert_array_equal(candidate.extract(*items[0]), fused[0])
        assert candidate.extract_batch([]).shape == (0, 6)
        with pytest.raises(ValueError, match="failed to decode"):
            candidate.extract_batch([(root / "absent.jpg", items[0][1])])
        np.testing.assert_array_equal(candidate.extract_batch(items), fused)
    finally:
        for ext in (reference, candidate):
            if ext._preprocess_pool:
                ext._preprocess_pool.shutdown()


def test_production_rejects_precision_or_fixed_batch_override(artifact):
    root, _, _ = artifact
    for kwargs in ({"precast_projections": True}, {"fusion_internal_batch_size": 8}):
        with pytest.raises(ValueError, match="original precision"):
            D082Extractor(root, "cpu", **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA required")
def test_singleton_graph_capture_partial_batches_errors_and_owned_outputs():
    class Toy:
        def features(self, batch):
            return batch.mean(dim=(-2, -1))

    candidate = object.__new__(D082Extractor)
    candidate.device = torch.device("cuda")
    candidate.fusion_internal_batch_size = 4
    candidate._preprocess_pool = None
    candidate._call_lock = threading.RLock()
    candidate._fusion_graphs = {}
    candidate.pipeline = True
    candidate.deferred = True
    candidate._fusion_components = [
        (
            GraphFeatures(Toy(), None),
            {"embed_dim": 3},
            None,
            std,
        )
        for std in (0.2, 0.4)
    ]

    def prepare(value):
        tensor = torch.tensor(
            [value, value * value, value + 1] if value else [0.0, 0.0, 0.0]
        ).reshape(3, 1, 1)
        return tensor, tensor + value

    candidate._decode_preprocess_fusion = prepare
    values = [1.0, 2.0, 4.0, 3.0, 5.0]
    candidate.extract_components_batch(values[:1])  # first graph capture from broadcast
    before = candidate.extract_components_batch(values)
    frozen = tuple(array.copy() for array in before)
    for size in (1, 2, 3, 4, 5, 17):
        inputs = (values * 4)[:size]
        actual = candidate.extract_components_batch(inputs)
        for index, value in enumerate(inputs):
            alone = candidate.extract_components_batch([value])
            for block, single in zip(actual, alone):
                np.testing.assert_array_equal(block[index : index + 1], single)
    reverse = candidate.extract_components_batch(values[::-1])
    for actual, expected in zip(reverse, before):
        np.testing.assert_array_equal(actual[::-1], expected)
    for value in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonzero"):
            candidate.extract_components_batch([value])
    after = candidate.extract_components_batch(values)
    for actual, expected, owned in zip(after, frozen, before):
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(owned, expected)
