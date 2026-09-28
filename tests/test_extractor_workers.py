from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reid.infer.extractor import Extractor


def _stub_extractor(workers: int) -> Extractor:
    extractor = Extractor.__new__(Extractor)
    extractor.cfg = {"embed_dim": 2}
    extractor.device = torch.device("cpu")
    extractor.amp_dtype = None
    extractor.preprocess_workers = workers
    extractor._preprocess_pool = None
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        extractor._preprocess_pool = ThreadPoolExecutor(max_workers=workers)
    extractor._decode_preprocess = lambda item: torch.tensor([float(Path(item[0]).stem), 1.0])
    extractor.model = SimpleNamespace(features=lambda batch: batch)
    return extractor


@pytest.mark.parametrize("workers", [1, 4])
def test_extract_batch_preserves_input_order(workers):
    extractor = _stub_extractor(workers)
    items = [(Path(str(value)), (0.0, 0.0, 1.0, 1.0)) for value in (3, 1, 2)]

    result = extractor.extract_batch(items)

    assert result.dtype == np.float32
    expected = torch.nn.functional.normalize(
        torch.tensor([[3.0, 1.0], [1.0, 1.0], [2.0, 1.0]]), dim=1
    ).numpy()
    assert np.allclose(result, expected)
    if extractor._preprocess_pool is not None:
        extractor._preprocess_pool.shutdown()


def test_extractor_rejects_nonpositive_worker_count(tmp_path, monkeypatch):
    monkeypatch.setattr("reid.infer.extractor.load_config", lambda _: {"infer_dtype": "fp32"})
    monkeypatch.setattr("reid.infer.extractor.load", lambda *_: lambda batch: batch)

    with pytest.raises(ValueError, match="preprocess_workers must be positive"):
        Extractor(tmp_path, device="cpu", preprocess_workers=0)


def test_extractor_matches_oof_pooling_and_bypasses_nonidentity_bnneck():
    from reid.model import Embedder, pool_tokens

    class Backbone(torch.nn.Module):
        num_features = 2
        num_prefix_tokens = 1

        def forward_features(self, batch):
            return torch.stack([batch, batch * torch.tensor([2.0, 3.0])], dim=1)

    extractor = _stub_extractor(1)
    extractor.model = Embedder(Backbone(), pool="cls_avg", neck="bnneck").eval()
    extractor.cfg["embed_dim"] = 4
    with torch.no_grad():
        extractor.model.neck.running_mean.copy_(torch.tensor([0.1, 0.5, 0.7, 0.2]))
        extractor.model.neck.running_var.copy_(torch.tensor([1.0, 4.0, 2.0, 3.0]))
        batch = torch.tensor([[3.0, 1.0], [1.0, 1.0]])
        pooled = pool_tokens(extractor.model.tokens(batch), 1, "cls_avg")
        expected = torch.nn.functional.normalize(pooled, dim=1).numpy()
        necked = extractor.model(batch).numpy()
    assert not np.allclose(necked, expected)
    items = [(Path(str(i)), (0, 0, 1, 1)) for i in (3, 1)]
    actual = extractor.extract_batch(items)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, atol=1e-7)
    np.testing.assert_allclose(extractor.extract_batch(items[:1])[0], actual[0], atol=1e-7)
