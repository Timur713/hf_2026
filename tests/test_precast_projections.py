"""Type/selection tests only: CUDA numeric equality and speed need real GPU runs."""
import pytest
import torch

from reid.infer.extractor import Extractor, _precast_projection_parameters


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = torch.nn.Conv2d(3, 4, 2)
        self.projection = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.bnneck = torch.nn.BatchNorm1d(4)
        self.embedding = torch.nn.Embedding(3, 4)
        self.cls_token = torch.nn.Parameter(torch.ones(1, 1, 4))
        self.pos_embed = torch.nn.Parameter(torch.ones(1, 2, 4))
        self.register_buffer("float_buffer", torch.ones(3))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_precast_selects_only_linear_conv_parameters_and_preserves_other_tensors(dtype):
    model = ToyModel().eval().requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    selected = {"patch.weight", "patch.bias", "projection.weight", "projection.bias"}
    result = _precast_projection_parameters(model, dtype)
    assert result["modules_cast"] == 2 and result["parameters_cast"] == 4
    assert result["parameter_bytes_before"] == sum(before[k].numel() * 4 for k in selected)
    assert result["parameter_bytes_after"] == result["parameter_bytes_before"] // 2
    for name, actual in model.state_dict().items():
        if name in selected:
            assert actual.dtype == dtype
            torch.testing.assert_close(actual, before[name].to(dtype), rtol=0, atol=0)
        else:
            assert actual.dtype == before[name].dtype
            torch.testing.assert_close(actual, before[name], rtol=0, atol=0)
    assert all(not p.requires_grad for p in model.parameters())
    assert _precast_projection_parameters(model, dtype)["parameters_cast"] == 0


@pytest.mark.parametrize("state", ["training", "trainable", "wrong_dtype", "protected_alias"])
def test_unsafe_cast_rejected_before_mutation(state):
    model = ToyModel().eval().requires_grad_(False)
    dtype = torch.float16
    if state == "training":
        model.norm.train()
    elif state == "trainable":
        model.projection.weight.requires_grad_(True)
    elif state == "wrong_dtype":
        dtype = torch.float32
    else:
        model.norm.weight = model.projection.bias
    with pytest.raises(ValueError):
        _precast_projection_parameters(model, dtype)
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("device,infer_dtype,cast", [
    ("cpu", "fp16", False), ("cuda", "fp32", False),
    ("cuda", "fp16", True), ("cuda", "bf16", True),
])
def test_single_init_is_opt_in_cuda_amp_only(tmp_path, monkeypatch, enabled, device, infer_dtype, cast):
    model = ToyModel().eval()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("reid.infer.extractor.load", lambda *_: model)
    monkeypatch.setattr("reid.infer.extractor.load_config", lambda *_: {"infer_dtype": infer_dtype})
    # CUDA is mocked only for initialization/branch selection; all fixture tensors
    # stay on CPU and this test makes no inference-equivalence claim.
    extractor = Extractor(tmp_path, device, preprocess_workers=1, precast_projections=enabled)
    active = enabled and cast
    expected = (torch.float16 if infer_dtype == "fp16" else torch.bfloat16) if active else torch.float32
    assert model.projection.weight.dtype == expected
    assert model.norm.weight.dtype == torch.float32
    assert model.cls_token.dtype == torch.float32
    assert extractor.precast_projection_stats["requested"] is enabled
    assert extractor.precast_projection_stats["applied"] is active
    assert extractor.precast_projection_stats["modules_cast"] == (2 if active else 0)
    assert model.projection.weight.requires_grad is (not active)


def test_fusion_uses_each_components_amp_dtype(tmp_path, monkeypatch):
    (tmp_path / "fusion_config.json").write_text("{}")
    models = [ToyModel(), ToyModel(), ToyModel()]
    dtypes = ["fp16", "bf16", "fp32"]
    manifest = {"components": [{"path": str(i), "similarity_std": .1} for i in range(3)]}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("reid.infer.extractor.load_fusion_config", lambda *_: manifest)
    monkeypatch.setattr("reid.infer.extractor.load", lambda path, _: models[int(path.rsplit("/", 1)[1])])
    monkeypatch.setattr("reid.infer.extractor.load_config", lambda path: {
        "infer_dtype": dtypes[int(path.rsplit("/", 1)[1])], "embed_dim": 4, "input_size": [8, 8],
    })
    extractor = Extractor(tmp_path, "cuda", preprocess_workers=1, precast_projections=True)
    assert [m.projection.weight.dtype for m in models] == [torch.float16, torch.bfloat16, torch.float32]
    assert all(m.norm.weight.dtype == torch.float32 for m in models)
    assert all(m.bnneck.weight.dtype == torch.float32 for m in models)
    assert all(m.embedding.weight.dtype == torch.float32 for m in models)
    assert extractor.precast_projection_stats["eligible_components"] == 2
    assert extractor.precast_projection_stats["modules_cast"] == 4
    assert extractor.precast_projection_stats["parameters_cast"] == 8
