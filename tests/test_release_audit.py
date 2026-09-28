import json
import pytest
import torch
import yaml

from scripts import audit_release
from scripts.audit_release import audit_weights


@pytest.fixture
def release(tmp_path):
    model = tmp_path / "final"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"weight-size-fixture")
    (model / "model_config.yaml").write_text(yaml.safe_dump({
        "name": "final_all_data", "train": {"folds": "all"},
    }))
    verifier = tmp_path / "verifier.pt"
    torch.save({"trained_on": "folds [0, 1, 2, 3, 4]", "arch": "dinov2_vits14", "input": 168}, verifier)
    return tmp_path, model, verifier


def test_weight_audit_counts_auxiliary_files_and_rejects_over_limit(release, monkeypatch):
    root, model, verifier = release
    (root / "extra.npz").write_bytes(b"1234")
    report = audit_weights(root, model, verifier, True)
    assert len(report["files"]) == 3
    assert report["total_weight_bytes"] == sum(
        (root / name).stat().st_size for name in report["files"]
    )
    monkeypatch.setattr(audit_release, "MAX_WEIGHT_BYTES", report["total_weight_bytes"] - 1)
    with pytest.raises(ValueError, match="exceed"):
        audit_weights(root, model, verifier, True)


def test_final_audit_rejects_fold0_embedder(release):
    root, model, verifier = release
    (model / "model_config.yaml").write_text(yaml.safe_dump({"train": {"folds": "all except 0"}}))
    with pytest.raises(ValueError, match="embedder must be trained on all"):
        audit_weights(root, model, verifier, True)


def test_final_audit_rejects_fold0_verifier(release):
    root, model, verifier = release
    torch.save({"trained_on": "folds [1, 2, 3, 4]"}, verifier)
    with pytest.raises(ValueError, match="verifier must be trained on all"):
        audit_weights(root, model, verifier, True)


def test_report_cannot_overwrite_audited_file(release):
    root, model, verifier = release
    report_path = model / "weights.safetensors"
    before = report_path.read_bytes()
    status = audit_release.main([
        "weights", "--root", str(root), "--model", str(model),
        "--verifier", str(verifier), "--report", str(report_path),
    ])
    assert status == 1
    assert report_path.read_bytes() == before


def test_fusion_audit_checks_all_members_and_extra_weights(release, monkeypatch):
    root, legacy, verifier = release
    fusion = root / "fusion"
    components = []
    for i in range(2):
        directory = fusion / "components" / str(i)
        directory.mkdir(parents=True)
        (directory / "model_config.yaml").write_text((legacy / "model_config.yaml").read_text())
        (directory / "weights.safetensors").write_bytes(b"component")
        components.append({"path": f"components/{i}", "similarity_std": 0.1})
    (fusion / "fusion_config.json").write_text(json.dumps({
        "schema_version": 1, "name": "fusion", "components": components,
        "candidate_policy": "gallery_z", "candidate_threshold": 6.0,
        "rerank_mode": "standardized",
    }))
    report = audit_weights(root, fusion, verifier, True)
    assert report["model_trained_on"] == ["all", "all"]
    assert len(report["files"]) == 4  # Includes the unrelated legacy weight!
    assert set(report["component_configs"]) == {"components/0", "components/1"}
    monkeypatch.setattr(audit_release, "MAX_WEIGHT_BYTES", report["total_weight_bytes"] - 1)
    with pytest.raises(ValueError, match="exceed"):
        audit_weights(root, fusion, verifier, True)
    monkeypatch.setattr(audit_release, "MAX_WEIGHT_BYTES", 2_000_000_000)
    (fusion / "components/1/model_config.yaml").write_text(yaml.safe_dump({"train": {"folds": "all except 1"}}))
    with pytest.raises(ValueError, match="embedder must be trained on all"):
        audit_weights(root, fusion, verifier, True)
