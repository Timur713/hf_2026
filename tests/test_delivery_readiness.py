import json
import shutil
import subprocess

import numpy as np
import pytest
import yaml

from scripts.check_delivery_readiness import (
    build_report, docker_defaults, scan_secrets, secret_types, upstream_provenance,
    verifier_provenance,
)
from scripts.audit_release import file_record


def init_repo(path):
    if shutil.which("git") is None:
        pytest.skip("Git-history audit is a host-side check; minimal inference image has no git")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.org"], check=True)
    return path


def commit(path):
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)


def test_secret_scan_does_not_print_values_and_catches_removed_history(tmp_path):
    root = init_repo(tmp_path)
    credential = "synthetic" + "-access" + "-value"
    (root / "notes.md").write_text("Password: `" + credential + "`\n")
    commit(root)
    (root / "notes.md").write_text("Access removed; contact researcher@example.org.\n")
    commit(root)
    assert not scan_secrets(root, False)["findings"]
    report = scan_secrets(root, True)
    assert report["findings"] == [{"scope": "history", "path": "notes.md", "type": "credential_assignment"}]
    assert credential not in json.dumps(report)
    assert secret_types(b"Contact: jane@example.org; phone +123456789") == []


def test_secret_scan_catches_token_and_key_without_binary_false_positive():
    token = "ghp_" + "aBcDeFgHiJ" * 4
    assert secret_types(token.encode()) == ["github_token"]
    assert secret_types(b"\0" + token.encode()) == []
    assert secret_types(("-----BEGIN " + "OPENSSH PRIVATE KEY-----").encode()) == ["private_key"]


def runtime_files(root, fusion=False):
    (root / "Dockerfile").write_text("FROM base AS fusion\nFROM base AS legacy\n")
    service = {"build": {"context": "."}, "network_mode": "none",
               "command": ["--model", "/app/artifacts/models/final"]}
    if fusion:
        service["build"]["target"] = "fusion"
        service["command"][1] = "${MODEL_DIR:-/app/artifacts/models/fusion}"
    (root / "docker-compose.yml").write_text(yaml.safe_dump({"services": {"reid": service}}))


def test_legacy_and_explicit_fusion_defaults(tmp_path):
    runtime_files(tmp_path)
    assert docker_defaults(tmp_path)["uses_fusion"] is False
    runtime_files(tmp_path, fusion=True)
    assert docker_defaults(tmp_path)["uses_fusion"] is True


def test_final_alias_and_explicit_pair_compose_default(tmp_path):
    runtime_files(tmp_path, fusion=True)
    (tmp_path / "Dockerfile").write_text(
        "FROM base AS fusion\nFROM base AS legacy\nFROM fusion AS final\n")
    actual = docker_defaults(tmp_path)
    assert actual["docker_last_stage"] == "final"
    assert actual["compose_target"] == "fusion"
    assert actual["uses_fusion"] is True
    assert actual["offline"] is True


def test_submission_receipt_mismatch_and_wrong_dimension_fail_closed(tmp_path):
    root = init_repo(tmp_path)
    runtime_files(root, fusion=True)
    out = root / "submission"
    out.mkdir()
    query, gallery = root / "query.csv", root / "gallery.csv"
    query.write_text("image_id\nq\n")
    gallery.write_text("image_id\n" + "\n".join(f"g{i}" for i in range(10)) + "\n")
    (out / "submission.csv").write_text("q," + ",".join(f"g{i}" for i in range(10)) + "\n")
    (out / "candidates.csv").write_text("query_id,gallery_id,confidence\nq,g0,0.9\n")
    np.save(out / "embeddings.npy", np.zeros((11, 2), dtype=np.float32))
    receipt = root / "accepted.json"
    receipt.write_text(json.dumps({"validated": True, "files": {}, "inputs": {}}))
    commit(root)
    report = build_report(root, query=query, gallery=gallery, expected_dim=3, acceptance_c6=receipt)
    states = {check["id"]: check["status"] for check in report["checks"]}
    assert states["submission_c6_and_dimension"] == "blocked"
    assert states["submission_matches_acceptance"] == "blocked"
    assert report["final_acceptance"] is False


def test_upstream_metadata_missing_and_component_escape(tmp_path):
    component = tmp_path / "components" / "0"
    component.mkdir(parents=True)
    (component / "model_config.yaml").write_text(yaml.safe_dump({"pretrained": True, "train": {"folds": "all"}}))
    config = tmp_path / "fusion_config.json"
    config.write_text(json.dumps({"components": [{"path": "components/0"}]}))
    assert upstream_provenance(tmp_path)["components"][0]["missing_upstream_metadata"] == ["url", "revision", "sha256"]
    config.write_text(json.dumps({"components": [{"path": "../escape"}]}))
    with pytest.raises(ValueError, match="escapes"):
        upstream_provenance(tmp_path)
    config.write_text(json.dumps({"components": []}))
    with pytest.raises(ValueError, match="nonempty"):
        upstream_provenance(tmp_path)


def verifier_sidecar(root):
    checkpoint = root / "verifier.pt"
    checkpoint.write_bytes(b"synthetic checkpoint: never unpickled")
    sidecar = {
        "name": "v_fus_all_frz_input392",
        "artifact": {"sha256": file_record(checkpoint)["sha256"],
                     "release_url": "https://example.org/releases/v1/verifier.pt"},
        "pretrained": {"author": "Research team", "revision": "b" * 40,
                       "url": "https://example.org/model/tree/" + "b" * 40,
                       "sha256": "a" * 64},
        "train": {"data": "all five train folds"},
    }
    (root / "verifier.provenance.yaml").write_text(yaml.safe_dump(sidecar))
    return sidecar


def test_verifier_metadata_and_checkpoint_sha_binding(tmp_path):
    verifier_sidecar(tmp_path)
    actual = verifier_provenance(tmp_path)
    assert not actual["missing_upstream_metadata"]
    assert not actual["missing_artifact_metadata"]
    assert actual["checkpoint_sha256_matches"] is True
    assert actual["all_data_training_declared"] is True
    assert actual["checkpoint_integrity"] == file_record(tmp_path / "verifier.pt")
    (tmp_path / "verifier.pt").write_bytes(b"different checkpoint")
    assert verifier_provenance(tmp_path)["checkpoint_sha256_matches"] is False


@pytest.mark.parametrize("value,field,missing", [
    (True, "pretrained", "author"),
    ({"author": "Team", "url": "https://example.org/latest", "revision": "latest", "sha256": "a" * 64},
     "pretrained", "immutable_revision"),
    ({"author": "Team", "url": "https://example.org/v1", "revision": "v1", "sha256": "bad"},
     "pretrained", "valid_sha256"),
    ({"release_url": "https://example.org/v1", "sha256": "bad"}, "artifact", "artifact_sha256"),
])
def test_verifier_incomplete_metadata_is_not_complete(tmp_path, value, field, missing):
    data = verifier_sidecar(tmp_path)
    data[field] = value
    (tmp_path / "verifier.provenance.yaml").write_text(yaml.safe_dump(data))
    actual = verifier_provenance(tmp_path)
    assert missing in actual["missing_upstream_metadata"] + actual["missing_artifact_metadata"]


def test_verifier_missing_malformed_symlink_and_oof_metadata(tmp_path):
    assert verifier_provenance(tmp_path)["missing_artifact_metadata"] == [
        "regular_verifier.provenance.yaml", "regular_verifier.pt"]
    data = verifier_sidecar(tmp_path)
    data["train"] = {"folds": "all except 0"}
    path = tmp_path / "verifier.provenance.yaml"
    path.write_text(yaml.safe_dump(data))
    assert verifier_provenance(tmp_path)["all_data_training_declared"] is False
    path.write_text("pretrained: [invalid")
    assert "valid_sidecar_yaml" in verifier_provenance(tmp_path)["missing_artifact_metadata"]
    path.unlink()
    path.symlink_to(tmp_path / "verifier.pt")
    assert "regular_verifier.provenance.yaml" in verifier_provenance(tmp_path)["missing_artifact_metadata"]


def test_readiness_uses_pair_a104_and_preserves_cpu_limits(tmp_path):
    root = init_repo(tmp_path)
    runtime_files(root, fusion=True)
    bundle = root / "bundle"
    component = bundle / "components" / "0"
    component.mkdir(parents=True)
    (component / "model_config.yaml").write_text(yaml.safe_dump({
        "pretrained": {"url": "https://example.org/v1", "revision": "v1", "sha256": "a" * 64},
        "train": {"folds": "all"},
    }))
    (bundle / "fusion_config.json").write_text(json.dumps({"components": [{"path": "components/0"}]}))
    verifier_sidecar(bundle)
    for fold in (0, 1):
        path = root / f"docs/evidence/quality_fold{fold}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "dumps": [{"path": "hires392"}, {"path": "div_clip"}],
            "logits": {"path": f"pair_logits_f{fold}.npz"},
            "policy": {"rerank_weight": 0.15, "candidate_threshold": 6.6},
            "training_provenance_verified": False,
        }))
    commit(root)
    report = build_report(root, bundle=bundle)
    checks = {check["id"]: check for check in report["checks"]}
    assert checks["component_provenance"]["status"] == "pass"
    assert checks["verifier_provenance"]["status"] == "pass"
    for fold in (0, 1):
        evidence = checks[f"a104_fold{fold}_evidence"]
        assert evidence["status"] == "review"
        assert evidence["evidence"]["policy"]["candidate_threshold"] == 6.6
        assert "C4" in evidence["evidence"]["scope"]
        assert "full-frame/GPU acceptance" in evidence["evidence"]["scope"]
    assert not any(key.startswith("a103_") for key in checks)
    assert "SigLIP" not in json.dumps(report)
    assert report["final_acceptance"] is False
    assert checks["public_repo_credentials"]["status"] == "missing"
    assert checks["official_extractor"]["status"] == "missing"
    (bundle / "verifier.pt").write_bytes(b"changed")
    changed = build_report(root, bundle=bundle)
    assert next(c for c in changed["checks"] if c["id"] == "verifier_provenance")["status"] == "blocked"
