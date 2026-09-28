"""Optional verifier metadata stays bound to its checkpoint across every stage."""
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import yaml

from scripts import prepare_fusion_artifact as prepare
from scripts.install_fusion_bundle import install_bundle, validate_verifier_provenance
from scripts.stage_fusion_release import stage_fusion


RELEASE = "https://github.com/Timur713/hack_falcon_2026/releases/tag/provenance-test"
URL = "https://github.com/Timur713/hack_falcon_2026/releases/download/provenance-test/pair.tar.gz"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    components = []
    for index in range(2):
        member = tmp_path / f"model{index}"
        member.mkdir()
        (member / "weights.safetensors").write_bytes(b"fixture-weights")
        (member / "model_config.yaml").write_text("train: {folds: all}\n")
        components.append(member)
    verifier = tmp_path / "source.pt"
    verifier.write_bytes(b"fixture-verifier")
    provenance = tmp_path / "source.provenance.yaml"
    provenance.write_text(yaml.safe_dump({"name": "fixture", "artifact": {"sha256": digest(verifier)}}))
    gallery = tmp_path / "gallery.csv"
    gallery.write_text("fixture-only-gallery")
    monkeypatch.setattr(prepare, "read_inputs", lambda *args: list(range(12)))
    monkeypatch.setattr(prepare, "Extractor", lambda *args: object())
    monkeypatch.setattr(prepare, "extract_rows", lambda *args: np.eye(12, dtype=np.float32))
    return dict(components=components, images=tmp_path, gallery=gallery, out=tmp_path / "bundle",
                verifier=verifier, verifier_provenance=provenance, candidate_threshold=6.6,
                threshold_source="D-065 / A-104", rerank_weight=0.15)


def repin(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["artifacts"] = {p.relative_to(bundle).as_posix(): {
        "bytes": p.stat().st_size, "sha256": digest(p),
    } for p in bundle.rglob("*") if p.is_file() and p.name != "manifest.json"}
    manifest["total_weight_bytes"] = sum(r["bytes"] for p, r in manifest["artifacts"].items()
                                         if p.endswith((".pt", ".safetensors")))
    (bundle / "manifest.json").write_text(json.dumps(manifest))


def install(bundle, destination):
    return install_bundle(bundle.parent, bundle.name, destination, digest(bundle / "manifest.json"), RELEASE)


def test_prepare_and_install_preserve_pinned_sidecar(inputs, tmp_path):
    result = prepare.prepare_bundle(**inputs)
    bundle = inputs["out"]
    sidecar = bundle / "verifier.provenance.yaml"
    assert sidecar.read_bytes() == inputs["verifier_provenance"].read_bytes()
    assert result["artifacts"][sidecar.name]["sha256"] == digest(sidecar)
    assert result["total_weight_bytes"] == sum(p.stat().st_size for p in (
        bundle / "verifier.pt", *(bundle / f"components/{i}/weights.safetensors" for i in range(2))))
    assert install(bundle, tmp_path / "installed")["total_weight_bytes"] == result["total_weight_bytes"]
    assert (tmp_path / "installed" / sidecar.name).read_bytes() == sidecar.read_bytes()


def test_sidecar_archive_bootstrap_is_offline_and_pinned(inputs, tmp_path, monkeypatch):
    prepare.prepare_bundle(**inputs)
    bundle = inputs["out"]
    archive = tmp_path / "pair.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.USTAR_FORMAT) as handle:
        for path in sorted(bundle.rglob("*")):
            handle.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    monkeypatch.setattr("scripts.stage_fusion_release._open_download", lambda *a: pytest.fail("network"))
    result = stage_fusion(tmp_path, "missing-bundle", archive.name, tmp_path / "installed",
                          digest(bundle / "manifest.json"), RELEASE, URL, digest(archive))
    assert result["bootstrap_source"] == "local_archive"
    assert (tmp_path / "installed/verifier.provenance.yaml").read_bytes() == inputs["verifier_provenance"].read_bytes()


@pytest.mark.parametrize("problem", ["orphan", "unlisted", "tampered", "wrong-checkpoint", "oversized"])
def test_installer_rejects_orphan_unlisted_or_tampered_sidecar(inputs, tmp_path, problem):
    prepare.prepare_bundle(**inputs)
    bundle = inputs["out"]
    sidecar = bundle / "verifier.provenance.yaml"
    if problem == "orphan":
        (bundle / "verifier.pt").unlink()
        repin(bundle)
    elif problem == "unlisted":
        manifest = json.loads((bundle / "manifest.json").read_text())
        del manifest["artifacts"][sidecar.name]
        (bundle / "manifest.json").write_text(json.dumps(manifest))
    elif problem == "oversized":
        sidecar.write_bytes(b"x" * (1024 * 1024 + 1))
        repin(bundle)
    else:
        sidecar.write_text(yaml.safe_dump({"artifact": {"sha256": "0" * 64}}))
        if problem == "wrong-checkpoint":
            repin(bundle)  # Even a freshly pinned sidecar cannot name another checkpoint.
    with pytest.raises(ValueError):
        install(bundle, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()


@pytest.mark.parametrize("problem", ["orphan", "wrong-checkpoint", "changed-copy"])
def test_preparer_checks_copied_sidecar_before_inference(inputs, monkeypatch, problem):
    if problem == "orphan":
        inputs["verifier"] = None
    elif problem == "wrong-checkpoint":
        inputs["verifier_provenance"].write_text(yaml.safe_dump({"artifact": {"sha256": "0" * 64}}))
    else:
        original = prepare.shutil.copyfile
        def copyfile(source, destination):
            result = original(source, destination)
            if Path(destination).name == "verifier.pt":
                Path(destination).write_bytes(b"different-checkpoint")
            return result
        monkeypatch.setattr(prepare.shutil, "copyfile", copyfile)
    monkeypatch.setattr(prepare, "extract_rows", lambda *a: pytest.fail("inference before validation"))
    with pytest.raises(ValueError, match="provenance"):
        prepare.prepare_bundle(**inputs)
    assert not inputs["out"].exists()


@pytest.mark.parametrize("data", [b"[]", b"artifact: []", b"artifact: {}", b"[invalid",
                                  b"artifact: {sha256: bad}", b"\xff", b"x" * (1024 * 1024 + 1)])
def test_invalid_or_oversized_provenance_fails_closed(tmp_path, data):
    sidecar = tmp_path / "invalid.yaml"
    sidecar.write_bytes(data)
    with pytest.raises(ValueError):
        validate_verifier_provenance(sidecar, "a" * 64)


def test_legacy_bundle_without_sidecar_remains_compatible(inputs, tmp_path):
    inputs["verifier_provenance"] = None
    prepare.prepare_bundle(**inputs)
    result = install(inputs["out"], tmp_path / "installed")
    assert result["component_count"] == 2
    assert not (tmp_path / "installed/verifier.provenance.yaml").exists()
