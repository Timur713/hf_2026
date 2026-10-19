"""Local installer integrity; no Docker daemon, GPU, download or real weights."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.install_fusion_bundle import install_bundle


RELEASE = "https://github.com/Timur713/hack_falcon_2026/releases/tag/fusion-oof-v0"


def write_manifest(bundle):
    records = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            data = path.read_bytes()
            records[path.relative_to(bundle).as_posix()] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    audit = {
        "kind": "fusion_artifact_preparation",
        "artifacts": records,
        "total_weight_bytes": sum(
            r["bytes"]
            for p, r in records.items()
            if p.endswith((".pt", ".safetensors"))
        ),
    }
    return save_manifest(bundle, audit)


def save_manifest(bundle, audit):
    data = (json.dumps(audit) + "\n").encode()
    (bundle / "manifest.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "cache" / "artifacts" / "fusion_bundle"
    for i in range(2):
        member = source / "components" / str(i)
        member.mkdir(parents=True)
        (member / "weights.safetensors").write_bytes(f"weight-{i}".encode())
        (member / "model_config.yaml").write_text("train: {folds: all}\n")
    (source / "verifier.pt").write_bytes(b"test-only-verifier")
    (source / "fusion_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test",
                "candidate_policy": "gallery_z",
                "candidate_threshold": 6.0,
                "rerank_mode": "standardized",
                "rerank_weight": 0.15,
                "components": [
                    {"path": f"components/{i}", "similarity_std": 0.1 + i / 10}
                    for i in range(2)
                ],
            }
        )
    )
    write_manifest(source)
    return source


def install(bundle, digest=None, **kwargs):
    return install_bundle(
        bundle.parents[1],
        "artifacts/fusion_bundle",
        bundle.parents[2] / "installed",
        digest or hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
        RELEASE,
        **kwargs,
    )


def test_install_only_pinned_files_and_preserve_manifest(bundle):
    result = install(bundle, require_verifier=True)
    dest = bundle.parents[2] / "installed"
    assert result["component_count"] == 2
    assert result["source_release_url_verified_online"] is False
    for path in bundle.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (dest / path.relative_to(bundle)).read_bytes()
    assert (
        json.loads((dest / "fusion_config.json").read_text())["rerank_weight"] == 0.15
    )
    assert set(
        p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()
    ) == {
        p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()
    } | {"installation.json"}
    with pytest.raises(FileExistsError, match="overwrite"):
        install(bundle)


@pytest.mark.parametrize(
    "mutation", ["manifest", "weights", "config", "missing", "extra", "symlink"]
)
def test_mutation_fails_without_installing(bundle, mutation):
    digest = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    target = bundle / "components/0/weights.safetensors"
    if mutation == "manifest":
        (bundle / "manifest.json").write_text("{}")
    elif mutation == "weights":
        target.write_bytes(b"tampered")
    elif mutation == "config":
        (bundle / "fusion_config.json").write_text("{}")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (bundle / "secret.env").write_text("must-not-copy")
    else:
        target.unlink()
        target.symlink_to(bundle / "verifier.pt")
    with pytest.raises(ValueError):
        install(bundle, digest)
    assert not (bundle.parents[2] / "installed").exists()


@pytest.mark.parametrize(
    "bad_name",
    ["../outside", "/absolute", "components/0/../weights.safetensors", "secret.env"],
)
def test_manifest_cannot_expand_copy_allowlist(bundle, bad_name):
    audit = json.loads((bundle / "manifest.json").read_text())
    audit["artifacts"][bad_name] = {"bytes": 3, "sha256": "a" * 64}
    digest = save_manifest(bundle, audit)
    with pytest.raises(ValueError):
        install(bundle, digest)


def test_budget_and_total_are_checked_before_copy(bundle):
    audit = json.loads((bundle / "manifest.json").read_text())
    audit["total_weight_bytes"] += 1
    with pytest.raises(ValueError, match="total_weight_bytes"):
        install(bundle, save_manifest(bundle, audit))
    audit["artifacts"]["verifier.pt"]["bytes"] = 2_000_000_001
    with pytest.raises(ValueError, match="exceed 2 GB"):
        install(bundle, save_manifest(bundle, audit))


def test_unused_component_rejected(bundle):
    member = bundle / "components/2"
    member.mkdir()
    (member / "weights.safetensors").write_bytes(b"not-selected")
    (member / "model_config.yaml").write_text("{}")
    with pytest.raises(ValueError, match="not selected"):
        install(bundle, write_manifest(bundle))


def test_verifier_required_by_docker_but_optional_for_installer(bundle):
    (bundle / "verifier.pt").unlink()
    write_manifest(bundle)
    with pytest.raises(ValueError, match="requires verifier"):
        install(bundle, require_verifier=True)
    assert install(bundle)["component_count"] == 2


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/Timur713/hack_falcon_2026/releases/tag/v1",
        "https://github.com/Timur713/hack_falcon_2026/releases/latest",
        RELEASE + "?token=do-not-leak",
        "https://token@github.com/Timur713/hack_falcon_2026/releases/tag/v1",
    ],
)
def test_release_annotation_is_fixed_and_credentials_free(bundle, url):
    with pytest.raises(ValueError, match="fixed HTTPS tag"):
        install_bundle(
            bundle.parents[1],
            "artifacts/fusion_bundle",
            bundle.parents[2] / "installed",
            "a" * 64,
            url,
        )


def test_relative_source_and_symlink_directory_rejected(bundle):
    with pytest.raises(ValueError, match="relative path"):
        install_bundle(
            bundle.parents[1],
            "../escape",
            bundle.parents[2] / "installed",
            "a" * 64,
            RELEASE,
        )
    link = bundle.parent / "linked_bundle"
    link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        install_bundle(
            bundle.parents[1],
            "artifacts/linked_bundle",
            bundle.parents[2] / "installed",
            "a" * 64,
            RELEASE,
        )


def test_docker_fusion_is_default_and_legacy_is_explicit():
    import re

    dockerfile = Path("Dockerfile").read_text()
    assert "FROM runtime-base AS fusion" in dockerfile
    assert "FROM runtime-base AS legacy\n" in dockerfile
    assert dockerfile.rsplit("FROM ", 1)[1].strip() == "d082-production AS final"
    fusion = dockerfile.split("FROM runtime-base AS fusion", 1)[1].split(
        "FROM runtime-base AS legacy", 1
    )[0]
    assert "RUN --mount=type=bind" in fusion
    assert "scripts/stage_fusion_release.py" in fusion
    assert '--archive-url "${FUSION_ARCHIVE_URL}"' in fusion
    assert '--archive-sha256 "${FUSION_ARCHIVE_SHA256}"' in fusion
    assert re.search(
        r"^ARG FUSION_ARCHIVE_URL=https://github\.com/Timur713/hf_2026/releases/download/[^/\s]+/[^\s]+\.tar\.gz$",
        fusion,
        re.M,
    )
    assert re.search(r"^ARG FUSION_ARCHIVE_SHA256=[a-f0-9]{64}$", fusion, re.M)
    assert re.search(r"^ARG FUSION_MANIFEST_SHA256=[a-f0-9]{64}$", fusion, re.M)
    assert "/latest/" not in fusion
    assert '--manifest-sha256 "${FUSION_MANIFEST_SHA256}"' in fusion
    assert "--require-verifier" in fusion
    assert "--require-all-data" in fusion
    assert '"--model", "/app/artifacts/models/fusion"' in fusion
    assert '"--verifier", "/app/artifacts/models/fusion/verifier.pt"' in fusion
    assert "ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in fusion
    assert "COPY artifacts" not in dockerfile
    assert "!artifacts/fusion_bundle/**" in Path(".dockerignore").read_text()
