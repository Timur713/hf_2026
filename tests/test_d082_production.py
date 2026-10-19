"""Production boundary tests: pin enforcement before model loading."""

import hashlib
import io
from pathlib import Path

import pytest

from reid.infer import d082_entrypoint as entry
from reid.infer.d082_production import check_tree, digest, metadata
from scripts.hashing import file_digest
from scripts.stage_d082_production import release_urls


@pytest.mark.parametrize("option", sorted(entry.FORBIDDEN))
@pytest.mark.parametrize("inline", [False, True])
def test_baked_recipe_cannot_be_overridden(option, inline):
    args = [option + "=other"] if inline else [option, "other"]
    assert entry.main(args) == 2


def test_entrypoint_forwards_baked_pin(tmp_path, monkeypatch):
    pin = "a" * 64
    (tmp_path / "manifest.sha256").write_text(pin + "\n")
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(entry.run, "main", lambda args: calls.append(args) or 0)
    assert entry.main(["--images", "/data/images"]) == 0
    assert calls == [
        [
            "--images",
            "/data/images",
            "--model",
            str(tmp_path / "bundle"),
            "--verifier",
            str(tmp_path / "bundle/verifier.pt"),
            "--d082-production-manifest-sha256",
            pin,
        ]
    ]


@pytest.mark.parametrize("content", [None, "invalid", "A" * 64])
def test_missing_or_invalid_baked_pin_fails_before_inference(
    tmp_path, monkeypatch, content
):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    if content is not None:
        (tmp_path / "manifest.sha256").write_text(content)
    monkeypatch.setattr(entry.run, "main", lambda args: pytest.fail("must not infer"))
    assert entry.main([]) == 1


@pytest.mark.parametrize("raw", ['{"a": 1, "a": 2}', '{"a": NaN}', "[1, 2]"])
def test_production_metadata_rejects_ambiguous_json(tmp_path, raw):
    path = tmp_path / "manifest.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        metadata(path)


def test_production_tree_rejects_symlink(tmp_path):
    root = tmp_path / "bundle"
    root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        check_tree(root)


@pytest.mark.parametrize("value", ["a" * 63, "A" * 64, None, 123])
def test_manifest_pin_requires_full_lowercase_sha(value):
    with pytest.raises(ValueError):
        digest(value)


@pytest.mark.parametrize("repo", ["hf_2026", "hack_falcon_2026"])
def test_release_allows_only_matching_fixed_team_assets(repo):
    release = f"https://github.com/Timur713/{repo}/releases/tag/d082-r2"
    archive = release.replace("/tag/", "/download/") + "/bundle.tar.gz"
    release_urls(release, archive)
    with pytest.raises(ValueError):
        release_urls(release, archive.replace("Timur713", "other"))
    with pytest.raises(ValueError):
        release_urls(release.replace("d082-r2", "latest"), archive)


@pytest.mark.parametrize("payload", [b"", b"abc", b"x" * (2 * 1024 * 1024 + 17)])
def test_streaming_hash_works_without_python311_api(payload, monkeypatch):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert (
        file_digest(io.BytesIO(payload), "sha256").hexdigest()
        == hashlib.sha256(payload).hexdigest()
    )


def test_benchmark_source_binding_matches_checkout():
    from scripts.bench_final_candidate import source_binding

    assert source_binding(Path("."))["verified_files_sha256"]


def test_smoke_rejects_partial_production_instead_of_falling_back(
    tmp_path, monkeypatch
):
    from reid.infer.smoke import _resolve_model

    monkeypatch.chdir(tmp_path)
    Path("artifacts/models/d082").mkdir(parents=True)
    Path("artifacts/models/final").mkdir()
    with pytest.raises(OSError):
        _resolve_model(None)


def test_smoke_validates_production_pin_before_using_default(tmp_path, monkeypatch):
    from reid.infer import d082_production
    from reid.infer.smoke import _resolve_model

    monkeypatch.chdir(tmp_path)
    root = Path("artifacts/models/d082")
    root.mkdir(parents=True)
    (root / "manifest.sha256").write_text("a" * 64)
    calls = []
    monkeypatch.setattr(
        d082_production, "inspect_bundle", lambda *args: calls.append(args)
    )
    assert _resolve_model(None) == root / "bundle"
    assert calls == [(root / "bundle", "a" * 64)]
