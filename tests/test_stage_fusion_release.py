"""Release transport/archive hardening without network, Docker, GPU or real weights."""
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from urllib.request import Request

import pytest

from scripts import stage_fusion_release as stage


RELEASE = "https://github.com/Timur713/hack_falcon_2026/releases/tag/fusion-test-v1"
URL = "https://github.com/Timur713/hack_falcon_2026/releases/download/fusion-test-v1/fusion.tar.gz"


def test_public_mirror_is_explicit_and_archive_must_match_repository():
    mirror_release = RELEASE.replace("hack_falcon_2026", "hf_2026")
    mirror_url = URL.replace("hack_falcon_2026", "hf_2026")
    assert stage._archive_url(mirror_url, mirror_release) == mirror_url
    with pytest.raises(ValueError):
        stage._archive_url(URL, mirror_release)
    with pytest.raises(ValueError):
        stage._archive_url(mirror_url, RELEASE)
    with pytest.raises(ValueError):
        stage._archive_url(URL.replace("Timur713", "untrusted"),
                           RELEASE.replace("Timur713", "untrusted"))
    with pytest.raises(ValueError):
        stage._archive_url(mirror_url + "?token=x", mirror_release)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    for index in range(2):
        member = root / "components" / str(index)
        member.mkdir(parents=True)
        (member / "model_config.yaml").write_text("train: {folds: all}\n")
        (member / "weights.safetensors").write_bytes(b"weights")
    (root / "verifier.pt").write_bytes(b"verifier")
    (root / "fusion_config.json").write_text(json.dumps({
        "schema_version": 1, "name": "fixture", "candidate_policy": "gallery_z",
        "candidate_threshold": 6.3, "rerank_mode": "standardized", "rerank_weight": 0.15,
        "components": [{"path": f"components/{i}", "similarity_std": 0.1} for i in range(2)],
    }))
    records = {p.relative_to(root).as_posix(): {
        "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    } for p in root.rglob("*") if p.is_file()}
    (root / "manifest.json").write_text(json.dumps({
        "kind": "fusion_artifact_preparation", "artifacts": records,
        "total_weight_bytes": sum(v["bytes"] for k, v in records.items()
                                  if k.endswith((".pt", ".safetensors"))),
    }))
    return root


def package(source, extra=()):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for path in sorted(source.rglob("*")):
            archive.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
        for name, kind, body in extra:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(body)
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "../../outside"
            archive.addfile(member, io.BytesIO(body))
    return output.getvalue()


def arguments(tmp_path, source, payload):
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    return {
        "cache_root": cache, "bundle": "artifacts/fusion_bundle",
        "archive_cache": "artifacts/fusion_bundle.tar.gz", "destination": tmp_path / "installed",
        "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "source_release_url": RELEASE, "archive_url": URL,
        "archive_sha256": hashlib.sha256(payload).hexdigest(), "require_verifier": True,
    }


def run_payload(tmp_path, source, payload, monkeypatch, **override):
    args = arguments(tmp_path, source, payload)
    args.update(override)
    monkeypatch.setattr(stage, "_open_download", lambda url: io.BytesIO(payload))
    return stage.stage_fusion(**args)


def assert_not_installed(tmp_path):
    assert not (tmp_path / "installed").exists()
    assert not list(tmp_path.glob(".installed-stage-*"))


def test_clean_clone_downloads_pinned_archive_and_installs_only_manifest(source, tmp_path, monkeypatch):
    payload = package(source)
    calls = []
    def download(url):
        calls.append(url)
        return io.BytesIO(payload)
    monkeypatch.setattr(stage, "_open_download", download)
    result = stage.stage_fusion(**arguments(tmp_path, source, payload))
    assert calls == [URL]
    assert result["bootstrap_source"] == "https_archive"
    assert result["archive_sha256_verified"] is True
    installed = tmp_path / "installed"
    assert json.loads((installed / "installation.json").read_text()) == result
    assert not (installed / "bundle.tar.gz").exists()
    for path in source.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (installed / path.relative_to(source)).read_bytes()
    assert not list(tmp_path.glob(".installed-stage-*"))


@pytest.mark.parametrize("cache_type", ["directory", "archive"])
def test_private_cache_needs_no_network(source, tmp_path, monkeypatch, cache_type):
    payload = package(source)
    args = arguments(tmp_path, source, payload)
    (args["cache_root"] / "artifacts").mkdir()
    if cache_type == "directory":
        source.rename(args["cache_root"] / args["bundle"])
        args.update(archive_url="", archive_sha256="")  # Backward-compatible local path.
    else:
        (args["cache_root"] / args["archive_cache"]).write_bytes(payload)
    monkeypatch.setattr(stage, "_open_download", lambda *a: pytest.fail("unexpected network"))
    result = stage.stage_fusion(**args)
    assert result["bootstrap_source"] == f"local_{cache_type}"
    assert result["archive_sha256_verified"] == (cache_type == "archive")


@pytest.mark.parametrize("cache_type", ["directory", "archive"])
def test_present_bad_cache_never_falls_back_to_network(source, tmp_path, monkeypatch, cache_type):
    args = arguments(tmp_path, source, package(source))
    (args["cache_root"] / "artifacts").mkdir()
    if cache_type == "directory":
        source.rename(args["cache_root"] / args["bundle"])
        (args["cache_root"] / args["bundle"] / "verifier.pt").write_bytes(b"tampered")
    else:
        (args["cache_root"] / args["archive_cache"]).write_bytes(b"tampered")
    monkeypatch.setattr(stage, "_open_download", lambda *a: pytest.fail("cache must fail closed"))
    with pytest.raises(ValueError, match="SHA-256|size"):
        stage.stage_fusion(**args)
    assert_not_installed(tmp_path)


@pytest.mark.parametrize("override", [
    {"archive_url": "", "archive_sha256": ""},
    {"archive_url": ""}, {"archive_sha256": ""},
    {"archive_sha256": "a" * 63}, {"manifest_sha256": ""},
    {"archive_url": URL + "?token=secret"},
    {"archive_url": URL.replace("https:", "http:")},
    {"archive_url": URL.replace("github.com", "secret@github.com")},
    {"archive_url": URL.replace("fusion-test-v1", "latest")},
    {"archive_url": URL.replace("fusion-test-v1", "different-tag")},
    {"archive_url": URL.replace("fusion.tar.gz", "../fusion.tar.gz")},
    {"bundle": "../escape"}, {"archive_cache": "/absolute"},
])
def test_invalid_pins_and_paths_fail_before_network(source, tmp_path, monkeypatch, override):
    args = arguments(tmp_path, source, package(source))
    args.update(override)
    monkeypatch.setattr(stage, "_open_download", lambda *a: pytest.fail("must reject before network"))
    with pytest.raises(ValueError):
        stage.stage_fusion(**args)
    assert_not_installed(tmp_path)


@pytest.mark.parametrize("path", ["artifacts", "artifacts/fusion_bundle", "artifacts/fusion_bundle.tar.gz"])
def test_symlink_cache_including_dangling_is_rejected(source, tmp_path, monkeypatch, path):
    args = arguments(tmp_path, source, package(source))
    link = args["cache_root"] / path
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(stage, "_open_download", lambda *a: pytest.fail("must reject symlink"))
    with pytest.raises(ValueError, match="symlink"):
        stage.stage_fusion(**args)
    assert_not_installed(tmp_path)


def test_archive_and_manifest_are_independently_verified(source, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="archive SHA-256"):
        run_payload(tmp_path, source, package(source), monkeypatch, archive_sha256="0" * 64)
    assert_not_installed(tmp_path)
    with pytest.raises(ValueError, match="manifest SHA-256"):
        run_payload(tmp_path, source, package(source), monkeypatch, manifest_sha256="0" * 64)
    assert_not_installed(tmp_path)


def test_signed_archive_cannot_override_inner_asset_hash(source, tmp_path, monkeypatch):
    (source / "verifier.pt").write_bytes(b"corrupt!")
    with pytest.raises(ValueError, match="asset size/SHA-256"):
        run_payload(tmp_path, source, package(source), monkeypatch)
    assert_not_installed(tmp_path)


def test_manifest_duplicate_keys_are_rejected_even_with_both_pins(source, tmp_path, monkeypatch):
    manifest = source / "manifest.json"
    manifest.write_text('{"kind":"fusion_artifact_preparation",' + manifest.read_text()[1:])
    with pytest.raises(ValueError, match="duplicate"):
        run_payload(tmp_path, source, package(source), monkeypatch)
    assert_not_installed(tmp_path)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "components/../escape", "./manifest.json",
                                  "components//0/weights.safetensors", "components\\escape", "secret.env"])
def test_archive_paths_and_extra_files_rejected(source, tmp_path, monkeypatch, name):
    payload = package(source, [(name, tarfile.REGTYPE, b"x")])
    with pytest.raises(ValueError):
        run_payload(tmp_path, source, payload, monkeypatch)
    assert_not_installed(tmp_path)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE,
                                  tarfile.FIFOTYPE, tarfile.GNUTYPE_SPARSE,
                                  tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME])
def test_archive_special_members_rejected_before_processing(source, tmp_path, monkeypatch, kind):
    payload = package(source, [("special", kind, b"")])
    with pytest.raises(ValueError, match="only regular"):
        run_payload(tmp_path, source, payload, monkeypatch)
    assert_not_installed(tmp_path)


def test_duplicate_names_and_unexpected_directories_rejected(source, tmp_path, monkeypatch):
    for name, kind in [("manifest.json", tarfile.REGTYPE), ("secrets", tarfile.DIRTYPE)]:
        payload = package(source, [(name, kind, b"" if kind == tarfile.DIRTYPE else b"x")])
        with pytest.raises(ValueError):
            run_payload(tmp_path, source, payload, monkeypatch)
        assert_not_installed(tmp_path)


@pytest.mark.parametrize("limit", ["MAX_ARCHIVE_BYTES", "MAX_MEMBERS", "MAX_METADATA_BYTES",
                                   "MAX_WEIGHT_BYTES", "MAX_UNPACKED_BYTES"])
def test_compressed_unpacked_member_and_weight_limits(source, tmp_path, monkeypatch, limit):
    monkeypatch.setattr(stage, limit, 5 if limit == "MAX_MEMBERS" else 10)
    with pytest.raises(ValueError, match="limit|members"):
        run_payload(tmp_path, source, package(source), monkeypatch)
    assert_not_installed(tmp_path)


def test_oversized_header_rejected_without_allocating_payload(source, tmp_path, monkeypatch):
    member = tarfile.TarInfo("verifier.pt")
    member.size = 2_000_000_001
    payload = gzip.compress(member.tobuf(format=tarfile.USTAR_FORMAT))
    with pytest.raises(ValueError, match="size limit"):
        run_payload(tmp_path, source, payload, monkeypatch)
    assert_not_installed(tmp_path)


@pytest.mark.parametrize("mutation", ["truncated", "tail", "bomb_tail", "checksum", "padding"])
def test_archive_structure_rejected(source, tmp_path, monkeypatch, mutation):
    raw = gzip.decompress(package(source))
    if mutation == "truncated":
        raw = raw[:100]
    elif mutation == "tail":
        raw += b"not another tar"
    elif mutation == "bomb_tail":
        raw += b"\0" * 50_000
    elif mutation == "checksum":
        raw = b"!" + raw[1:]
    else:
        member = tarfile.TarInfo("verifier.pt")
        member.size = 1
        raw = member.tobuf(format=tarfile.USTAR_FORMAT) + b"x!" + b"\0" * 511
    with pytest.raises((ValueError, tarfile.TarError)):
        run_payload(tmp_path, source, gzip.compress(raw), monkeypatch)
    assert_not_installed(tmp_path)


@pytest.mark.parametrize("target", ["http://release-assets.githubusercontent.com/file",
                                    "https://example.org/file", "https://secret@github.com/file",
                                    "https://github.com:444/file", "https://github.com/file#fragment"])
def test_redirect_guard_rejects_unsafe_targets(target):
    handler = stage._SafeRedirect()
    with pytest.raises(ValueError, match="allowlist"):
        handler.redirect_request(Request(URL), None, 302, "Found", {}, target)


def test_redirect_guard_permits_signed_https_github_asset():
    target = "https://release-assets.githubusercontent.com/id/file?signature=fixture"
    request = stage._SafeRedirect().redirect_request(Request(URL), None, 302, "Found", {}, target)
    assert request.full_url == target


def test_existing_destination_is_never_overwritten(source, tmp_path, monkeypatch):
    args = arguments(tmp_path, source, package(source))
    args["destination"].mkdir()
    sentinel = args["destination"] / "existing"
    sentinel.write_text("keep")
    monkeypatch.setattr(stage, "_open_download", lambda *a: pytest.fail("must reject first"))
    with pytest.raises(FileExistsError):
        stage.stage_fusion(**args)
    assert sentinel.read_text() == "keep"


def test_cli_accepts_local_archive_and_reports_missing_pins(source, tmp_path):
    payload = package(source)
    args = arguments(tmp_path, source, payload)
    archive = args["cache_root"] / args["archive_cache"]
    archive.parent.mkdir()
    archive.write_bytes(payload)
    command = [sys.executable, "-m", "scripts.stage_fusion_release",
               "--cache-root", str(args["cache_root"]),
               "--destination", str(args["destination"]),
               "--manifest-sha256", args["manifest_sha256"],
               "--source-release-url", RELEASE, "--require-verifier"]
    missing = subprocess.run(command, capture_output=True, text=True)
    assert missing.returncode == 1
    assert "pinned archive URL and SHA-256 are required" in missing.stderr
    assert "Traceback" not in missing.stderr
    complete = subprocess.run(command + ["--archive-url", URL, "--archive-sha256", args["archive_sha256"]],
                              capture_output=True, text=True)
    assert complete.returncode == 0, complete.stderr
    assert "Verified 2 components" in complete.stdout and "local_archive" in complete.stdout


def test_documented_system_tar_command_produces_accepted_archive(source, tmp_path, monkeypatch):
    tar = shutil.which("tar")
    if tar is None:
        pytest.skip("system tar is not installed")
    archive = tmp_path / "system-tar-bundle.tar.gz"
    subprocess.run([
        tar, "--format=ustar", "-czf", str(archive), "-C", str(source),
        "manifest.json", "fusion_config.json", "components", "verifier.pt",
    ], env={**os.environ, "COPYFILE_DISABLE": "1"}, check=True, capture_output=True)
    payload = archive.read_bytes()
    args = arguments(tmp_path, source, payload)
    cached = args["cache_root"] / args["archive_cache"]
    cached.parent.mkdir()
    archive.rename(cached)
    monkeypatch.setattr(stage, "_open_download", lambda *a: pytest.fail("unexpected network"))
    result = stage.stage_fusion(**args)
    assert result["bootstrap_source"] == "local_archive"
    assert result["component_count"] == 2
    assert result["archive_sha256_verified"] is True
