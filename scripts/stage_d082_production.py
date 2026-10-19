"""Pinned build-time D082 bootstrap. No credentials and no final-PASS claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tarfile
import tempfile
from urllib.parse import urlsplit
import zlib

from reid.infer.d082_production import (
    DIRECTORIES,
    FILES,
    MANIFEST,
    digest,
    load_production_runtime,
)
from scripts.stage_fusion_release import _cache_path, _extract_archive, _stage_archive


def release_urls(source_release_url, archive_url):
    """Private working repo or the already established clean public copy only."""
    release = urlsplit(source_release_url)
    match = re.fullmatch(
        r"/Timur713/(hack_falcon_2026|hf_2026)/releases/tag/([A-Za-z0-9][A-Za-z0-9._-]{0,127})",
        release.path,
    )
    if (
        release.scheme != "https"
        or release.netloc != "github.com"
        or release.query
        or release.fragment
        or match is None
        or match[2].lower() == "latest"
    ):
        raise ValueError("D082 source must be a fixed HTTPS team GitHub release")
    prefix = source_release_url.replace("/releases/tag/", "/releases/download/") + "/"
    if (
        not archive_url.startswith(prefix)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.tar\.gz", archive_url[len(prefix) :]
        )
        is None
    ):
        raise ValueError(
            "D082 archive must be a fixed .tar.gz asset in the declared release"
        )


def stage_d082(
    cache_root: Path,
    archive_cache: str,
    destination: Path,
    manifest_sha256: str,
    source_release_url: str,
    archive_url: str,
    archive_sha256: str,
):
    digest(manifest_sha256)
    digest(archive_sha256)
    release_urls(source_release_url, archive_url)
    archive = _cache_path(cache_root, archive_cache)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite D082 installation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".d082-stage-", dir=destination.parent
    ) as temp:
        work = Path(temp)
        mode = _stage_archive(
            archive, work / "bundle.tar.gz", archive_url, archive_sha256
        )
        installed = work / "installed"
        installed.mkdir()
        bundle = installed / "bundle"
        _extract_archive(
            work / "bundle.tar.gz",
            bundle,
            allowed_files=FILES | {MANIFEST},
            allowed_directories=DIRECTORIES,
            manifest_name=MANIFEST,
        )
        report = load_production_runtime(bundle, manifest_sha256).identity
        report = dict(
            report,
            source_release_url=source_release_url,
            archive_url=archive_url,
            archive_sha256=archive_sha256,
            bootstrap_source=mode,
            status="INSTALLED_CANDIDATE_NOT_FINAL_ACCEPTANCE",
        )
        (installed / "installation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        (installed / "manifest.sha256").write_text(manifest_sha256 + "\n")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("D082 installation appeared during staging")
        installed.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--archive-cache", default="artifacts/d082_production_bundle.tar.gz"
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-release-url", required=True)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--archive-sha256", required=True)
    try:
        report = stage_d082(**vars(parser.parse_args()))
    except (
        OSError,
        ValueError,
        RuntimeError,
        tarfile.TarError,
        EOFError,
        zlib.error,
    ) as exc:
        message = (
            "archive I/O failed; check cache/public access"
            if isinstance(exc, OSError)
            else str(exc)
        )
        parser.exit(1, f"D082 BOOTSTRAP FAILED: {message}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
