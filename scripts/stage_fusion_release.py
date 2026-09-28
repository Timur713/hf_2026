"""Build-time fusion bootstrap: pinned local cache or public HTTPS release archive.

No credentials or runtime downloads. A present but invalid cache fails closed;
it never triggers a network fallback. Both archive and manifest are trust roots.
Only a small, regular-file-only USTAR tar.gz bundle layout is supported.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tarfile
import tempfile
import zlib
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

from scripts.install_fusion_bundle import (
    ASSET_PATH, BUNDLE_FILES, MAX_MANIFEST_BYTES, MAX_WEIGHT_BYTES,
    _relative, _release_url, _sha, install_bundle,
)


MAX_METADATA_BYTES = MAX_MANIFEST_BYTES
MAX_UNPACKED_BYTES = MAX_WEIGHT_BYTES + 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = MAX_UNPACKED_BYTES + 16 * 1024 * 1024
MAX_MEMBERS = 256
CHUNK_BYTES = 1024 * 1024
DOWNLOAD_HOSTS = {"github.com", "release-assets.githubusercontent.com",
                  "objects.githubusercontent.com", "github-releases.githubusercontent.com"}


def _archive_url(value: str, release: str) -> str:
    _release_url(release)  # Both private provenance and the exact public mirror only.
    tag = release.rsplit("/", 1)[-1]
    prefix = release.rsplit("/releases/tag/", 1)[0] + "/releases/download/" + tag + "/"
    filename = value.removeprefix(prefix)
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", tag) or
            not value.startswith(prefix) or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.tar\.gz", filename)):
        raise ValueError("archive URL must be a fixed HTTPS .tar.gz asset in the source release")
    return value


def _download_target(value: str) -> None:
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or parsed.hostname not in DOWNLOAD_HOSTS or
            parsed.username or parsed.password or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("release download redirected outside the HTTPS GitHub asset allowlist")


class _SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _download_target(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_download(url: str):
    response = build_opener(_SafeRedirect()).open(url, timeout=60)
    try:
        _download_target(response.geturl())
    except Exception:
        response.close()
        raise
    return response


def _cache_path(root: Path, relative: str) -> Path:
    path = root
    if path.is_symlink():
        raise ValueError("cache root must not be a symlink")
    if not path.is_dir():
        raise ValueError("cache root must be a directory")
    for part in _relative(relative).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("cache symlinks are not allowed")
    return path


def _stage_archive(cache: Path, out: Path, url: str, sha256: str) -> str:
    cached = cache.exists()
    if cached and not cache.is_file():
        raise ValueError("archive cache must be a regular file")
    if cached and cache.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("archive exceeds the compressed size limit")
    try:
        source = cache.open("rb") if cached else _open_download(url)
    except OSError:
        raise RuntimeError("release archive unavailable; check public access or provide a local cache") from None
    count, digest = 0, hashlib.sha256()
    with source, out.open("xb") as output:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            count += len(chunk)
            if count > MAX_ARCHIVE_BYTES:
                raise ValueError("archive exceeds the compressed size limit")
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise ValueError("archive SHA-256 mismatch")
    return "local_archive" if cached else "https_archive"


def _extract_archive(archive: Path, root: Path) -> None:
    """Read bounded 512-byte tar headers, never extractall or PAX/GNU extensions.

    Parsing headers ourselves avoids an oversized PAX header being allocated by
    tarfile before a member-size check. gzip reads and member copies are bounded.
    UID, mode, mtime and any other archive metadata never control output files.
    """
    root.mkdir()
    seen, total, weight_bytes, members = set(), 0, 0, 0
    with gzip.open(archive, "rb") as stream:
        while True:
            header = stream.read(tarfile.BLOCKSIZE)
            if header == b"\0" * tarfile.BLOCKSIZE:
                if stream.read(tarfile.BLOCKSIZE) != header:
                    raise ValueError("archive must end with two zero tar blocks")
                # USTAR record padding only; no concatenated tar or gzip bombs.
                tail = stream.read(tarfile.RECORDSIZE + 1)
                if len(tail) > tarfile.RECORDSIZE or any(tail):
                    raise ValueError("unexpected data after the tar end marker")
                break
            if len(header) != tarfile.BLOCKSIZE:
                raise ValueError("truncated archive header or missing tar end marker")
            if header[257:265] != b"ustar\x00" + b"00":
                raise ValueError("archive must use USTAR format without GNU/PAX extensions")
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            members += 1
            if members > MAX_MEMBERS:
                raise ValueError("too many archive members")
            # Reject links, sparse files, device nodes, FIFOs, PAX and GNU names.
            if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
                raise ValueError("archive supports only regular files/directories, no links or extensions")
            name = member.name.removesuffix("/") if member.isdir() else member.name
            _relative(name)
            if len(name) > 256 or name in seen:
                raise ValueError("duplicate or oversized archive path")
            seen.add(name)
            if member.linkname:
                raise ValueError("archive link targets are forbidden")
            output = root.joinpath(*_relative(name).parts)
            if member.isdir():
                if member.size != 0 or not re.fullmatch(r"components(?:/[0-9]+)?", name):
                    raise ValueError("unexpected archive directory")
                output.mkdir(parents=True, exist_ok=True)
                continue
            if name not in BUNDLE_FILES | {"manifest.json"} and not ASSET_PATH.fullmatch(name):
                raise ValueError("archive file is not on the installation allowlist")
            is_weight = name.endswith((".pt", ".safetensors"))
            limit = MAX_WEIGHT_BYTES if is_weight else MAX_METADATA_BYTES
            if member.size < 1 or member.size > limit:
                raise ValueError("archive member exceeds its size limit")
            total += member.size
            weight_bytes += member.size if is_weight else 0
            if total > MAX_UNPACKED_BYTES or weight_bytes > MAX_WEIGHT_BYTES:
                raise ValueError("archive exceeds unpacked or 2 GB weight limit")
            output.parent.mkdir(parents=True, exist_ok=True)
            remaining = member.size
            with output.open("xb") as handle:
                while remaining:
                    chunk = stream.read(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ValueError("truncated archive member")
                    handle.write(chunk)
                    remaining -= len(chunk)
            output.chmod(0o644)
            padding = (-member.size) % tarfile.BLOCKSIZE
            if stream.read(padding) != b"\0" * padding:
                raise ValueError("invalid or truncated tar member padding")
    if "manifest.json" not in seen:
        raise ValueError("archive is missing manifest.json")


def stage_fusion(cache_root: Path, bundle: str, archive_cache: str,
                 destination: Path, manifest_sha256: str, source_release_url: str,
                 archive_url: str = "", archive_sha256: str = "",
                 *, require_verifier: bool = False) -> dict:
    _sha(manifest_sha256)
    _release_url(source_release_url)
    if bool(archive_url) != bool(archive_sha256):
        raise ValueError("archive URL and SHA-256 must both be provided")
    if archive_url:
        _archive_url(archive_url, source_release_url)
        _sha(archive_sha256)
    source = _cache_path(cache_root, bundle)
    archive = _cache_path(cache_root, archive_cache)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite destination")
    if source.exists() and destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("destination must be outside the source bundle")
    if not source.exists() and not archive_url:
        raise ValueError("local bundle missing: a pinned archive URL and SHA-256 are required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-stage-", dir=destination.parent) as temp:
        work = Path(temp)
        mode = "local_directory"
        install_root, install_path = cache_root, bundle
        if not source.exists():
            downloaded = work / "bundle.tar.gz"
            mode = _stage_archive(archive, downloaded, archive_url, archive_sha256)
            _extract_archive(downloaded, work / "unpacked")
            install_root, install_path = work, "unpacked"
        result = install_bundle(install_root, install_path, work / "installed",
                                manifest_sha256, source_release_url,
                                require_verifier=require_verifier)
        result.update({"bootstrap_source": mode, "archive_url": archive_url or None,
                       "archive_sha256": archive_sha256 or None,
                       "archive_sha256_verified": mode != "local_directory"})
        (work / "installed" / "installation.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("destination appeared during installation")
        (work / "installed").rename(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--bundle", default="artifacts/fusion_bundle")
    parser.add_argument("--archive-cache", default="artifacts/fusion_bundle.tar.gz")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-release-url", required=True)
    parser.add_argument("--archive-url", default="")
    parser.add_argument("--archive-sha256", default="")
    parser.add_argument("--require-verifier", action="store_true")
    args = parser.parse_args()
    try:
        result = stage_fusion(**vars(args))
    except (OSError, ValueError, KeyError, RuntimeError, tarfile.TarError, EOFError, zlib.error) as exc:
        # Do not echo signed CDN redirect URLs or authorization service messages.
        if isinstance(exc, OSError):
            parser.exit(1, "FUSION BOOTSTRAP FAILED: archive I/O failed; check cache/public access\n")
        parser.exit(1, f"FUSION BOOTSTRAP FAILED: {exc}\n")
    print(f"Verified {result['component_count']} components, {result['total_weight_bytes']} "
          f"weight bytes ({result['bootstrap_source']})")


if __name__ == "__main__":
    main()
