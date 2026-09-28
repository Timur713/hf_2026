"""Install a local, SHA-pinned prepare_fusion_artifact.py bundle without network.

The source release URL is an explicit provenance annotation, not a claim that
GitHub was contacted. Verify the original release assets before preparation.
The manifest digest is the trust root; every copied byte is checked against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import yaml

from reid.infer.fusion import _unique_object, load_fusion_config


MAX_WEIGHT_BYTES = 2_000_000_000
MAX_MANIFEST_BYTES = 1024 * 1024
ASSET_PATH = re.compile(r"components/[0-9]+/(weights\.safetensors|model_config\.yaml)\Z")
VERIFIER_PROVENANCE = "verifier.provenance.yaml"
BUNDLE_FILES = frozenset({"fusion_config.json", "verifier.pt", VERIFIER_PROVENANCE})


def _sha(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA-256 checksum")
    return value


def validate_verifier_provenance(path: Path, checkpoint_sha256: str) -> dict:
    """Bind optional source metadata to the verified checkpoint, without loading it.

    This is a local integrity check, not an online validation of authorship or
    training claims. Unknown descriptive fields are preserved for future handoffs.
    """
    with path.open("rb") as handle:
        data = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ValueError("verifier provenance exceeds 1 MiB")
    try:
        provenance = yaml.safe_load(data)
    except (yaml.YAMLError, UnicodeError) as exc:
        raise ValueError("invalid verifier provenance YAML") from exc
    if not isinstance(provenance, dict) or not isinstance(provenance.get("artifact"), dict):
        raise ValueError("verifier provenance must contain an artifact mapping")
    if _sha(provenance["artifact"].get("sha256")) != _sha(checkpoint_sha256):
        raise ValueError("verifier provenance checkpoint SHA-256 mismatch")
    return provenance


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (not value or "\\" in value or path.is_absolute() or
            any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"expected a canonical relative path: {value!r}")
    return path


def _source(cache_root: Path, relative: str) -> Path:
    path = cache_root
    if path.is_symlink():
        raise ValueError("cache root must not be a symlink")
    for part in _relative(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlinks are not allowed: {path}")
    if not path.is_dir():
        raise ValueError(f"bundle directory is missing: {path}")
    return path


def _release_url(value: str) -> str:
    url = urlsplit(value)
    prefixes = ("/Timur713/hack_falcon_2026/releases/tag/",
                "/Timur713/hf_2026/releases/tag/")
    prefix = next((p for p in prefixes if url.path.startswith(p)), prefixes[0])
    tag = url.path.removeprefix(prefix)
    if (url.scheme != "https" or url.netloc != "github.com" or
            not url.path.startswith(prefix) or not tag or "/" in tag or
            tag.lower() == "latest" or url.query or url.fragment):
        raise ValueError("source release must be a fixed HTTPS tag URL in the team repository")
    return value


def install_bundle(cache_root: Path, bundle: str, destination: Path,
                   manifest_sha256: str, source_release_url: str,
                   *, require_verifier: bool = False) -> dict:
    expected_manifest = _sha(manifest_sha256)
    source_release_url = _release_url(source_release_url)
    source = _source(cache_root, bundle)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite destination: {destination}")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("destination must be outside the source bundle")
    manifest_path = source / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("manifest.json must be a regular file")
    with manifest_path.open("rb") as handle:
        manifest_bytes = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds 1 MiB")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest:
        raise ValueError("manifest SHA-256 mismatch")
    audit = json.loads(manifest_bytes, object_pairs_hook=_unique_object)
    if not isinstance(audit, dict) or audit.get("kind") != "fusion_artifact_preparation":
        raise ValueError("expected a prepare_fusion_artifact.py manifest")
    records = audit.get("artifacts")
    if not isinstance(records, dict) or "fusion_config.json" not in records:
        raise ValueError("manifest must contain fusion_config.json and C3 artifacts")
    if require_verifier and "verifier.pt" not in records:
        raise ValueError("this Docker target requires verifier.pt")
    if VERIFIER_PROVENANCE in records and "verifier.pt" not in records:
        raise ValueError("verifier provenance requires verifier.pt")
    weight_bytes = 0
    for name, record in records.items():
        _relative(name)
        if name not in BUNDLE_FILES and not ASSET_PATH.fullmatch(name):
            raise ValueError(f"artifact is not on the installation allowlist: {name}")
        if (not isinstance(record, dict) or set(record) != {"bytes", "sha256"} or
                type(record["bytes"]) is not int or record["bytes"] < 1):
            raise ValueError(f"invalid file record: {name}")
        _sha(record["sha256"])
        if name == VERIFIER_PROVENANCE and record["bytes"] > MAX_MANIFEST_BYTES:
            raise ValueError("verifier provenance exceeds 1 MiB")
        if name.endswith((".safetensors", ".pt")):
            weight_bytes += record["bytes"]
    if weight_bytes > MAX_WEIGHT_BYTES:
        raise ValueError("inference weights exceed 2 GB")
    if type(audit.get("total_weight_bytes")) is not int or audit["total_weight_bytes"] != weight_bytes:
        raise ValueError("manifest total_weight_bytes does not match its artifacts")
    allowed = set(records) | {"manifest.json"}
    actual = set()
    for path in source.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"bundle must contain only real files/directories: {path}")
        if path.is_file():
            actual.add(path.relative_to(source).as_posix())
    if actual != allowed:
        raise ValueError(f"bundle file set differs from manifest: {sorted(actual ^ allowed)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        for name, record in records.items():
            copied = staged / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            digest, count = hashlib.sha256(), 0
            with (source / name).open("rb") as input_file, copied.open("xb") as output_file:
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    count += len(chunk)
                    if count > record["bytes"]:
                        raise ValueError(f"file exceeds pinned size: {name}")
                    digest.update(chunk)
                    output_file.write(chunk)
            if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"asset size/SHA-256 mismatch: {name}")
            copied.chmod(0o644)
        cfg = load_fusion_config(staged)
        expected = set(records) & BUNDLE_FILES
        if VERIFIER_PROVENANCE in records:
            validate_verifier_provenance(staged / VERIFIER_PROVENANCE, records["verifier.pt"]["sha256"])
        for index, member in enumerate(cfg["components"]):
            if member["path"] != f"components/{index}":
                raise ValueError("component paths must match the preparer's ordered components/0..N")
            expected.update(f"components/{index}/{name}" for name in ("weights.safetensors", "model_config.yaml"))
        if expected != set(records):
            raise ValueError("manifest contains components not selected by fusion_config.json")
        (staged / "manifest.json").write_bytes(manifest_bytes)
        receipt = {
            "kind": "fusion_bundle_installation", "manifest_sha256": expected_manifest,
            "source_release_url": source_release_url,
            "source_release_url_verified_online": False,
            "total_weight_bytes": weight_bytes, "component_count": len(cfg["components"]),
            "scope": "pinned local bytes only; not model quality, all-data or GPU acceptance",
        }
        (staged / "installation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"destination appeared during installation: {destination}")
        staged.rename(destination)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-release-url", required=True)
    parser.add_argument("--require-verifier", action="store_true")
    args = parser.parse_args()
    try:
        result = install_bundle(args.cache_root, args.bundle, args.destination,
                                args.manifest_sha256, args.source_release_url,
                                require_verifier=args.require_verifier)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"INSTALL FAILED: {exc}\n")
    print(f"Verified {result['component_count']} components, {result['total_weight_bytes']} weight bytes")


if __name__ == "__main__":
    main()
