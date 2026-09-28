"""Build-time only: obtain pinned release assets and verify them before installation.

A local cache supports authenticated private-repository development. A clean
public checkout downloads the same fixed URLs inside Docker build (TZ section 9).
No access token is passed to Docker, and a bad cache fails rather than falling back.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


ASSETS = {
    "model": "artifacts/models/final/weights.safetensors",
    "config": "artifacts/models/final/model_config.yaml",
    "verifier": "artifacts/models/verifier_f0/verifier.pt",
}


def stage_asset(cache: Path, destination: Path, url: str, sha256: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("expected a lowercase SHA-256 checksum")
    if urlsplit(url).scheme != "https":
        raise ValueError("release URL must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            source = cache.open("rb") if cache.exists() else urlopen(url, timeout=60)
            with source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError(f"SHA-256 mismatch: {destination.name}")
        temporary.chmod(0o644)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, default=Path("/app"))
    for name in ASSETS:
        parser.add_argument(f"--{name}-url", required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    for name, relative in ASSETS.items():
        stage_asset(args.cache_root / relative, args.destination_root / relative,
                    getattr(args, f"{name}_url"), getattr(args, f"{name}_sha256"))
        print(f"verified release asset: {relative}")


if __name__ == "__main__":
    main()
