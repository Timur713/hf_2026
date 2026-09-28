"""Verify immutable public source metadata for the final inference models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def read_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "falcon-provenance/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def stream_digest(url: str) -> tuple[str, int]:
    request = urllib.request.Request(url, headers={"User-Agent": "falcon-provenance/1"})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_source(source: dict, download: bool) -> None:
    api_url = (
        f"https://huggingface.co/api/models/{source['model_id']}"
        f"/revision/{source['revision']}?blobs=true"
    )
    metadata = read_json(api_url)
    assert metadata["sha"] == source["revision"], (
        source["model_id"], "revision", metadata["sha"], source["revision"]
    )
    sibling = next(
        item for item in metadata["siblings"] if item["rfilename"] == source["filename"]
    )
    assert sibling["size"] == source["bytes"], (
        source["model_id"], "bytes", sibling["size"], source["bytes"]
    )
    assert sibling["lfs"]["sha256"] == source["sha256"], (
        source["model_id"], "sha256", sibling["lfs"]["sha256"], source["sha256"]
    )
    if download:
        sha256, size = stream_digest(source["weights_url"])
        assert size == source["bytes"], (source["model_id"], "download bytes", size)
        assert sha256 == source["sha256"], (source["model_id"], "download sha256", sha256)
    print(f"PASS {source['model_id']}@{source['revision']} {source['sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    try:
        for source in manifest["sources"]:
            verify_source(source, args.download)
    except (AssertionError, KeyError, StopIteration, OSError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
