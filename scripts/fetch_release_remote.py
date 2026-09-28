"""Download pinned GitHub release assets on an existing SSH host, without forwarding GH credentials.

The local GitHub token is sent only to api.github.com. The authorized own host
receives a short-lived asset URL over SSH stdin, never a GitHub token. No URLs
with credentials are logged. Existing complete files are verified, not replaced.
"""
from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
import subprocess
from urllib.parse import urlparse


REMOTE = r'''
import hashlib, json, os, pathlib, sys, urllib.request
spec = json.load(sys.stdin)
root = pathlib.Path(spec["directory"])
root.mkdir(parents=True, exist_ok=True)
for asset in spec["assets"]:
    dest = root / asset["name"]
    def check(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
        return path.stat().st_size == asset["size"] and h.hexdigest() == asset["sha256"]
    if dest.exists():
        if not check(dest): raise ValueError("existing asset differs: " + dest.name)
    else:
        temp = dest.with_name(dest.name + ".partial")
        offset = temp.stat().st_size if temp.exists() else 0
        if temp.exists() and offset >= asset["size"]:
            if not check(temp): raise ValueError("completed partial checksum/size mismatch: " + dest.name)
            temp.rename(dest)
            print(json.dumps({"name": dest.name, "bytes": asset["size"], "sha256": asset["sha256"], "verified": True}), flush=True)
            continue
        req = urllib.request.Request(asset["url"], headers={"Range": "bytes=%d-" % offset} if offset else {})
        with urllib.request.urlopen(req, timeout=90) as response:
            append = offset and response.status == 206
            with temp.open("ab" if append else "wb") as out:
                for block in iter(lambda: response.read(1024 * 1024), b""): out.write(block)
        if not check(temp): raise ValueError("download checksum mismatch: " + dest.name)
        temp.rename(dest)
    print(json.dumps({"name": dest.name, "bytes": asset["size"], "sha256": asset["sha256"], "verified": True}), flush=True)
'''


def signed_url(repo: str, asset_id: int) -> str:
    token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    connection = http.client.HTTPSConnection("api.github.com", timeout=60)
    try:
        connection.request("GET", f"/repos/{repo}/releases/assets/{asset_id}", headers={
            "Authorization": "Bearer " + token, "Accept": "application/octet-stream",
            "User-Agent": "falcon-release-transfer", "X-GitHub-Api-Version": "2022-11-28"})
        response = connection.getresponse()
        url = response.getheader("Location")
        if response.status != 302 or not url:
            raise ValueError(f"GitHub did not return an asset redirect (HTTP {response.status})")
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in (
                "release-assets.githubusercontent.com", "objects.githubusercontent.com"):
            raise ValueError("unexpected release asset destination")
        return url
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Timur713/hack_falcon_2026")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--asset", action="append", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    import shlex
    release = json.loads(subprocess.check_output([
        "gh", "api", f"repos/{args.repo}/releases/tags/{args.tag}"], text=True))
    wanted = set(args.asset)
    assets = []
    for item in release["assets"]:
        if item["name"] not in wanted:
            continue
        if Path(item["name"]).name != item["name"]:
            raise ValueError("unsafe asset filename")
        digest = item.get("digest", "")
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError("release asset has no SHA-256 digest")
        assets.append({"name": item["name"], "size": item["size"],
                       "sha256": digest[7:], "url": signed_url(args.repo, item["id"])})
    if {a["name"] for a in assets} != wanted:
        raise ValueError("one or more requested assets are missing")
    command = ["ssh", "-i", str(args.key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "StrictHostKeyChecking=yes", args.host, "python3 -c " + shlex.quote(REMOTE)]
    # stderr can contain a signed URL in urllib errors; do not relay it.
    result = subprocess.run(command, input=json.dumps({"directory": args.directory, "assets": assets}),
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stdout, end="")
    if result.returncode:
        import re
        errors = re.findall(r"HTTP Error \d+|ConnectionResetError|TimeoutError|IncompleteRead", result.stderr)
        print("Remote error classes: " + ", ".join(sorted(set(errors))))
        raise SystemExit("Remote download failed; URL-bearing diagnostics suppressed. Retry resumes .partial files.")


if __name__ == "__main__":
    main()
