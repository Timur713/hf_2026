"""Export reviewed tracked worktree bytes, never Git history or local artifacts.

Standard-library only. This is a conservative delivery filter, not proof of
licensing or absence of every kind of private information. No GitHub mutation.
The destination must not exist; an interrupted export is not complete unless
export_manifest.json exists. Source files and raw diagnostic results are never
rewritten. Excluded Markdown gets an explicit stub; other exclusions stay absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess


MANIFEST = "export_manifest.json"
STUB = b"Private internal file intentionally excluded from this public snapshot.\n"
TEXT_LIMIT = 2_000_000
PROTECTED = {"reid", "scripts", "tests", "configs", "splits", "submission"}
EXCLUDED_TOP = {
    ".git", ".venv", "venv", ".agents", ".codex", "dataset", "artifacts",
    "outputs", "output", "logs", "tmp", "runs", "wandb", "org_rdocs",
    "VehicleID_V1.0", "vehicleid_small", "__pycache__", ".pytest_cache",
}
INTERNAL_DOCS = {
    "docs/infra.md", "docs/handoff.md", "docs/external_resources.md",
    "docs/presentation/handoff_20260927.md",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff", ".bmp"}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "credential_assignment": re.compile(
        r"(?:password|пароль)(?:[^\n:|]{0,40})[:=]\s*`[^`\s<>$]{6,}`", re.I),
}
PRIVATE_PATH = re.compile(r"/(?:Users|home)/[A-Za-z][^/\s`]*")


class ExportError(ValueError):
    """Value-safe export failure; messages include paths/types, not file contents."""


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if result.returncode:
        raise ExportError("Git metadata operation failed; source must be a repository with HEAD")
    return result.stdout


def no_symlink_components(path: Path) -> None:
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ExportError(f"Symlink path rejected: {item.name}")


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def is_protected(name: str) -> bool:
    return name.split("/", 1)[0] in PROTECTED or name.startswith("third_party/official/")


def excluded_reason(name: str) -> str | None:
    path = PurePosixPath(name)
    if path.parts[0] in EXCLUDED_TOP or any(p in {".git", "__pycache__"} for p in path.parts):
        return "private_history_or_local_artifact"
    if name == MANIFEST:
        raise ExportError("Source contains reserved export_manifest.json")
    if path.name == ".env" or path.suffix.lower() in {".pem", ".key"}:
        raise ExportError(f"Potential credential file rejected: {name}")
    if is_protected(name):
        return None
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return "source_imagery_not_redistributed"
    if name in INTERNAL_DOCS:
        return "private_internal_document"
    if name.startswith("docs/"):
        if path.suffix.lower() == ".log":
            return "raw_log_kept_in_private_archive"
        if (path.name in {"image_inspect.json", "image_identity.json", "compose_resolved.yml"}
                or re.fullmatch(r"container_.*\.json", path.name)):
            return "host_environment_introspection"
        return None
    if len(path.parts) == 1:
        return None
    return "outside_delivery_allowlist"


def text_findings(path: Path, name: str) -> tuple[list[str], bool]:
    with path.open("rb") as handle:
        data = handle.read(TEXT_LIMIT + 1)
    if b"\0" in data:
        return [], False
    if len(data) > TEXT_LIMIT:
        raise ExportError(f"Oversized unreviewed text rejected: {name}")
    text = data.decode("utf-8", errors="replace")
    return ([kind for kind, pattern in SECRET_PATTERNS.items() if pattern.search(text)],
            bool(PRIVATE_PATH.search(text)))


def export_snapshot(source: Path, destination: Path) -> dict:
    source, destination = Path(source).absolute(), Path(destination).absolute()
    no_symlink_components(source)
    no_symlink_components(destination)
    if destination.exists():
        raise ExportError("Destination already exists; export never overwrites it")
    if not destination.parent.is_dir():
        raise ExportError("Destination parent must already exist")
    if not (destination.parent.stat().st_mode & 0o222) or not os.access(destination.parent, os.W_OK):
        raise ExportError("Destination parent is read-only")
    if source == destination or source in destination.parents:
        raise ExportError("Destination must be outside the source tree")
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    tracked = git(source, "ls-files", "--cached", "-z").decode("utf-8").split("\0")
    names = sorted({name for name in tracked if name})
    plans, excluded, source_records = [], [], []
    for name in names:
        rel = PurePosixPath(name)
        if rel.is_absolute() or any(p in {"", ".", "..", ".git"} for p in rel.parts):
            raise ExportError("Unsafe tracked path rejected")
        original = source / name
        no_symlink_components(original)
        if not original.is_file():
            raise ExportError(f"Missing or non-regular tracked file: {name}")
        mode = original.stat().st_mode
        if not stat.S_ISREG(mode):
            raise ExportError(f"Non-regular tracked file: {name}")
        reason = excluded_reason(name)
        if reason is None:
            secrets, private_path = text_findings(original, name)
            if secrets:
                raise ExportError(f"Secret-like content rejected: {name}; types={','.join(secrets)}")
            if private_path and name.startswith("docs/") and not is_protected(name):
                reason = "private_local_path_in_document"
        record = {"path": name, **file_record(original)}
        source_records.append(record)
        if reason:
            stub = rel.suffix.lower() == ".md"
            excluded.append({"path": name, "reason": reason, "markdown_stub": stub,
                             "source_sha256": record["sha256"]})
            if stub:
                plans.append((name, "stub", record, 0o644))
        else:
            plans.append((name, "copy", record, 0o755 if mode & 0o111 else 0o644))
    dirty = bool(git(source, "status", "--porcelain", "--untracked-files=no").strip())
    destination.mkdir(mode=0o755)
    exported = []
    for name, operation, original_record, mode in plans:
        src, dst = source / name, destination / name
        # Fail closed if any source changed between planning and copying.
        no_symlink_components(src)
        if file_record(src) != {k: original_record[k] for k in ("bytes", "sha256")}:
            raise ExportError(f"Source changed during export: {name}; incomplete destination retained")
        dst.parent.mkdir(parents=True, exist_ok=True)
        no_symlink_components(dst)
        with dst.open("xb") as out:
            if operation == "stub":
                out.write(STUB)
            else:
                with src.open("rb") as handle:
                    shutil.copyfileobj(handle, out, 1024 * 1024)
        dst.chmod(mode)
        actual = file_record(dst)
        if operation == "copy" and actual != {k: original_record[k] for k in ("bytes", "sha256")}:
            raise ExportError(f"Copy integrity mismatch: {name}; incomplete destination retained")
        exported.append({"path": name, "operation": operation, "mode": oct(mode), **actual})
    report = {
        "schema_version": 1, "complete": True, "source_commit": commit,
        "source_kind": "git-tracked worktree bytes; not a history clone",
        "tracked_worktree_dirty": dirty, "source_files": source_records,
        "exported_files": exported, "exclusions": excluded,
        "privacy_review_scope": "Heuristic selected-text scan; excluded internal documents are not published. "
                                "Binary contents and licenses require separate review.",
        "diagnostics_policy": "Copied raw diagnostic bytes are unchanged; excluded files are listed, never rewritten as PASS.",
        "manifest_self_hash": "Not included; hash export_manifest.json externally after export.",
    }
    with (destination / MANIFEST).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = export_snapshot(args.source, args.destination)
    except (ExportError, OSError, UnicodeError) as error:
        print(f"Export failed: {error}")
        return 1
    print(json.dumps({"complete": True, "source_commit": report["source_commit"],
                      "exported_files": len(report["exported_files"]),
                      "exclusions": len(report["exclusions"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
