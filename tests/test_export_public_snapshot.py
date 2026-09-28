import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.export_public_snapshot import ExportError, STUB, export_snapshot


@pytest.fixture
def source(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Snapshot export is a host-side Git operation")
    root = tmp_path.resolve() / "source"
    root.mkdir()
    for args in (("init", "-q"), ("config", "user.name", "Test"),
                 ("config", "user.email", "test@example.org")):
        subprocess.run(["git", "-C", str(root), *args], check=True)
    return root


def write(source, name, data):
    target = source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def commit(source):
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)


def export(source):
    target = source.parent / "public"
    return target, export_snapshot(source, target)


def test_only_tracked_files_no_history_or_ignored_credentials(source):
    secret = ("ghp_" + "AbCdEf0123" * 4).encode()
    write(source, "README.md", b"original removed content: " + secret)
    write(source, ".gitignore", b".env\nignored/\n")
    commit(source)
    write(source, "README.md", b"Public current version\n")
    commit(source)
    write(source, ".env", secret)
    write(source, "ignored/access.txt", secret)
    write(source, "untracked.txt", secret)
    target, report = export(source)
    assert not (target / ".git").exists()
    assert not (target / ".env").exists()
    assert not (target / "ignored").exists()
    assert not (target / "untracked.txt").exists()
    assert {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} == {
        ".gitignore", "README.md", "export_manifest.json"}
    assert secret not in (target / "export_manifest.json").read_bytes()
    assert report["complete"] is True
    assert report["tracked_worktree_dirty"] is False


def test_runtime_submission_and_raw_fail_are_byte_exact(source):
    originals = {
        "reid/model.py": b"print('unchanged')\n",
        "scripts/run.sh": b"#!/bin/sh\nexit 0\n",
        "tests/fixture.py": b"assert True\n",
        "configs/final.yaml": b"weight: 0.15\n",
        "splits/fold0.csv": b"image_id\r\nx\r\n",
        "third_party/official/evaluate.py": b"# unchanged evaluator\n",
        "submission/submission.csv": b"q,g\r\n",
        "submission/embeddings.npy": b"\x93NUMPY\0exact-binary\xff",
        "docs/benchmarks/check.json": b'{"passed":false,"delta":0.000393}\n',
    }
    for name, data in originals.items():
        write(source, name, data)
    (source / "scripts/run.sh").chmod(0o755)
    commit(source)
    target, report = export(source)
    assert not report["exclusions"]
    for name, data in originals.items():
        assert (target / name).read_bytes() == data
        record = next(r for r in report["exported_files"] if r["path"] == name)
        assert record["operation"] == "copy"
        assert record["sha256"] == hashlib.sha256(data).hexdigest()
    assert (target / "scripts/run.sh").stat().st_mode & 0o111


def test_internal_docs_stub_images_logs_and_host_inspection_absent(source):
    originals = {
        "README.md": b"public\n",
        "docs/infra.md": b"internal operations\n",
        "docs/handoff.md": b"handoff\n",
        "docs/external_resources.md": b"access correspondence\n",
        "docs/presentation/handoff_20260927.md": b"old handoff\n",
        "docs/errors/car.jpg": b"\0not redistributed image\n",
        "docs/check.log": b"unmodified raw private log\n",
        "docs/image_inspect.json": b'{"Env": ["private"]}',
        "docs/container_1.json": b'{"Mounts": ["private"]}',
        "docs/report.json": b'{"path":"/Users/example/private"}',
        "docs/private_note.md": b"local /Users/example/path\n",
    }
    for name, data in originals.items():
        write(source, name, data)
    commit(source)
    target, report = export(source)
    for record in report["exclusions"]:
        name = record["path"]
        assert (source / name).read_bytes() == originals[name]
        if name.endswith(".md"):
            assert (target / name).read_bytes() == STUB
        else:
            assert not (target / name).exists()
    assert len(report["exclusions"]) == len(originals) - 1


def test_existing_destination_never_overwritten(source):
    write(source, "README.md", b"public\n")
    commit(source)
    target = source.parent / "public"
    target.mkdir()
    sentinel = write(target, "sentinel", b"preserve me")
    with pytest.raises(ExportError, match="already exists"):
        export_snapshot(source, target)
    assert sentinel.read_bytes() == b"preserve me"
    assert not (target / "export_manifest.json").exists()


def test_readonly_destination_parent_fails_before_writes(source):
    write(source, "README.md", b"public\n")
    commit(source)
    parent = source.parent / "readonly"
    parent.mkdir()
    parent.chmod(0o555)
    try:
        with pytest.raises(ExportError, match="read-only"):
            export_snapshot(source, parent / "public")
        assert not (parent / "public").exists()
    finally:
        parent.chmod(0o755)


@pytest.mark.parametrize("kind", ["file", "parent", "destination", "source"])
def test_symlink_paths_rejected(source, kind):
    write(source, "README.md", b"public\n")
    write(source, "reid/module.py", b"original\n")
    commit(source)
    target, actual_source = source.parent / "public", source
    outside = source.parent / "outside"
    outside.mkdir()
    if kind == "file":
        (source / "README.md").unlink()
        (source / "README.md").symlink_to(write(outside, "file", b"private"))
    elif kind == "parent":
        (source / "reid/module.py").unlink()
        (source / "reid").rmdir()
        write(outside, "module.py", b"private")
        (source / "reid").symlink_to(outside, target_is_directory=True)
    elif kind == "destination":
        link = source.parent / "linked-parent"
        link.symlink_to(outside, target_is_directory=True)
        target = link / "public"
    else:
        actual_source = source.parent / "linked-source"
        actual_source.symlink_to(source, target_is_directory=True)
    with pytest.raises(ExportError, match="Symlink"):
        export_snapshot(actual_source, target)
    assert not target.exists()


def test_selected_secret_is_rejected_without_value_or_partial_export(source):
    secret = "hf_" + "AbCdEf1234" * 4
    write(source, "README.md", ("Secret: " + secret).encode())
    commit(source)
    with pytest.raises(ExportError) as error:
        export(source)
    assert "huggingface_token" in str(error.value)
    assert secret not in str(error.value)
    assert not (source.parent / "public").exists()


def test_dirty_tracked_worktree_is_explicit_and_bytes_match(source):
    write(source, "README.md", b"committed\n")
    commit(source)
    write(source, "README.md", b"reviewed tracked modification\n")
    target, report = export(source)
    assert report["tracked_worktree_dirty"] is True
    assert (target / "README.md").read_bytes() == b"reviewed tracked modification\n"
    assert json.loads((target / "export_manifest.json").read_text()) == report


def test_missing_tracked_file_and_nested_destination_rejected(source):
    write(source, "README.md", b"public\n")
    commit(source)
    with pytest.raises(ExportError, match="outside"):
        export_snapshot(source, source / "public")
    (source / "README.md").unlink()
    with pytest.raises(ExportError, match="Missing"):
        export(source)
    assert not (source.parent / "public").exists()
