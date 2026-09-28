"""CPU-only delivery evidence check. Never claims GPU acceptance or rewrites Git.

Exit 1 means at least one blocked/missing check. The report contains paths and
credential TYPES only, never credential values, matching lines or Git diffs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import yaml

from scripts.audit_release import audit_submission, file_record

TEXT_LIMIT = 2_000_000
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "credential_assignment": re.compile(
        r"(?:password|пароль)(?:[^\n:|]{0,40})[:=]\s*`[^`\s<>$]{6,}`", re.I),
}


def git(root, *args, text=True):
    return subprocess.check_output(["git", "-C", str(root), *args], text=text)


def secret_types(content: bytes) -> list[str]:
    if b"\0" in content:
        return []
    text = content.decode("utf-8", errors="replace")
    return [kind for kind, pattern in SECRET_PATTERNS.items() if pattern.search(text)]


def scan_secrets(root: Path, history: bool) -> dict:
    """Heuristic scan of tracked text + all local refs, with explicit coverage gaps."""
    findings, skipped = set(), set()
    for name in git(root, "ls-files", "-z").split("\0"):
        if not name:
            continue
        path = root / name
        if not path.is_file() or path.is_symlink():
            skipped.add(name)
            continue
        if path.stat().st_size > TEXT_LIMIT:
            skipped.add(name)
            continue
        for kind in secret_types(path.read_bytes()):
            findings.add(("working_tree", name, kind))
    if history:
        objects = {}
        for line in git(root, "rev-list", "--objects", "--all").splitlines():
            oid, _, name = line.partition(" ")
            if name:
                objects[oid] = name
        records = subprocess.run(
            ["git", "-C", str(root), "cat-file", "--batch-check"],
            input="\n".join(objects) + "\n", text=True, capture_output=True, check=True,
        ).stdout.splitlines()
        process = subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            for record in records:
                oid, kind, length = record.split()
                if kind != "blob":
                    continue
                if int(length) > TEXT_LIMIT:
                    skipped.add(objects[oid])
                    continue
                process.stdin.write((oid + "\n").encode())
                process.stdin.flush()
                header = process.stdout.readline().split()
                size = int(header[2])
                content = process.stdout.read(size)
                if len(content) != size or process.stdout.read(1) != b"\n":
                    raise ValueError("Incomplete Git object stream")
                for secret_kind in secret_types(content):
                    findings.add(("history", objects[oid], secret_kind))
        finally:
            process.stdin.close()
            process.stdout.close()
            if process.wait() != 0:
                raise ValueError("Git object scan failed")
    return {
        "findings": [{"scope": scope, "path": path, "type": kind}
                     for scope, path, kind in sorted(findings)],
        "history_scanned": history,
        "skipped_paths": sorted(skipped),
        "scope": "heuristic tracked text scan; binaries, files >2MB, remote-only refs, "
                 "release assets, untracked files and license permissions require separate review",
    }


def docker_defaults(root: Path) -> dict:
    dockerfile = (root / "Dockerfile").read_text()
    compose = yaml.safe_load((root / "docker-compose.yml").read_text())["services"]["reid"]
    stages = re.findall(r"^FROM\s+.+?\s+AS\s+(\S+)\s*$", dockerfile, re.I | re.M)
    build = compose.get("build", {})
    target = build.get("target", stages[-1] if stages else "unknown") if isinstance(build, dict) else stages[-1]
    command = compose.get("command", [])
    model = command[command.index("--model") + 1] if "--model" in command else None
    return {"docker_last_stage": stages[-1] if stages else None,
            "compose_target": target, "compose_model": model,
            "offline": compose.get("network_mode") == "none",
            "uses_fusion": target == "fusion" and model is not None and "/fusion" in model}


def upstream_provenance(bundle: Path) -> dict:
    config = json.loads((bundle / "fusion_config.json").read_text())
    if not isinstance(config.get("components"), list) or not config["components"]:
        raise ValueError("Fusion bundle needs nonempty components")
    components = []
    for item in config["components"]:
        path = (bundle / item["path"] / "model_config.yaml").resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError("Component config escapes bundle")
        cfg = yaml.safe_load(path.read_text())
        source = cfg.get("pretrained", {})
        source = source if isinstance(source, dict) else {}
        missing = [key for key in ("url", "revision", "sha256") if not source.get(key)]
        if source.get("sha256") and not re.fullmatch(r"[0-9a-fA-F]{64}", str(source["sha256"])):
            missing.append("valid_sha256")
        if source.get("revision") == "latest" or "latest" in str(source.get("url", "")):
            missing.append("immutable_revision")
        components.append({"path": item["path"], "arch": cfg.get("arch"),
                           "pool": cfg.get("pool"), "input_size": cfg.get("input_size"),
                           "embed_dim": cfg.get("embed_dim"), "trained_on": cfg.get("train", {}).get("folds"),
                           "missing_upstream_metadata": missing})
    return {"components": components, "config": file_record(bundle / "fusion_config.json")}


def verifier_provenance(bundle: Path) -> dict:
    """Check the sidecar and bind its training/source claims to actual bytes.

    No pickle/checkpoint is loaded. This only checks metadata completeness and
    file integrity; audit_release weights still owns the all-data weight gate.
    """
    sidecar, checkpoint = bundle / "verifier.provenance.yaml", bundle / "verifier.pt"
    result = {"sidecar": sidecar.name, "checkpoint": checkpoint.name,
              "missing_upstream_metadata": [], "missing_artifact_metadata": [],
              "checkpoint_sha256_matches": False, "all_data_training_declared": False,
              "scope": "local metadata and checkpoint SHA binding only; not proof of training or online source verification"}
    missing = result["missing_artifact_metadata"]
    for path in (sidecar, checkpoint):
        if path.is_symlink() or not path.is_file():
            missing.append(f"regular_{path.name}")
    if missing:
        return result
    if sidecar.stat().st_size > 1024 * 1024:
        missing.append("sidecar_size_limit")
        return result
    try:
        data = yaml.safe_load(sidecar.read_text())
    except (yaml.YAMLError, UnicodeError):
        missing.append("valid_sidecar_yaml")
        return result
    if not isinstance(data, dict):
        missing.append("sidecar_mapping")
        return result
    result["sidecar_integrity"] = file_record(sidecar)
    result["checkpoint_integrity"] = file_record(checkpoint)
    source = data.get("pretrained", {})
    source = source if isinstance(source, dict) else {}
    upstream_missing = result["missing_upstream_metadata"]
    upstream_missing.extend(key for key in ("author", "url", "revision", "sha256")
                            if not isinstance(source.get(key), str) or not source[key].strip())
    if source.get("sha256") and not re.fullmatch(r"[0-9a-fA-F]{64}", str(source["sha256"])):
        upstream_missing.append("valid_sha256")
    if (str(source.get("revision", "")).lower() == "latest" or
            "latest" in str(source.get("url", "")).lower()):
        upstream_missing.append("immutable_revision")
    artifact = data.get("artifact", {})
    artifact = artifact if isinstance(artifact, dict) else {}
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(artifact.get("sha256", ""))):
        missing.append("artifact_sha256")
    else:
        result["checkpoint_sha256_matches"] = (
            artifact["sha256"].lower() == result["checkpoint_integrity"]["sha256"])
    if not isinstance(artifact.get("release_url"), str) or not artifact["release_url"].strip():
        missing.append("artifact_release_url")
    elif "latest" in artifact["release_url"].lower():
        missing.append("immutable_artifact_release")
    train = data.get("train", {})
    train = train if isinstance(train, dict) else {}
    result["all_data_training_declared"] = (
        train.get("folds") == "all" or train.get("data") == "all five train folds")
    return result


def build_report(root, *, query=None, gallery=None, submission_dir=None, expected_dim=None,
                 acceptance_c6=None, bundle=None, scan_history=False):
    checks = []

    def add(name, status, evidence):
        checks.append({"id": name, "status": status, "evidence": evidence})

    defaults = docker_defaults(root)
    add("default_runtime", "pass" if defaults["uses_fusion"] and defaults["offline"] else "blocked", defaults)
    if query is not None and gallery is not None:
        actual = audit_submission(query, gallery, submission_dir or root / "submission")
        dimension_ok = expected_dim is not None and actual["embedding_shape"][1] == expected_dim
        add("submission_c6_and_dimension", "pass" if dimension_ok else "blocked",
            {**actual, "expected_embedding_dim": expected_dim})
        if acceptance_c6 is not None:
            accepted = json.loads(acceptance_c6.read_text())
            matches = (accepted.get("validated") is True and accepted.get("files") == actual["files"]
                       and accepted.get("inputs") == actual["inputs"])
            add("submission_matches_acceptance", "pass" if matches else "blocked",
                {"acceptance_report": str(acceptance_c6), "report_integrity": file_record(acceptance_c6),
                 "scope": "byte equality only; reviewer must link acceptance report to final image and model manifest"})
        else:
            add("submission_matches_acceptance", "missing", "Pass final full-run audit_release c6.json via --acceptance-c6")
    else:
        add("submission_c6_and_dimension", "missing", "Need --query --gallery --expected-dim; mere presence of files is not acceptance")
    if bundle is not None:
        provenance = upstream_provenance(bundle)
        complete = all(not c["missing_upstream_metadata"] and c["trained_on"] == "all"
                       for c in provenance["components"])
        add("component_provenance", "pass" if complete else "blocked", provenance)
        verifier = verifier_provenance(bundle)
        complete = (not verifier["missing_upstream_metadata"] and
                    not verifier["missing_artifact_metadata"] and
                    verifier["checkpoint_sha256_matches"] and verifier["all_data_training_declared"])
        add("verifier_provenance", "pass" if complete else "blocked", verifier)
    else:
        add("component_provenance", "missing", "Need --bundle; run audit_release weights --require-all-data separately")
        add("verifier_provenance", "missing", "Need --bundle with verifier.pt and verifier.provenance.yaml")
    for fold in (0, 1):
        path = root / f"docs/benchmarks/fusion_pair_exact_20260927/raw/f{fold}_w015_t660/report.json"
        if not path.exists():
            add(f"a104_fold{fold}_evidence", "missing", str(path.relative_to(root)))
            continue
        report = json.loads(path.read_text())
        add(f"a104_fold{fold}_evidence", "review", {
            "report": str(path.relative_to(root)),
            "dumps": report["dumps"], "logits": report["logits"],
            "policy": report.get("policy"),
            "training_provenance_verified": report.get("training_provenance_verified"),
            "scope": "A104 matched C4 train-fold evidence for DINO392 + CLIP288; "
                     "not an automated OOF training certificate or full-frame/GPU acceptance",
        })
    scan = scan_secrets(root, scan_history)
    add("public_repo_credentials", "blocked" if scan["findings"] else ("review" if scan_history else "missing"), scan)
    add("official_extractor", "review" if (root / "third_party/official/extractor.py").is_file() else "missing",
        "Organizers must confirm official Extractor contract; file presence alone does not certify compatibility")
    return {"kind": "delivery_evidence_check", "code_commit": git(root, "rev-parse", "HEAD").strip(),
            "checks": checks, "automated_checks_pass": not any(c["status"] in ("blocked", "missing") for c in checks),
            "final_acceptance": False,
            "scope": "CPU integrity evidence only. No GPU measurements, OOF training verification, license/legal clearance, "
                     "public access check, or human submission. See docs/final_delivery_checklist.md."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    for name in ("query", "gallery", "submission-dir", "acceptance-c6", "bundle"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--expected-dim", type=int)
    parser.add_argument("--scan-history", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(args.root.resolve(), query=args.query, gallery=args.gallery,
                              submission_dir=args.submission_dir, expected_dim=args.expected_dim,
                              acceptance_c6=args.acceptance_c6, bundle=args.bundle, scan_history=args.scan_history)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        # Don't print arbitrary exceptions: Git/config text may contain secrets.
        print(f"CHECK FAILED: {type(exc).__name__}; inspect paths/schema locally")
        return 2
    for check in report["checks"]:
        print(f"{check['status'].upper()}: {check['id']}")
    print(f"Report saved: {args.report}; GPU/final acceptance NOT certified")
    return 0 if report["automated_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
