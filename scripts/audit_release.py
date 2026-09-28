#!/usr/bin/env python3
"""Audit C6 deliverables or all inference-weight files in a release directory.

These are structural/integrity checks, not test-set evaluation or calibration.
No data or weights are modified. A report never claims to verify GPU inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from reid.eval.validate_outputs import _read_input_ids, validate_outputs


WEIGHT_SUFFIXES = frozenset({
    ".pt", ".pth", ".bin", ".onnx", ".engine", ".plan", ".safetensors",
    ".ckpt", ".trt", ".pb", ".tflite", ".npz",
})
MAX_WEIGHT_BYTES = 2_000_000_000  # conservative decimal GB, not GiB


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def audit_submission(query: Path, gallery: Path, submission_dir: Path) -> dict:
    submission = submission_dir / "submission.csv"
    embeddings = submission_dir / "embeddings.npy"
    candidates = submission_dir / "candidates.csv"
    validate_outputs(query, gallery, submission, embeddings, candidates)
    query_ids, gallery_ids = _read_input_ids(query), _read_input_ids(gallery)
    array = np.load(embeddings, allow_pickle=False, mmap_mode="r")
    with candidates.open(encoding="utf-8", newline="") as handle:
        answers = list(csv.DictReader(handle))
    answered = {row["query_id"] for row in answers}
    return {
        "audit": "submission_format_and_integrity",
        "validated": True,
        "query_count": len(query_ids),
        "gallery_count": len(gallery_ids),
        "candidate_rows": len(answers),
        "answered_queries": len(answered),
        "rejected_queries": len(query_ids) - len(answered),
        "embedding_shape": list(array.shape),
        "embedding_dtype": str(array.dtype),
        "files": {path.name: file_record(path) for path in (submission, embeddings, candidates)},
        "inputs": {"query": file_record(query), "gallery": file_record(gallery)},
        "scope": "C6 validation only; model provenance, row semantics and GPU reproduction not verified",
    }


def audit_weights(root: Path, model: Path, verifier: Path, require_all_data: bool) -> dict:
    import torch
    import yaml

    root, model, verifier = root.resolve(), model.resolve(), verifier.resolve()
    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    fusion = (model / "fusion_config.json").exists()
    component_configs = {}
    if fusion:
        from reid.infer.fusion import load_fusion_config

        cfg = load_fusion_config(model)
        config_path = model / "fusion_config.json"
        required = [config_path, verifier]
        model_folds = []
        for member in cfg["components"]:
            directory = model / member["path"]
            yaml_path = directory / "model_config.yaml"
            member_cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if not isinstance(member_cfg, dict):
                raise ValueError(f"invalid component model config: {yaml_path}")
            folds = member_cfg.get("train", {}).get("folds")
            if require_all_data and folds != "all":
                raise ValueError(f"final embedder must be trained on all folds: {member['path']}")
            model_folds.append(folds)
            required.extend((yaml_path, directory / "weights.safetensors"))
            component_configs[member["path"]] = file_record(yaml_path)
    else:
        config_path = model / "model_config.yaml"
        required = [model / "weights.safetensors", config_path, verifier]
    for path in required:
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"required artifact must be a file inside release directory: {path}")
    if not fusion:
        with config_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        model_folds = cfg.get("train", {}).get("folds")
        if require_all_data and model_folds != "all":
            raise ValueError(f"final embedder must be trained on all folds; found {model_folds!r}")

    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"release directory must contain real files, not symlinks: {path}")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
            files[str(path.relative_to(root))] = file_record(path)
    total = sum(record["bytes"] for record in files.values())
    if total > MAX_WEIGHT_BYTES:
        raise ValueError(f"inference weights exceed 2 GB: {total} > {MAX_WEIGHT_BYTES} bytes")

    payload = torch.load(verifier, map_location="cpu", weights_only=True)
    verifier_folds = payload.get("trained_on")
    if require_all_data and verifier_folds != "folds [0, 1, 2, 3, 4]":
        raise ValueError(f"final verifier must be trained on all folds; found {verifier_folds!r}")
    return {
        "audit": "weights_size_integrity_and_training_metadata",
        "validated": True,
        "require_all_data": require_all_data,
        "total_weight_bytes": total,
        "limit_bytes": MAX_WEIGHT_BYTES,
        "files": files,
        "model_config": file_record(config_path),
        "component_configs": component_configs,
        "model_name": cfg.get("name"),
        "model_trained_on": model_folds,
        "verifier_trained_on": verifier_folds,
        "verifier_arch": payload["arch"],
        "verifier_input": payload["input"],
        "scope": "file/metadata checks only; GPU VRAM and runtime must be measured separately",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    outputs = sub.add_parser("submission", help="validate the delivered C6 files")
    outputs.add_argument("--query", type=Path, required=True)
    outputs.add_argument("--gallery", type=Path, required=True)
    outputs.add_argument("--submission-dir", type=Path, required=True)
    outputs.add_argument("--report", type=Path, required=True)
    weights = sub.add_parser("weights", help="scan the complete staged solution or Docker /app")
    weights.add_argument("--root", type=Path, required=True)
    weights.add_argument("--model", type=Path, required=True)
    weights.add_argument("--verifier", type=Path, required=True)
    weights.add_argument("--require-all-data", action="store_true")
    weights.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.kind == "submission":
            report = audit_submission(args.query, args.gallery, args.submission_dir)
        else:
            report = audit_weights(args.root, args.model, args.verifier, args.require_all_data)
        # Report path must not overwrite audited artifacts or other user files.
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {report['audit']}; report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
