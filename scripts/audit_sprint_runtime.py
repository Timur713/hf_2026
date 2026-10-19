"""Read-only B-F1 identity/config audit of a real fusion bundle, without GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from reid.infer.fusion import load_fusion_config
from reid.infer.run import _parser
from scripts.audit_release import file_record, WEIGHT_SUFFIXES
from reid.post.policy import candidate_threshold_for, rerank_weight_for


def runtime_policy(config, threshold=None, weight=None):
    if (
        config.get("candidate_policy") != "gallery_z"
        or config.get("rerank_mode") != "standardized"
    ):
        raise ValueError("this audit requires gallery_z/standardized fusion policy")
    return {
        **config,
        "candidate_threshold": candidate_threshold_for(config, threshold),
        "rerank_weight": rerank_weight_for(config, weight),
    }


def code_identity():
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None  # a release image deliberately need not include Git/.git
    sources = (
        "reid/infer/run.py",
        "reid/infer/extractor.py",
        "reid/infer/verifier.py",
        "reid/post/verifier.py",
        "reid/post/pipeline.py",
        "scripts/bench_pipeline.py",
    )
    return {
        "code_commit": commit,
        "source_files": {
            name: file_record(root / name)
            for name in sources
            if (root / name).is_file()
        },
    }


def audit(model, verifier=None, threshold=None, weight=None):
    cfg = load_fusion_config(model)
    effective = runtime_policy(cfg, threshold, weight)
    files = {
        str(p.relative_to(model)): file_record(p)
        for p in sorted(model.rglob("*"))
        if p.is_file() and p.suffix in WEIGHT_SUFFIXES
    }
    if verifier and not verifier.resolve().is_relative_to(model.resolve()):
        files["external_verifier"] = file_record(verifier)
    total = sum(r["bytes"] for r in files.values())
    manifest = model / "manifest.json"
    verified = False
    if manifest.exists():
        raw = json.loads(manifest.read_text())
        records = raw.get("artifacts", {})
        required = {"fusion_config.json"}
        for member in cfg["components"]:
            required.update(
                f"{member['path']}/{name}"
                for name in ("model_config.yaml", "weights.safetensors")
            )
        if not required.issubset(records):
            raise ValueError("manifest does not cover selected components/config")
        for name, expected in records.items():
            path = (model / name).resolve()
            if (
                not path.is_relative_to(model.resolve())
                or file_record(path) != expected
            ):
                raise ValueError(f"bundle manifest mismatch: {name}")
        verified = True
    args = [
        "--images",
        "/data/images",
        "--query",
        "/data/test_query.csv",
        "--gallery",
        "/data/test_gallery.csv",
        "--out",
        "/out",
        "--model",
        str(model),
    ]
    if verifier:
        args.extend(["--verifier", str(verifier)])
    if threshold is not None:
        args.extend(["--candidate-threshold", str(threshold)])
    if weight is not None:
        args.extend(["--rerank-weight", str(weight)])
    parsed = vars(_parser().parse_args(args))
    return {
        "kind": "read_only_runtime_audit",
        **code_identity(),
        "inference_command_argv": ["python", "-m", "reid.infer.run", *args],
        "parsed_cli": {
            k: str(v) if isinstance(v, Path) else v for k, v in parsed.items()
        },
        "config": file_record(model / "fusion_config.json"),
        "effective_policy": effective,
        "threshold_source": "CLI" if threshold is not None else "artifact config",
        "weight_source": "CLI" if weight is not None else "artifact config",
        "manifest": file_record(manifest) if manifest.exists() else None,
        "manifest_files_verified": verified,
        "weight_files": files,
        "weight_bytes": total,
        "within_2GB": total <= 2_000_000_000,
        "components": [
            yaml.safe_load((model / c["path"] / "model_config.yaml").read_text())
            for c in cfg["components"]
        ],
        "evaluator": file_record(Path("reid/eval/eval_local.py")),
        "official_evaluator": file_record(Path("third_party/official/evaluate.py")),
        "scope": "Local bytes/config/CLI verified; NOT evidence of the judges' actual invocation or image.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--verifier", type=Path)
    p.add_argument("--candidate-threshold", type=float)
    p.add_argument("--rerank-weight", type=float)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    report = audit(a.model, a.verifier, a.candidate_threshold, a.rerank_weight)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Audit saved: {a.out}; threshold={report['effective_policy']['candidate_threshold']}"
    )


if __name__ == "__main__":
    main()
