"""Independent, read-only audit of supplied A-104/D-066 handoff artifacts.

Run from the repository root with PYTHONPATH=.; generated receipts are not
claims about full-frame extraction, genuine OOF training, or final GPU quality.
No images or test inputs are opened. No hyperparameter search is performed.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import yaml

from reid.eval.validate_outputs import _read_input_ids
from reid.post.gallery_policy import GalleryPolicy
from scripts.analysis.verify_model_provenance import read_json, verify_source
from scripts.replay_fusion import fuse_components, load_components


def record(path):
    path = Path(path)
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--old-bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"refusing to overwrite {args.out}")
    manifest_dir = Path("docs/benchmarks/final_pair_provenance_20260927")
    source_manifest = json.loads((manifest_dir / "source_models.json").read_text())
    receipt = {"utc": datetime.now(timezone.utc).isoformat(),
               "code_commit": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], text=True).strip(),
               "platform": platform.platform(), "numpy_version": np.__version__,
               "policy": {"rerank_weight": 0.15, "candidate_threshold": 6.6},
               "scope": "train C4 postprocessing and source metadata; not end-to-end quality",
               "assets": {}, "provenance": {}, "folds": {}}
    config_text = io.StringIO()
    with contextlib.redirect_stdout(config_text):
        np.show_config()
    receipt["numpy_config"] = config_text.getvalue()
    # Release checksum manifest was independently checked against GitHub API.
    sums = args.assets / "SHA256SUMS-pair-20260927.txt"
    require(record(sums)["sha256"] ==
            "4ac921f247ae1c0822c8227babee8fad010ea1333334814469f1d63ba9b9aa16",
            "release checksum manifest differs from GitHub asset digest")
    receipt["assets"][sums.name] = record(sums)
    for line in sums.read_text().splitlines():
        expected, name = line.split()
        path = args.assets / name
        if not path.exists():
            receipt["assets"][name] = {"status": "not downloaded", "expected_sha256": expected}
            continue
        actual = record(path)
        require(actual["sha256"] == expected, f"asset digest mismatch: {name}")
        receipt["assets"][name] = {"status": "PASS", **actual}

    # Recheck pinned public API metadata without downloading source tensors.
    registry_text = subprocess.check_output(
        ["git", "show", "b2823b0:reid/model/__init__.py"], text=True)
    registry = next(ast.literal_eval(n.value) for n in ast.parse(registry_text).body
                    if isinstance(n, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "REGISTRY" for t in n.targets))
    for source in source_manifest["sources"]:
        verify_source(source, download=False)
        require(".".join(registry[source["arch"]][:2]) == source["timm_name"],
                f"training registry does not match {source['arch']}")
        meta = read_json(f"https://huggingface.co/api/models/{source['model_id']}"
                         f"/revision/{source['revision']}?blobs=true")
        blob = next(x for x in meta["siblings"] if x["rfilename"] == source["filename"])
        receipt["provenance"][source["arch"]] = {
            "status": "PASS metadata only; source bytes not re-downloaded",
            "revision": meta["sha"], "blob": blob,
            "registry_commit": "b2823b0", "timm_name": source["timm_name"]}

    component_files = (("hires392_all", "components/0/weights.safetensors"),
                       ("div_clip_all", "components/1/weights.safetensors"),
                       ("v_fus_all_frz_input392", "verifier.pt"))
    receipt["final_weights"] = {}
    for (name, relative), artifact in zip(component_files, source_manifest["final_artifacts"]):
        actual = record(args.old_bundle / relative)
        require(actual["sha256"] == artifact["sha256"], f"final tensor SHA mismatch: {name}")
        receipt["final_weights"][name] = {"status": "PASS", **actual}
    receipt["final_weight_bytes"] = sum(v["bytes"] for v in receipt["final_weights"].values())
    require(receipt["final_weight_bytes"] <= 2_000_000_000, "weights exceed budget")
    for index, name in enumerate(("hires392_all", "div_clip_all")):
        old = yaml.safe_load((args.old_bundle / f"components/{index}/model_config.yaml").read_text())
        new = yaml.safe_load((manifest_dir / f"{name}.yaml").read_text())
        old.pop("pretrained", None)
        new.pop("pretrained", None)
        require(old == new, f"non-provenance model config changed: {name}")
    receipt["metadata_only_config_change"] = "PASS: only pretrained mapping differs"

    for fold, replay_name in ((0, "replay_f0"), (1, "replay_f1_base")):
        replay_dir = args.assets / replay_name
        replay = json.loads((replay_dir / "report.json").read_text())
        reference = json.loads(Path(
            f"docs/benchmarks/fusion_pair_exact_20260927/raw/f{fold}_w015_t660/report.json").read_text())
        for actual, expected in zip(replay["dumps"], reference["dumps"]):
            require(actual["sha256"] == expected["sha256"], "C4 dump differs from A reference")
        result = {"replay_report": record(replay_dir / "report.json"),
                  "metrics": {key: value["metrics"] for key, value in replay["stages"].items()},
                  "reference_metric_delta": {
                      key: {metric: value["metrics"][metric] - reference["stages"][key]["metrics"][metric]
                            for metric in ("map10", "candidate_score")}
                      for key, value in replay["stages"].items()}}
        with np.load(args.assets / f"pair_logits_f{fold}.npz", allow_pickle=False) as logits, \
                np.load(args.assets / f"pair_window_f{fold}.npz", allow_pickle=False) as window, \
                np.load(replay_dir / "candidate_window.npz", allow_pickle=False) as native:
            require(np.array_equal(logits["query_ids"], native["query_ids"]), "query order differs")
            require(np.array_equal(logits["candidates"], window["candidates"]),
                    "A published window and logits differ")
            differing = np.flatnonzero(np.any(logits["candidates"] != native["candidates"], axis=1))
            set_mismatch = [int(i) for i in differing if
                            set(logits["candidates"][i]) != set(native["candidates"][i])]
            result["query_count"] = int(len(native["query_ids"]))
            result["window_differing_rows"] = differing.tolist()
            result["window_set_mismatch_rows"] = set_mismatch
            result["status"] = "PASS" if not len(differing) else "STRICT LOGIT GATE REFUSED"
            if len(differing):
                qids = _read_input_ids(Path(f"splits/fold{fold}_query.csv"))
                gids = _read_input_ids(Path(f"splits/fold{fold}_gallery.csv"))
                q, g, _ = fuse_components(load_components(
                    [Path(d["path"]) for d in replay["dumps"]], qids, gids))
                policy = GalleryPolicy(g, gids)
                result["cutoff_diagnostics"] = []
                for i in differing:
                    scores = policy.ranking_scores(q[i:i + 1])[0]
                    order = np.argsort(-scores, kind="stable")
                    result["cutoff_diagnostics"].append({
                        "query_row": int(i), "query_id": qids[i],
                        "rank10_11": [{"gallery_id": gids[j], "score": float(scores[j])}
                                      for j in order[9:11]],
                        "score_gap": float(scores[order[9]] - scores[order[10]])})
        receipt["folds"][str(fold)] = result
    receipt["limits"] = [
        "macOS ARM BLAS changes one fold1 rank10/11 near-tie; strict original-logit replay refuses",
        "fold1 original and pinned-scale replay both refuse; no missing logit was fabricated",
        "fold1 verifier replay must be independently rerun on source-compatible Linux/x86",
        "source API metadata and final tensor digests checked, not a forensic proof of actual training initialization",
        "OOF checkpoints not downloaded or rescored, no proof of train/validation exclusion from tensors alone",
        "no test data, full-frame extraction, plate coverage, GPU speed, or final submission touched"]
    args.out.mkdir(parents=True)
    for fold, name in ((0, "replay_f0"), (1, "replay_f1_base")):
        shutil.copyfile(args.assets / name / "report.json", args.out / f"fold{fold}_report.json")
    (args.out / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"weight_bytes": receipt["final_weight_bytes"],
                      "fold_status": {k: v["status"] for k, v in receipt["folds"].items()}}, indent=2))


if __name__ == "__main__":
    main()
