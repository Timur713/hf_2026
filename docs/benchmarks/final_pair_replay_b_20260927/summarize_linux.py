"""Compare saved Linux CPU replay receipts against A-104, without model loading."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    reference_root = root.parent / "fusion_pair_exact_20260927" / "raw"
    output = root / "linux" / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    expected_reports = (
        "ebbc98f282ffc75e42f479c7cc020c6bff4af8efcb83677f9dc176a2723fbe02",
        "3059e2051bfbc1c5dd40d72d17c9eb9c7f8ab6b4a4b2cdb0dc808591f17e50a4",
    )
    expected_windows = (
        "d4601daa6d019487472f4c07a87d225e810171f2fd64ec9547b1c841687d3c3c",
        "f3df791c3a725788bf3c6c26904e99ed88a6293654e4c4f98cb71b304eed9d22",
    )
    result = {
        "kind": "independent_linux_cpu_pair_replay_comparison",
        "source_archive_commit": "6a496647898f091c32f2d40f63822547f62550c3",
        "runtime_image_id": "sha256:6f3cd39865cd7bce3085ef0ed50dd4d2163cd20e32318196562cb17d384a5af9",
        "runtime": {"host": "Linux 7.0.0-31-generic x86_64", "image_arch": "amd64",
                    "network": "none", "gpus_exposed": False, "cpus_limit": 1,
                    "memory_limit": "3g", "OPENBLAS_NUM_THREADS": 1, "OMP_NUM_THREADS": 1},
        "policy": {"rerank_weight": 0.15, "candidate_threshold": 6.6},
        "final_acceptance": False,
        "scope": "strict train C4 replay only; not full-frame, final GPU image, or OOF training certification",
        "folds": {},
    }
    for fold in (0, 1):
        path = root / "linux" / f"fold{fold}_report.json"
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != expected_reports[fold]:
            raise ValueError(f"local report differs from remote hash: fold{fold}")
        report = json.loads(data)
        ref = json.loads((reference_root / f"f{fold}_w015_t660/report.json").read_text())
        checks = {}
        for stage, actual in report["stages"].items():
            expected = ref["stages"][stage]
            checks[stage] = {
                "all_metrics_equal": actual["metrics"] == expected["metrics"],
                "output_file_record_equal": {name: value == expected["files"][name]
                                             for name, value in actual["files"].items()},
            }
            if not checks[stage]["all_metrics_equal"]:
                raise ValueError(f"A104 metric mismatch: fold{fold}/{stage}")
        if report["logits"]["sha256"] != ref["logits"]["sha256"]:
            raise ValueError("not the original logits")
        if [d["sha256"] for d in report["dumps"]] != [d["sha256"] for d in ref["dumps"]]:
            raise ValueError("not the original C4 inputs")
        window = root / "linux" / f"fold{fold}_candidate_window.npz"
        window_bytes = window.read_bytes()
        window_sha = hashlib.sha256(window_bytes).hexdigest()
        if window_sha != expected_windows[fold]:
            raise ValueError(f"local window differs from remote hash: fold{fold}")
        with np.load(window, allow_pickle=False) as window_data:
            expected_shape = (report["stages"]["verifier"]["metrics"]["n_query"], 10)
            if window_data["fold"].item() != fold or window_data["candidates"].shape != expected_shape:
                raise ValueError("candidate window fold/shape mismatch")
        result["folds"][str(fold)] = {
            "status": "PASS strict original-logit replay",
            "report": {"bytes": len(data), "sha256": sha},
            "candidate_window": {"bytes": len(window_bytes), "sha256": window_sha,
                                 "shape": list(expected_shape)},
            "map10": report["stages"]["verifier"]["metrics"]["map10"],
            "candidate_score": report["stages"]["verifier"]["metrics"]["candidate_score"],
            "comparison": checks,
        }
    result["arithmetic_mean"] = {
        metric: sum(fold[metric] for fold in result["folds"].values()) / 2
        for metric in ("map10", "candidate_score")
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["arithmetic_mean"], indent=2))


if __name__ == "__main__":
    main()
