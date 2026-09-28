"""One fixed train-fold0 scale control; no search, weights, extraction or GPU."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from scripts.audit_release import file_record
from scripts.replay_fusion import fuse_components

spec = importlib.util.spec_from_file_location(
    "fullframe_replay", Path(__file__).with_name("replay_fullframe_embeddings.py"))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--c4-report", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(args.out)
    baseline, c4 = (json.loads(path.read_text()) for path in (args.baseline_report, args.c4_report))
    if baseline["fold"] != 0 or c4["fold"] != 0 or len(baseline["shape"]) != 2 or baseline["shape"][1] != 3840:
        raise ValueError("this archived control is specific to the verified D-062 fold0 run")
    path = Path(baseline["input_embeddings"]["path"])
    if file_record(path)["sha256"] != baseline["input_embeddings"]["sha256"]:
        raise ValueError("baseline source embeddings changed")
    matrix = np.load(path, allow_pickle=False)
    nq = baseline["metrics"]["n_query"]
    blocks = [matrix[:, a:b] for a, b in ((0, 1536), (1536, 3072), (3072, 3840))]
    arrays = []
    for block in blocks:
        unit = block / np.linalg.norm(block, axis=1, keepdims=True)
        arrays.append((unit[:nq], unit[nq:]))
    args.out.mkdir(parents=True)
    report = {"scope": "D-062 train fold0 only, one predefined scale replacement, not tuning",
              "script": file_record(Path(__file__)), "baseline_report": file_record(args.baseline_report),
              "c4_report": file_record(args.c4_report), "block_dimensions": [1536, 1536, 768],
              "baseline_map10": baseline["metrics"]["map10"], "variants": {}}
    # First check recovery arithmetic at the ORIGINAL scales, then replace only
    # scales with the previously released C4 constants. No free parameters.
    for name, scales in (("original_scales_roundtrip", [c["similarity_std"] for c in baseline["metadata"]["components"]]),
                         ("released_c4_scales", c4["similarity_std"])):
        q, g, _ = fuse_components(arrays, scales)
        emb = args.out / f"{name}.npy"
        np.save(emb, np.concatenate([q, g]))
        cfg = json.loads(json.dumps(baseline["metadata"]))
        for component, scale in zip(cfg["components"], scales):
            component["similarity_std"] = scale
        config = args.out / f"{name}.json"
        config.write_text(json.dumps(cfg))
        result = helper.replay(emb, config, args.images, args.out / name)
        report["variants"][name] = {
            "scales": scales, "metrics": result["metrics"], "files": result["files"],
            "max_abs_embedding_delta_from_baseline": float(np.max(np.abs(matrix - np.concatenate([q, g]))))}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: value["metrics"]["map10"] for name, value in report["variants"].items()}))


if __name__ == "__main__":
    main()
