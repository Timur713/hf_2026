"""Recompare saved TRAIN numeric checks; no inference, tuning or test inputs."""
import argparse
import json
from pathlib import Path

from reid.infer.run import read_inputs
from scripts.audit_release import file_record
from scripts.check_query_independence import compare, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite report")
    names = ["batch32", "reverse", "with_extra_queries"]
    states, sources = {}, {}
    for name in names:
        directory = args.root / name
        rows = read_inputs(directory / "query.csv", args.images)
        states[name] = snapshot(directory, rows)
        sources[name] = {file: file_record(directory / file) for file in
                         ("query.csv", "submission.csv", "candidates.csv", "embeddings.npy")}
    result = {"kind": "saved_train_batch_variant_recomparison", "source_files": sources,
              "reference": "batch32", "scope": "24 fixed train queries; no universal independence claim",
              "comparisons": {name: compare(states["batch32"], states[name]) for name in names[1:]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({name: item["passed"] for name, item in result["comparisons"].items()}))


if __name__ == "__main__":
    main()
