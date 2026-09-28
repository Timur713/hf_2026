"""Compare saved train query variants, without inference or parameter changes."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_release import file_record
from scripts.check_query_content_isolation import compare_anchors
from scripts.check_query_independence import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    states, sources = {}, {}
    for name in ("batch32", "reverse", "with_extra_queries"):
        root = args.inputs / name
        with (root / "query.csv").open(newline="") as handle:
            rows = [SimpleNamespace(image_id=row["image_id"]) for row in csv.DictReader(handle)]
        states[name] = snapshot(root, rows)
        sources[name] = {file: file_record(root / file) for file in
                         ("query.csv", "submission.csv", "candidates.csv", "embeddings.npy")}
    comparisons = {name: compare_anchors(states["batch32"], states[name])
                   for name in ("reverse", "with_extra_queries")}
    report = {"kind": "saved_train_batched_variant_comparison",
              "scope": "24 fixed train queries; exact aligned embeddings/top10/candidate rows. "
                       "Supplementary only: original singleton/batched FAIL is unchanged.",
              "sources": sources, "comparisons": comparisons,
              "passed": all(item["passed"] for item in comparisons.values())}
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print({name: result["passed"] for name, result in comparisons.items()})
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
