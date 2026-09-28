#!/usr/bin/env bash
# Additional fresh-process checks of the image's unmodified ENTRYPOINT/CMD.
set -Eeuo pipefail
image="${FUSION_IMAGE:?}"
data="${DATA_ROOT:?}"
output="${OUTPUT_ROOT:?}"
accepted="${ACCEPTANCE_ROOT:?}"
[[ ! -e "$output" ]] || { echo 'Refusing existing output' >&2; exit 2; }
mkdir -p "$output"
for number in 1 2; do
  name="falcon-pair-default-20260927-${number}"
  mkdir "$output/run_${number}"
  docker create --name "$name" --gpus all --network none \
    -v "$data:/data:ro" -v "$output/run_${number}:/out" \
    "$image" > "$output/container_${number}.id"
  docker start --attach "$name" > "$output/run_${number}.log" 2>&1
  docker inspect "$name" > "$output/container_${number}.json"
  [[ "$(docker inspect --format '{{.State.ExitCode}}' "$name")" == 0 ]]
  docker run --rm --gpus all --network none --entrypoint python3 \
    -e PYTHONPATH=/app -v "$data:/data:ro" -v "$output:/out" "$image" \
    scripts/audit_release.py submission --query /data/test_query.csv \
    --gallery /data/test_gallery.csv --submission-dir "/out/run_${number}" \
    --report "/out/c6_${number}.json"
done
docker run --rm --network none --entrypoint python3 -e PYTHONPATH=/app \
  -v "$output:/out" -v "$accepted:/accepted:ro" "$image" -c '
import json
from pathlib import Path
from scripts.check_determinism import compare_outputs
runs = compare_outputs(Path("/out/run_1"), Path("/out/run_2"))
reference = compare_outputs(Path("/accepted/determinism/run_1"), Path("/out/run_1"))
report = {"kind": "two_fresh_containers_default_entrypoint", "files": runs,
          "byte_identical": all(v["identical"] for v in runs.values()),
          "vs_deterministic_harness": reference,
          "identical_to_harness": all(v["identical"] for v in reference.values())}
Path("/out/comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["byte_identical"] and report["identical_to_harness"] else 1)
'
