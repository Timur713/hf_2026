#!/usr/bin/env bash
# Technical acceptance on published/staged weights; never trains or tunes on test.
set -Eeuo pipefail

data_root="${DATA_ROOT:-./dataset}"
output_root="${OUTPUT_ROOT:-./outputs/release-acceptance}"
if [[ -e "$output_root" ]]; then
  echo "Refusing to overwrite acceptance directory: $output_root" >&2
  exit 2
fi
[[ -f "$data_root/test_query.csv" && -f "$data_root/test_gallery.csv" ]] || {
  echo "Missing input CSV files in $data_root" >&2; exit 2;
}
mkdir -p "$output_root"
export DATA_ROOT="$data_root" OUTPUT_ROOT="$output_root"
git rev-parse HEAD > "$output_root/source_commit.txt"
git diff --exit-code
nvidia-smi -q > "$output_root/nvidia_smi.txt"
docker compose version > "$output_root/compose_version.txt"
docker compose config > "$output_root/compose_resolved.yml"
docker compose build reid
docker image inspect falcon-reid:local > "$output_root/image_inspect.json"

run_python() {
  docker compose run --rm --entrypoint python3 --env PYTHONPATH=/app reid "$@"
}
docker compose run --rm --entrypoint make reid check PYTHON=python3 DATA_ROOT=/data
# Plain production entry point, with network_mode: none and read-only inputs.
OUT_DIR=/out/production docker compose run --rm reid
run_python scripts/audit_release.py weights --root /app \
  --model /app/artifacts/models/final \
  --verifier /app/artifacts/models/verifier_f0/verifier.pt \
  --report /out/weights_audit.json
run_python scripts/audit_release.py submission \
  --query /data/test_query.csv --gallery /data/test_gallery.csv \
  --submission-dir /out/production --report /out/format_audit.json
run_python scripts/check_determinism.py \
  --images /data/images --query /data/test_query.csv --gallery /data/test_gallery.csv \
  --model /app/artifacts/models/final --verifier /app/artifacts/models/verifier_f0/verifier.pt \
  --device cuda:0 --out /out/determinism
run_python scripts/bench_extract.py \
  --model /app/artifacts/models/final --images /data/images --csv /data/test_query.csv \
  --device cuda:0 --warmup 50 --iterations 300 --batch-sizes 1,8,16,32 \
  --throughput-seconds 10 --require-cuda \
  --run-label release-acceptance --json-out /out/bench_extract.json
echo "PASS: offline production, format, weight budget, determinism and extraction benchmark."
echo "Back up $output_root and shelve the paid GPU (power-off does not stop its charge)."
