#!/usr/bin/env bash
# TRAIN-only experiment. Requires a real, exact CUDA parity gate first.
set -Eeuo pipefail
session_root="${SESSION_ROOT:?Set isolated session directory}"
data_root="${DATA_ROOT:?Set dataset directory}"
image="${FUSION_BASE_IMAGE:-falcon-runtime:20260927}"
report_root="$session_root/outputs/precast_benchmark"
[[ ! -e "$report_root" ]] || { echo 'Refusing existing output directory' >&2; exit 2; }
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["passed"] and r["precast_cast_stats"]["applied"]' \
  "$session_root/outputs/precast_parity/report.json"
mkdir -p "$report_root"
for threads in 8 2; do
  docker run --rm --gpus all --network none --entrypoint python3 \
    -e PYTHONPATH=/app -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$session_root/src:/app:ro" -v "$session_root/artifacts:/artifacts:ro" \
    -v "$data_root:/data:ro" -v "$report_root:/out" -w /app "$image" \
    scripts/bench_extract.py --model /artifacts/fusion_all_d063 \
    --images /data/images --csv /app/splits/fold0_query.csv --device cuda:0 \
    --torch-threads "$threads" --preprocess-workers 8 --precast-projections \
    --warmup 50 --iterations 300 --batch-sizes 1,8,16,32 \
    --throughput-seconds 10 --require-cuda \
    --run-label "A5000-fusion3-precast-t${threads}-w8" \
    --json-out "/out/threads${threads}.json"
done
echo 'PASS: experimental C5 benchmark; production default remains unchanged.'
