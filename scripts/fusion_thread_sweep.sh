#!/usr/bin/env bash
# Serial CPU-thread/batch profiling on TRAIN images. No model/quality changes.
set -Eeuo pipefail
session_root="${SESSION_ROOT:?Set isolated session directory}"
data_root="${DATA_ROOT:?Set dataset directory}"
image="${FUSION_BASE_IMAGE:-falcon-runtime:20260927}"
report_root="$session_root/outputs/thread_sweep"
[[ ! -e "$report_root" ]] || { echo 'Refusing existing report directory' >&2; exit 2; }
mkdir -p "$report_root"
for spec in 1:8 2:8 1:4; do
  threads="${spec%:*}"
  workers="${spec#*:}"
  docker run --rm --gpus all --network none --entrypoint python3 \
    -e PYTHONPATH=/app -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$session_root/src:/app:ro" -v "$session_root/artifacts:/artifacts:ro" \
    -v "$data_root:/data:ro" -v "$report_root:/out" -w /app "$image" \
    scripts/bench_extract.py --model /artifacts/fusion_all_d063 \
    --images /data/images --csv /app/splits/fold0_query.csv --device cuda:0 \
    --torch-threads "$threads" --preprocess-workers "$workers" \
    --warmup 50 --iterations 300 --batch-sizes 1,8,16,32,64,128 \
    --throughput-seconds 10 --require-cuda \
    --run-label "A5000-fusion3-t${threads}-w${workers}" \
    --json-out "/out/threads${threads}_workers${workers}.json"
done
echo 'PASS: serial thread/batch sweep; no production default changed by this script.'
