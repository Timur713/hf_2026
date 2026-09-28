#!/usr/bin/env bash
# Same C5 workload, serial runs; do not compare these A10 numbers to A5000 as fact.
set -Eeuo pipefail
session_root="${SESSION_ROOT:?Set session directory}"
data_root="${DATA_ROOT:?Set dataset directory}"
image="${FUSION_BASE_IMAGE:-falcon-reid:local}"
report_root="$session_root/outputs/compare"
[[ ! -e "$report_root" ]] || { echo 'Refusing existing comparison' >&2; exit 2; }
mkdir -p "$report_root"
run_python() {
  docker run --rm --gpus all --network none --entrypoint python3 \
    -e PYTHONPATH=/app -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$session_root/src:/app:ro" -v "$session_root/artifacts:/artifacts" \
    -v "$data_root:/data:ro" -v "$report_root:/out" -w /app "$image" "$@"
}
bench() {
  run_python scripts/bench_extract.py --model "$1" --images /data/images \
    --csv /app/splits/fold0_query.csv --device cuda:0 --warmup 50 --iterations 300 \
    --batch-sizes 1,8,16,32 --throughput-seconds 10 --require-cuda \
    --run-label "$2" --json-out "/out/$3.json"
}
bench /artifacts/fusion_oof A10-fusion3-oof-repeat trio_repeat
# DINO392+CLIP288: speed fallback only, not a quality-selected replacement.
run_python scripts/prepare_fusion_artifact.py \
  --component /artifacts/oof_models/hires392 --component /artifacts/oof_models/div_clip \
  --images /data/images --gallery /app/splits/fold0_gallery.csv \
  --out /artifacts/fusion_oof_pair --name fusion-pair-speed-probe \
  --device cuda:0 --batch-size 32 --rerank-weight 0.15 \
  --verifier /artifacts/release-cache/fusion-oof-v0/v_fus_f1only.pt
bench /artifacts/fusion_oof_pair A10-DINO392-CLIP288 pair
