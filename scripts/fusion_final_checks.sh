#!/usr/bin/env bash
# Run only after fusion_offline_checks has finished: never overlap GPU benchmarks.
set -Eeuo pipefail
session_root="${SESSION_ROOT:?Set isolated session directory}"
data_root="${DATA_ROOT:?Set existing dataset directory}"
report_root="$session_root/outputs/final_checks"
[[ -f "$data_root/test_query.csv" && -d "$data_root/images" && ! -e "$report_root" ]] \
  || { echo 'Missing dataset or output already exists' >&2; exit 2; }
mkdir -p "$report_root"
run_python() {
  local image="$1"
  shift
  docker run --rm --gpus all --network none --entrypoint python3 -e PYTHONPATH=/app \
    -v "$data_root:/data:ro" -v "$report_root:/out" "$image" "$@"
}
set +e
run_python falcon-fusion:oof-v0 scripts/audit_release.py weights --root /app \
  --model /app/artifacts/models/fusion --verifier /app/artifacts/models/fusion/verifier.pt \
  --require-all-data --report /out/oof_gate.json > "$report_root/oof_gate.txt" 2>&1
gate_status=$?
set -e
printf 'exit_code=%s\n' "$gate_status" >> "$report_root/oof_gate.txt"
[[ "$gate_status" == 1 ]] || { echo 'Unexpected OOF gate result'; exit 1; }
grep -q 'final embedder must be trained on all folds' "$report_root/oof_gate.txt"
run_python falcon-fusion:all-v1-technical scripts/audit_release.py weights --root /app \
  --model /app/artifacts/models/fusion --verifier /app/artifacts/models/fusion/verifier.pt \
  --require-all-data --report /out/all_data_gate.json
run_python falcon-fusion:oof-v0 scripts/bench_extract.py \
  --model /app/artifacts/models/fusion --images /data/images --csv /app/splits/fold0_query.csv \
  --device cuda:0 --warmup 50 --iterations 300 --batch-sizes 1,8,16,32 \
  --throughput-seconds 10 --require-cuda --run-label A10-built-offline-fusion3-OOF \
  --json-out /out/built_image_bench.json
echo 'PASS: final metadata gates and self-contained image extraction benchmark.'
