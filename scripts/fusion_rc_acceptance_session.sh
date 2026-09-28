#!/usr/bin/env bash
# Self-contained final-pair technical acceptance. No all-data quality evaluation.
set -Eeuo pipefail
context="${BUILD_CONTEXT:?Set a tracked-source build context with verified archive cache}"
data_root="${DATA_ROOT:?Set existing dataset directory}"
output_root="${OUTPUT_ROOT:?Set NEW output directory}"
image="${FUSION_IMAGE:-falcon-fusion:pair-d065-rc1}"
[[ ! -e "$output_root" ]] || { echo 'Refusing existing output directory' >&2; exit 2; }
# The specification permits network at build time, but never during inference.
docker build --progress plain -t "$image" "$context"
FUSION_IMAGE="$image" DATA_ROOT="$data_root" OUTPUT_ROOT="$output_root" \
  EVALUATE_OOF=0 bash "$context/scripts/fusion_offline_checks.sh"
run_python() {
  docker run --rm --gpus all --network none --entrypoint python3 \
    -e PYTHONPATH=/app -v "$data_root:/data:ro" -v "$output_root:/out" "$image" "$@"
}
run_python scripts/audit_release.py weights --root /app \
  --model /app/artifacts/models/fusion --verifier /app/artifacts/models/fusion/verifier.pt \
  --require-all-data --report /out/all_data_weights.json
run_python scripts/bench_extract.py --model /app/artifacts/models/fusion \
  --images /data/images --csv /app/splits/fold0_query.csv --device cuda:0 \
  --warmup 50 --iterations 300 --batch-sizes 1,8,16,32 --throughput-seconds 10 \
  --require-cuda --run-label "${BENCH_LABEL:-final-pair-self-contained}" \
  --json-out /out/benchmark.json
# Content replacement at the same tensor shape is distinct from cross-shape
# FP16 rounding. Preserve both reports; neither silently replaces the other.
run_python scripts/check_query_content_isolation.py \
  --model /app/artifacts/models/fusion --verifier /app/artifacts/models/fusion/verifier.pt \
  --data-root /data --fold 0 --device cuda:0 --workers 8 --torch-threads 1 \
  --out /out/query_content_isolation
# Keep a strict numeric/semantic difference report, not a false PASS. A
# confidence-only FP16 ULP difference does not prove use of other queries.
if run_python scripts/check_query_independence.py \
  --model /app/artifacts/models/fusion --verifier /app/artifacts/models/fusion/verifier.pt \
  --data-root /data --fold 0 --queries 24 --extras 8 --device cuda:0 \
  --workers 8 --torch-threads 1 --out /out/query_independence; then
  echo 'Query-independence strict numeric check: PASS'
else
  echo 'Query-independence check: nonzero exit; inspect report/errors before acceptance.' >&2
  exit 1
fi
echo 'PASS: technical RC acceptance only; quality/provenance/mask-coverage gates are separate.'
