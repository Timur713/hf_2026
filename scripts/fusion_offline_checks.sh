#!/usr/bin/env bash
# Verify an already built, self-contained fusion image; no source/model bind mounts.
set -Eeuo pipefail
image="${FUSION_IMAGE:?Set the built fusion image}"
data_root="${DATA_ROOT:?Set the existing dataset directory}"
output_root="${OUTPUT_ROOT:?Set a NEW output directory}"
[[ ! -e "$output_root" ]] || { echo "Refusing existing output: $output_root" >&2; exit 2; }
mkdir -p "$output_root"
docker image inspect "$image" > "$output_root/image_inspect.json"
run_python() {
  docker run --rm --gpus all --network none --entrypoint python3 \
    -e PYTHONPATH=/app -v "$data_root:/data:ro" -v "$output_root:/out" \
    "$image" "$@"
}
docker run --rm --gpus all --network none --entrypoint make \
  -e PYTHONPATH=/app -v "$data_root:/data:ro" -v "$output_root:/out" \
  "$image" check PYTHON=python3 DATA_ROOT=/data
run_python -m reid.infer.smoke --data-root /data --n-images 50 --require-runner \
  --model /app/artifacts/models/fusion --verifier /app/artifacts/models/fusion/verifier.pt
# Full official-shaped inputs, format only: no test labels, tuning or error inspection.
run_python scripts/check_determinism.py --images /data/images \
  --query /data/test_query.csv --gallery /data/test_gallery.csv \
  --model /app/artifacts/models/fusion --verifier /app/artifacts/models/fusion/verifier.pt \
  --device cuda:0 --batch-size 32 --out /out/determinism
run_python scripts/audit_release.py submission --query /data/test_query.csv \
  --gallery /data/test_gallery.csv --submission-dir /out/determinism/run_1 \
  --report /out/c6.json
# Only train-fold labels are used for accuracy or masking diagnostics.
run_python -m reid.infer.run --images /data/images --query /app/splits/fold0_query.csv \
  --gallery /app/splits/fold0_gallery.csv --model /app/artifacts/models/fusion \
  --verifier /app/artifacts/models/fusion/verifier.pt --device cuda:0 \
  --out /out/fold0
run_python scripts/run_plate_mask_control.py --images /data/images \
  --query /app/splits/fold0_query.csv --gallery /app/splits/fold0_gallery.csv \
  --model /app/artifacts/models/fusion --verifier /app/artifacts/models/fusion/verifier.pt \
  --device cuda:0 --out /out/fold0_masked --report /out/mask_control.json
if [[ "${EVALUATE_OOF:-0}" == 1 ]]; then
  for variant in fold0 fold0_masked; do
    run_python -m reid.eval.eval_local --fold 0 --submission "/out/$variant/submission.csv" \
      --candidates "/out/$variant/candidates.csv" --out "/out/${variant}_metrics.json"
  done
fi
echo 'PASS: offline image tests, full C6 outputs, determinism and approximate plate-area control.'
echo 'Mask coverage is not certified; all-data outputs are not OOF quality estimates.'
