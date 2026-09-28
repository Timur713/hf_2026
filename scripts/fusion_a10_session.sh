#!/usr/bin/env bash
# A bounded, detached GPU run. GPU billing must be stopped by provider Shelve after backup.
set -Eeuo pipefail
session_root="${SESSION_ROOT:?Set the isolated session directory}"
data_root="${DATA_ROOT:?Set the existing read-only dataset directory}"
image="${FUSION_BASE_IMAGE:-falcon-reid:local}"
mode="${1:?oof or all}"
case "$mode" in
  oof) tag=fusion-oof-v0; suffix=f0; verifier=v_fus_f1only.pt; weight=0.15; extra=() ;;
  all) tag=fusion-all-v1; suffix=all; verifier=v_fus_all_input168.pt; weight=0; extra=(--require-all-data) ;;
  *) echo 'Expected oof or all' >&2; exit 2 ;;
esac
report_root="$session_root/outputs/$mode"
[[ ! -e "$report_root" ]] || { echo "Refusing existing output $report_root" >&2; exit 2; }
mkdir -p "$report_root" "$session_root/artifacts/${mode}_models"
cache="$session_root/artifacts/release-cache/$tag"
(cd "$cache" && sha256sum --ignore-missing -c SHA256SUMS.txt) | tee "$report_root/release_sha256.txt"
for component in hires392 div_clip div_siglip; do
  dest="$session_root/artifacts/${mode}_models/$component"
  [[ ! -e "$dest" ]] || { echo "Refusing existing component $dest" >&2; exit 2; }
  mkdir "$dest"
  cp "$cache/${component}_${suffix}.yaml" "$dest/model_config.yaml"
  cp "$cache/${component}_${suffix}.safetensors" "$dest/weights.safetensors"
done
nvidia-smi -q > "$report_root/nvidia_smi.txt"
docker inspect "$image" > "$report_root/base_image_inspect.json"
run_python() {
  docker run --rm --gpus all --network none --entrypoint python3 \
    -e PYTHONPATH=/app -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$session_root/src:/app:ro" -v "$session_root/artifacts:/artifacts" \
    -v "$data_root:/data:ro" -v "$report_root:/out" -w /app "$image" "$@"
}
run_python scripts/prepare_fusion_artifact.py \
  --component "/artifacts/${mode}_models/hires392" \
  --component "/artifacts/${mode}_models/div_clip" \
  --component "/artifacts/${mode}_models/div_siglip" \
  --images /data/images --gallery /app/splits/fold0_gallery.csv \
  --out "/artifacts/fusion_$mode" --name "fusion-$mode-technical-20260925" \
  --device cuda:0 --batch-size 32 --rerank-weight "$weight" \
  --verifier "/artifacts/release-cache/$tag/$verifier" "${extra[@]}"
# This is the first expensive measurement: no need to wait for a final verifier.
run_python scripts/bench_extract.py --model "/artifacts/fusion_$mode" \
  --images /data/images --csv /app/splits/fold0_query.csv --device cuda:0 \
  --warmup 50 --iterations 300 --batch-sizes 1,8,16,32 --throughput-seconds 10 \
  --require-cuda --run-label "A10-fusion3-$mode" --json-out /out/bench_extract.json
run_python scripts/audit_release.py weights --root "/artifacts/fusion_$mode" \
  --model "/artifacts/fusion_$mode" --verifier "/artifacts/fusion_$mode/verifier.pt" \
  --report /out/weights_audit.json "${extra[@]}"
echo 'PASS: bundle, extraction benchmark and weight audit. Backup then provider Shelve.'
