#!/usr/bin/env bash
# D-063 all-data artifact + serial C5 comparison. Run detached under a wall-time limit.
# Provider Stop/Shelve is a separate operation: guest shutdown does not end billing.
set -Eeuo pipefail
session_root="${SESSION_ROOT:?Set isolated session directory}"
data_root="${DATA_ROOT:?Set existing dataset directory}"
image="${FUSION_BASE_IMAGE:-falcon-runtime:20260927}"
report_root="$session_root/outputs/final_prepare"
cache="$session_root/artifacts/release-cache/fusion-all-v1"
[[ ! -e "$report_root" ]] || { echo 'Refusing existing report directory' >&2; exit 2; }
mkdir -p "$report_root" "$session_root/artifacts/all_models"
# Independent of the stale upstream SHA256SUMS.txt: pin the accepted API digests.
verify() { echo "$2  $cache/$1" | sha256sum --check --strict; }
verify hires392_all.safetensors f69eb2283c03a2748ac44cb4e1a4902bbc6f3129440cf3799edafa0f0ea0aaeb
verify div_clip_all.safetensors 6157e1eb862a0c88f637ed2a163bf6bbbe52459a88d41f3b623e83e69180548f
verify div_siglip_all.safetensors 6686c5e495b73c084b774c4cd4da9c50a9df300b184b1b70ad8b4d0946f1137b
verify v_fus_all_frz_input392.pt aa254e396ac0be78d29f32641e83f8d18e165d76eb3345c18c1054d105b9abf5
verify hires392_all.yaml 8e5fe32a0d56b960caea8d13769e1d0d48a7b1a3bc4672a464c30c24e60ce12c
verify div_clip_all.yaml b4d5b93d0c633ff0083876ccdfc8436cfd54a15e3270078108366ad0f5038a11
verify div_siglip_all.yaml 37204d110f1012408fa438d4b583ed429572f64bf232b6444f80b35b8dc15ec6
for component in hires392 div_clip div_siglip; do
  dest="$session_root/artifacts/all_models/$component"
  [[ ! -e "$dest" ]] || { echo "Refusing existing component $dest" >&2; exit 2; }
  mkdir "$dest"
  cp "$cache/${component}_all.yaml" "$dest/model_config.yaml"
  cp "$cache/${component}_all.safetensors" "$dest/weights.safetensors"
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
  --component /artifacts/all_models/hires392 --component /artifacts/all_models/div_clip \
  --component /artifacts/all_models/div_siglip \
  --images /data/images --gallery /app/splits/fold0_gallery.csv \
  --out /artifacts/fusion_all_d063 --name fusion-all-d063-20260927 \
  --device cuda:0 --batch-size 32 --rerank-weight 0.15 \
  --candidate-threshold 6.3 --threshold-source 'D-063 / A-103 exact-window OOF reports; full-frame transfer remains to be audited' \
  --verifier /artifacts/release-cache/fusion-all-v1/v_fus_all_frz_input392.pt --require-all-data
bench() {
  run_python scripts/bench_extract.py --model "$1" --images /data/images \
    --csv /app/splits/fold0_query.csv --device cuda:0 --warmup 50 --iterations 300 \
    --batch-sizes 1,8,16,32 --throughput-seconds 10 --require-cuda \
    --run-label "$2" --json-out "/out/$3.json"
}
bench /artifacts/fusion_all_d063 A5000-fusion3-all-D063 trio
run_python scripts/audit_release.py weights --root /artifacts/fusion_all_d063 \
  --model /artifacts/fusion_all_d063 --verifier /artifacts/fusion_all_d063/verifier.pt \
  --report /out/weights_audit.json --require-all-data
# Pair is a SPEED probe, not a calibrated/quality-selected release candidate.
run_python scripts/prepare_fusion_artifact.py \
  --component /artifacts/all_models/hires392 --component /artifacts/all_models/div_clip \
  --images /data/images --gallery /app/splits/fold0_gallery.csv \
  --out /artifacts/fusion_all_pair_probe --name fusion-pair-speed-probe \
  --device cuda:0 --batch-size 32 --rerank-weight 0.15 \
  --verifier /artifacts/release-cache/fusion-all-v1/v_fus_all_frz_input392.pt --require-all-data
bench /artifacts/fusion_all_pair_probe A5000-fusion2-speed-probe pair
bench /artifacts/fusion_all_d063 A5000-fusion3-all-D063-repeat trio_repeat
echo 'PASS: SHA-pinned all-data packaging, serial C5 comparison, all-data weight audit.'
echo 'No final quality or submission acceptance claim. Backup and release provider GPU resources.'
