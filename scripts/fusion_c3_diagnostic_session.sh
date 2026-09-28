#!/usr/bin/env bash
# Fixed fold0 OOF checkpoints, no verifier, no test data or tuning.
set -Eeuo pipefail
session_root="${SESSION_ROOT:?Set isolated session directory}"
data_root="${DATA_ROOT:?Set dataset directory}"
image="${FUSION_BASE_IMAGE:-falcon-runtime:20260927}"
cache="$session_root/artifacts/release-cache/fusion-oof-v0"
[[ ! -e "$session_root/outputs/c3_gap" ]] || { echo 'Refusing existing diagnostic output' >&2; exit 2; }
(cd "$cache" && sha256sum --check --strict --ignore-missing SHA256SUMS.txt)
mkdir -p "$session_root/artifacts/oof_models"
for component in hires392 div_clip div_siglip; do
  dest="$session_root/artifacts/oof_models/$component"
  [[ ! -e "$dest" ]] || { echo "Refusing existing component $dest" >&2; exit 2; }
  mkdir "$dest"
  cp "$cache/${component}_f0.yaml" "$dest/model_config.yaml"
  cp "$cache/${component}_f0.safetensors" "$dest/weights.safetensors"
done
docker run --rm --gpus all --network none --entrypoint python3 \
  -e PYTHONPATH=/app -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$session_root/src:/app:ro" -v "$session_root/artifacts:/artifacts:ro" \
  -v "$data_root:/data:ro" -v "$session_root/outputs:/out" -w /app "$image" \
  scripts/diagnose_c3_gap.py --component /artifacts/oof_models/hires392 \
  --component /artifacts/oof_models/div_clip --component /artifacts/oof_models/div_siglip \
  --data-root /data --fold 0 --out /out/c3_gap --device cuda:0 --batch 32 --workers 8 \
  --similarity-std 0.05278994732695534 --similarity-std 0.07144717974866033 \
  --similarity-std 0.002510766842444479
echo 'PASS: train-only fixed-scale precision/cache diagnostic; see results, not a release gate.'
