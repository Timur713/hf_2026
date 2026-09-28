#!/usr/bin/env bash
# Build from a verified local bundle and execute the bounded offline acceptance.
# This does NOT implement provider Shelve; backup and stop billing explicitly.
set -Eeuo pipefail
session_root="${SESSION_ROOT:?Set isolated session directory}"
data_root="${DATA_ROOT:?Set existing dataset directory}"
[[ -f "$data_root/test_query.csv" && -f "$data_root/test_gallery.csv" && -d "$data_root/images" ]] \
  || { echo 'Dataset preflight failed; refusing to build/run' >&2; exit 2; }
mode="${1:?oof or all}"
case "$mode" in
  oof) tag=fusion-oof-v0; image=falcon-fusion:oof-v0; evaluate=1 ;;
  all) tag=fusion-all-v1; image=falcon-fusion:all-v1-technical; evaluate=0 ;;
  *) exit 2 ;;
esac
source_root="$session_root/src"
bundle="$session_root/artifacts/fusion_$mode"
# Staging is explicit to avoid deleting/replacing an earlier bundle by mistake.
cmp "$bundle/manifest.json" "$source_root/artifacts/fusion_bundle/manifest.json"
digest=$(sha256sum "$bundle/manifest.json" | cut -d ' ' -f 1)
[[ ! -e "$session_root/outputs/offline_$mode" ]] || { echo 'Output already exists' >&2; exit 2; }
cd "$source_root"
docker build --network "${BUILD_NETWORK:-default}" --pull=false --progress=plain --target fusion \
  --build-arg "FUSION_MANIFEST_SHA256=$digest" \
  --build-arg "FUSION_RELEASE_URL=https://github.com/Timur713/hack_falcon_2026/releases/tag/$tag" \
  -t "$image" .
FUSION_IMAGE="$image" DATA_ROOT="$data_root" \
  OUTPUT_ROOT="$session_root/outputs/offline_$mode" EVALUATE_OOF="$evaluate" \
  bash scripts/fusion_offline_checks.sh
