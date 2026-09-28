#!/usr/bin/env bash
set -Eeuo pipefail

# Reproduce the exact Day 3 inference artifact from the immutable GitHub Release.
# The repository is private during development, so authenticated `gh` is the
# preferred downloader. The curl fallback works if the release is made public.

repo="Timur713/hack_falcon_2026"
tag="model-v1"
weights_asset="vitb14_in280_ep100.safetensors"
config_asset="vitb14_in280_ep100.yaml"
weights_sha256="645d7cbdd7b530958a66c95ac072419f3b3cdd903808997e26069def2fc70cb4"
config_sha256="ead3748ae7e8a13fe978a285cb3d647444070a71517bb412f88fd5718184dd7e"
release_url="https://github.com/${repo}/releases/download/${tag}"
output_dir="${1:-artifacts/models/final}"

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

download_asset() {
  local asset="$1"
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    local attempt
    for attempt in 1 2 3; do
      if gh release download "$tag" --repo "$repo" --pattern "$asset" --dir "$temp_dir" --clobber; then
        return 0
      fi
      rm -f "$temp_dir/$asset"
      sleep "$attempt"
    done
    echo "failed to download $asset after three attempts" >&2
    return 1
  elif command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$temp_dir/$asset" "$release_url/$asset"
  else
    echo "need an authenticated GitHub CLI or curl to download $asset" >&2
    return 1
  fi
}

check_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$path" | awk '{print $1}')"
  else
    actual="$(shasum -a 256 "$path" | awk '{print $1}')"
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "sha256 mismatch for $path: expected $expected, got $actual" >&2
    return 1
  fi
}

download_asset "$weights_asset"
download_asset "$config_asset"
check_sha256 "$weights_sha256" "$temp_dir/$weights_asset"
check_sha256 "$config_sha256" "$temp_dir/$config_asset"

mkdir -p "$output_dir"
install -m 0644 "$temp_dir/$weights_asset" "$output_dir/weights.safetensors"
install -m 0644 "$temp_dir/$config_asset" "$output_dir/model_config.yaml"

echo "model-v1 installed in $output_dir"
echo "weights sha256: $weights_sha256"
