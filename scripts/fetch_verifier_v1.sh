#!/usr/bin/env bash
set -Eeuo pipefail

# Reproduce the exact Day 3 pair-verifier from the immutable GitHub Release.
# The repository is private during development, so authenticated `gh` is the
# preferred downloader. The curl fallback works if the release is made public.

repo="Timur713/hack_falcon_2026"
tag="model-v1"
asset="verifier_f0.pt"
sha256="52fc9b111351dbcd021ddf221b6546761e1609a96beeb5b9697c3e03585667c6"
release_url="https://github.com/${repo}/releases/download/${tag}/${asset}"
output_dir="${1:-artifacts/models/verifier_f0}"

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  for attempt in 1 2 3; do
    if gh release download "$tag" --repo "$repo" --pattern "$asset" \
        --dir "$temp_dir" --clobber; then
      break
    fi
    rm -f "$temp_dir/$asset"
    if (( attempt == 3 )); then
      echo "failed to download $asset after three attempts" >&2
      exit 1
    fi
    sleep "$attempt"
  done
elif command -v curl >/dev/null 2>&1; then
  curl --fail --location --retry 3 --output "$temp_dir/$asset" "$release_url"
else
  echo "need an authenticated GitHub CLI or curl to download $asset" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$temp_dir/$asset" | awk '{print $1}')"
else
  actual="$(shasum -a 256 "$temp_dir/$asset" | awk '{print $1}')"
fi
if [[ "$actual" != "$sha256" ]]; then
  echo "sha256 mismatch for $asset: expected $sha256, got $actual" >&2
  exit 1
fi

mkdir -p "$output_dir"
install -m 0644 "$temp_dir/$asset" "$output_dir/verifier.pt"

echo "verifier-v1 installed in $output_dir"
echo "sha256: $sha256"
