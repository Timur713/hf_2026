#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/a10_acceptance.sh [options]

Accept the Day-1/Day-2 track-B deliverables on an NVIDIA A10: run the local
test suite and real-image smoke, benchmark all decoder backends and the full
C5 extraction zone, then build and run the Docker image without networking.

Options:
  --data DIR       Dataset or representative smoke subset (default: dataset)
  --model DIR      Repo-relative model artifact (default: zero-shot ViT-B/224)
  --verifier FILE  Repo-relative verifier checkpoint
                   (default: artifacts/models/verifier_f0/verifier.pt)
  --output DIR     Result directory (default: outputs/a10_acceptance)
  --python PATH    Python with project dependencies (default: .venv/bin/python)
  --label LABEL    Benchmark label (default: a10-<UTC timestamp>)
  --dry-run        Print commands without running hardware checks or work
  -h, --help       Show this help
EOF
}

data_root="dataset"
model_dir="artifacts/models/dinov2_vitb14_zeroshot_224_cls"
verifier_path="artifacts/models/verifier_f0/verifier.pt"
output_root="outputs/a10_acceptance"
python_bin=".venv/bin/python"
run_label="a10-$(date -u +%Y%m%dT%H%M%SZ)"
dry_run=0

while (( $# )); do
  case "$1" in
    --data) data_root="${2:?missing value for --data}"; shift 2 ;;
    --model) model_dir="${2:?missing value for --model}"; shift 2 ;;
    --verifier) verifier_path="${2:?missing value for --verifier}"; shift 2 ;;
    --output) output_root="${2:?missing value for --output}"; shift 2 ;;
    --python) python_bin="${2:?missing value for --python}"; shift 2 ;;
    --label) run_label="${2:?missing value for --label}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for path in "$data_root" "$model_dir" "$verifier_path" "$output_root"; do
  if [[ "$path" == /* || "$path" == *".."* ]]; then
    printf 'paths must be repo-relative and may not contain ..: %s\n' "$path" >&2
    exit 2
  fi
done

print_command() (
  # Bash 3.2's %q needs a separate locale assignment for non-ASCII paths.
  export LC_ALL=C
  printf '+'
  printf ' %q' "$@"
  printf '\n'
)

run() {
  print_command "$@"
  if (( ! dry_run )); then
    "$@"
  fi
}

if (( ! dry_run )); then
  command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is required' >&2; exit 1; }
  [[ -x "$python_bin" ]] || { echo "Python is missing: $python_bin" >&2; exit 1; }
  [[ -f "$data_root/train.csv" && -f "$data_root/test_query.csv" \
      && -f "$data_root/test_gallery.csv" && -d "$data_root/images" ]] || {
    echo "dataset is incomplete: $data_root" >&2
    exit 1
  }
  [[ -f "$model_dir/model_config.yaml" && -f "$model_dir/weights.safetensors" ]] || {
    echo "model artifact is incomplete: $model_dir" >&2
    exit 1
  }
  [[ -f "$verifier_path" ]] || { echo "verifier is missing: $verifier_path" >&2; exit 1; }
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  [[ "$gpu_name" == *"A10"* ]] || { echo "expected NVIDIA A10, found: $gpu_name" >&2; exit 1; }
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
  (( ${driver%%.*} >= 535 )) || { echo "NVIDIA driver is older than R535: $driver" >&2; exit 1; }
  "$python_bin" -c 'import nvidia.dali, torch, torchvision; assert torch.cuda.is_available()'
  mkdir -p "$output_root"
  nvidia-smi -q > "$output_root/nvidia_smi.txt"
fi

if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo docker)
fi

run make check PYTHON="$python_bin" DATA_ROOT="$data_root"

run "$python_bin" scripts/bench_decoders.py \
  --csv "$data_root/train.csv" \
  --images "$data_root/images" \
  --methods pil_full,pil_draft,torchvision_cuda,dali_roi \
  --batch-sizes 1,32 \
  --warmup 50 \
  --iterations 300 \
  --shuffle \
  --run-label "${run_label}-decoders" \
  --format markdown \
  --json-out "$output_root/decoders.json" \
  --fail-on-unavailable

run env PYTHONPATH=. "$python_bin" scripts/bench_extract.py \
  --model "$model_dir" \
  --images "$data_root/images" \
  --csv "$data_root/test_query.csv" \
  --device cuda:0 \
  --warmup 50 \
  --iterations 300 \
  --batch-sizes 1,8,16,32 \
  --throughput-seconds 10 \
  --run-label "${run_label}-extract" \
  --json-out "$output_root/extract.json" \
  --require-cuda

run env DATA_ROOT="$data_root" OUTPUT_ROOT="$output_root" \
  "${docker_command[@]}" compose build reid

container_model="/app/$model_dir"
container_verifier="/app/$verifier_path"
run env \
  DATA_ROOT="$data_root" \
  OUTPUT_ROOT="$output_root" \
  MODEL_DIR="$container_model" \
  VERIFIER_PATH="$container_verifier" \
  OUT_DIR=/out/docker_smoke \
  "${docker_command[@]}" compose run --rm \
    --volume "$(pwd)/$model_dir:$container_model:ro" \
    --volume "$(pwd)/$verifier_path:$container_verifier:ro" reid

if (( dry_run )); then
  echo 'DRY RUN: no hardware checks or commands were executed.'
else
  echo "A10 acceptance complete. Results: $output_root"
fi
