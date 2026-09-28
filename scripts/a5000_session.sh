#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/a5000_session.sh [options]

Build the offline image, run the C5 pipeline, and benchmark one or more model
artifacts during the paid RTX A5000 validation session.

Options:
  --models PATHS   Comma-separated repo-relative model directories
                   (default: artifacts/models/final)
  --verifier FILE  Repo-relative verifier checkpoint
                   (default: artifacts/models/verifier_f0/verifier.pt)
  --output DIR     Host output directory (default: outputs/a5000_session)
  --data DIR       Dataset directory (default: dataset)
  --label LABEL    Result label (default: a5000-<UTC timestamp>)
  --dry-run        Print commands without checking hardware or executing them
  -h, --help       Show this help
EOF
}

models_csv="artifacts/models/final"
verifier_path="artifacts/models/verifier_f0/verifier.pt"
output_root="outputs/a5000_session"
data_root="dataset"
run_label="a5000-$(date -u +%Y%m%dT%H%M%SZ)"
dry_run=0

while (( $# )); do
  case "$1" in
    --models) models_csv="${2:?missing value for --models}"; shift 2 ;;
    --verifier) verifier_path="${2:?missing value for --verifier}"; shift 2 ;;
    --output) output_root="${2:?missing value for --output}"; shift 2 ;;
    --data) data_root="${2:?missing value for --data}"; shift 2 ;;
    --label) run_label="${2:?missing value for --label}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$verifier_path" =~ ^artifacts/models/[A-Za-z0-9._/-]+$ ]] \
    || [[ "$verifier_path" == *".."* ]]; then
  echo "verifier path must be repo-relative under artifacts/models: $verifier_path" >&2
  exit 2
fi

IFS=',' read -r -a model_dirs <<< "$models_csv"
if (( ${#model_dirs[@]} == 0 )); then
  echo "at least one model is required" >&2
  exit 2
fi

for model_dir in "${model_dirs[@]}"; do
  if [[ ! "$model_dir" =~ ^artifacts/models/[A-Za-z0-9._/-]+$ ]] || [[ "$model_dir" == *".."* ]]; then
    echo "model path must be repo-relative under artifacts/models: $model_dir" >&2
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
  command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 1; }
  docker compose version >/dev/null

  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  if ! grep -q 'A5000' <<< "$gpu_names"; then
    echo "expected an RTX A5000 session, found: $gpu_names" >&2
    exit 1
  fi
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
  if (( ${driver%%.*} < 535 )); then
    echo "NVIDIA driver $driver is older than required R535" >&2
    exit 1
  fi
  [[ -f "$data_root/test_query.csv" && -f "$data_root/test_gallery.csv" ]] || {
    echo "dataset CSV files are missing under $data_root" >&2
    exit 1
  }
  [[ -d "$data_root/images" ]] || { echo "image directory is missing: $data_root/images" >&2; exit 1; }
  for model_dir in "${model_dirs[@]}"; do
    [[ -f "$model_dir/model_config.yaml" && -f "$model_dir/weights.safetensors" ]] || {
      echo "incomplete model artifact: $model_dir" >&2
      exit 1
    }
  done
  [[ -f "$verifier_path" ]] || { echo "verifier is missing: $verifier_path" >&2; exit 1; }
  mkdir -p "$output_root"
  nvidia-smi -q > "$output_root/nvidia_smi.txt"
  git rev-parse HEAD > "$output_root/git_commit.txt"
fi

session_started="$(date +%s)"
run env DATA_ROOT="$data_root" OUTPUT_ROOT="$output_root" docker compose build reid

first_model="${model_dirs[0]}"
first_container_model="/app/$first_model"
container_verifier="/app/$verifier_path"
run env \
  DATA_ROOT="$data_root" \
  OUTPUT_ROOT="$output_root" \
  MODEL_DIR="$first_container_model" \
  VERIFIER_PATH="$container_verifier" \
  OUT_DIR=/out/smoke \
  docker compose run --rm \
    --volume "$(pwd)/$first_model:$first_container_model:ro" \
    --volume "$(pwd)/$verifier_path:$container_verifier:ro" reid

for model_dir in "${model_dirs[@]}"; do
  model_name="$(basename "$model_dir")"
  container_model="/app/$model_dir"
  result_file="/out/bench_${model_name}.json"
  run env DATA_ROOT="$data_root" OUTPUT_ROOT="$output_root" \
    docker compose run --rm --entrypoint python3 --env PYTHONPATH=/app \
    --volume "$(pwd)/$model_dir:$container_model:ro" reid \
    scripts/bench_extract.py \
    --model "$container_model" \
    --images /data/images \
    --csv /data/test_query.csv \
    --device cuda:0 \
    --warmup 50 \
    --iterations 300 \
    --batch-sizes 1,8,16,32 \
    --throughput-seconds 10 \
    --run-label "${run_label}-${model_name}" \
    --json-out "$result_file" \
    --require-cuda
done

session_finished="$(date +%s)"
if (( dry_run )); then
  echo "DRY RUN: no hardware checks or commands were executed."
else
  echo "A5000 session completed in $((session_finished - session_started)) seconds."
  echo "Results: $output_root"
  echo "Copy results off the server, record the cost in docs/infra.md, then delete the paid server."
fi
