#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/a10_model_matrix.sh [options]

Build zero-shot DINOv2 S/B/L artifacts for 224/256/320 inputs and benchmark
the complete C5 extraction zone on an NVIDIA A10. Existing complete artifacts
are reused.

Options:
  --data DIR       Dataset directory (default: dataset)
  --output DIR     Benchmark output directory (default: outputs/a10_model_matrix)
  --python PATH    Python with project dependencies (default: .venv/bin/python)
  --label LABEL    Run label prefix (default: a10-matrix-<UTC timestamp>)
  --dry-run        Print the queue without hardware checks or execution
  -h, --help       Show this help
EOF
}

data_root="dataset"
output_root="outputs/a10_model_matrix"
python_bin=".venv/bin/python"
run_label="a10-matrix-$(date -u +%Y%m%dT%H%M%SZ)"
dry_run=0

while (( $# )); do
  case "$1" in
    --data) data_root="${2:?missing value for --data}"; shift 2 ;;
    --output) output_root="${2:?missing value for --output}"; shift 2 ;;
    --python) python_bin="${2:?missing value for --python}"; shift 2 ;;
    --label) run_label="${2:?missing value for --label}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

for path in "$data_root" "$output_root"; do
  if [[ "$path" == /* || "$path" == *".."* ]]; then
    printf 'paths must be repo-relative and may not contain ..: %s\n' "$path" >&2
    exit 2
  fi
done

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_command "$@"
  if (( ! dry_run )); then
    "$@"
  fi
}

if (( ! dry_run )); then
  command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is required' >&2; exit 1; }
  [[ -x "$python_bin" ]] || { echo "Python is missing: $python_bin" >&2; exit 1; }
  [[ -f "$data_root/test_query.csv" && -d "$data_root/images" ]] || {
    echo "dataset is incomplete: $data_root" >&2
    exit 1
  }
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  [[ "$gpu_name" == *"A10"* ]] || { echo "expected NVIDIA A10, found: $gpu_name" >&2; exit 1; }
  mkdir -p "$output_root"
fi

arches=(dinov2_vits14 dinov2_vitb14 dinov2_vitl14)
sizes=(224 256 320)
models=()

for arch in "${arches[@]}"; do
  for size in "${sizes[@]}"; do
    name="${arch}_zeroshot_${size}_cls"
    model_dir="artifacts/models/${name}"
    models+=("$model_dir")
    if (( ! dry_run )) && [[ -f "$model_dir/model_config.yaml" && -f "$model_dir/weights.safetensors" ]]; then
      printf 'Reusing complete artifact: %s\n' "$model_dir"
    else
      run env PYTHONPATH=. "$python_bin" scripts/make_pretrained_artifact.py \
        --arch "$arch" --input "$size" --pool cls --name "$name"
    fi
  done
done

for model_dir in "${models[@]}"; do
  name="$(basename "$model_dir")"
  run env PYTHONPATH=. "$python_bin" scripts/bench_extract.py \
    --model "$model_dir" \
    --images "$data_root/images" \
    --csv "$data_root/test_query.csv" \
    --device cuda:0 \
    --preprocess-workers 8 \
    --warmup 50 \
    --iterations 300 \
    --batch-sizes 1,8,16,32 \
    --throughput-seconds 10 \
    --run-label "${run_label}-${name}" \
    --json-out "$output_root/${name}.json" \
    --require-cuda
done

if (( dry_run )); then
  echo 'DRY RUN: no artifacts were created and no benchmarks were executed.'
else
  echo "A10 model matrix complete. Results: $output_root"
fi
