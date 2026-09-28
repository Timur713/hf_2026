#!/usr/bin/env bash
set -Eeuo pipefail

threshold="${GPU_IDLE_THRESHOLD:-10}"
idle_limit_seconds="${GPU_IDLE_LIMIT_SECONDS:-3600}"
state_file="${GPU_IDLE_STATE_FILE:-/var/tmp/falcon-gpu-idle-since}"
alert_file="${GPU_IDLE_ALERT_FILE:-/var/tmp/falcon-gpu-idle-alerted}"
log_file="${GPU_IDLE_LOG_FILE:-/var/log/falcon-gpu-idle.log}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    message="$(date --iso-8601=seconds) WARNING: nvidia-smi is unavailable"
    printf '%s\n' "$message" >>"$log_file"
    logger -p user.warning -t falcon-gpu-watchdog -- "$message" || true
    exit 0
fi

max_utilization="$({
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
} | awk 'BEGIN { max = 0 } { if ($1 + 0 > max) max = $1 + 0 } END { print max }')"

if (( max_utilization >= threshold )); then
    rm -f -- "$state_file" "$alert_file"
    exit 0
fi

now="$(date +%s)"
if [[ ! -s "$state_file" ]]; then
    printf '%s\n' "$now" >"$state_file"
    exit 0
fi

idle_since="$(<"$state_file")"
if [[ ! "$idle_since" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$now" >"$state_file"
    rm -f -- "$alert_file"
    exit 0
fi

idle_seconds=$((now - idle_since))
if (( idle_seconds >= idle_limit_seconds )) && [[ ! -e "$alert_file" ]]; then
    message="$(date --iso-8601=seconds) WARNING: GPU utilization stayed below ${threshold}% for $((idle_seconds / 60)) minutes; shelve or delete the server if no job is queued"
    printf '%s\n' "$message" >>"$log_file"
    logger -p user.warning -t falcon-gpu-watchdog -- "$message" || true
    : >"$alert_file"
fi
