#!/usr/bin/env bash

set -euo pipefail
REPO_DIR="${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
if [[ ! -f "${REPO_DIR}/attacks/scripts/run_attack.py" ]]; then
  echo "Submit from the benchmark repository root, or set REPO_DIR." >&2
  exit 2
fi
export REPO_DIR
export ATTACK=qedks
export BUDGET=1000
export METHOD=adfp
export COUNTERMEASURE=dipper
export PREPARE_COUNTER_BASELINES=1
export COUNTER_BASELINES=1
export COUNTER_CLEAN_ONLY=1
export REQUIRE_COUNTER_BASELINES=0
# Isolate diagnostics from existing completed or partially written baselines.
export STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
export COUNTER_BASELINE_ROOT="${COUNTER_BASELINE_ROOT:-${STORAGE_ROOT}/outputs/counter_diagnostics/${RUN_ID:-local}}"
export MEA_GENERATION_DIAGNOSTICS="${MEA_GENERATION_DIAGNOSTICS:-1}"
export QEDKS_REQUEST_TIMEOUT_SECONDS="${QEDKS_REQUEST_TIMEOUT_SECONDS:-3600}"

mkdir -p "${REPO_DIR}/logs"
GPU_LOG="${REPO_DIR}/logs/gpu_${RUN_ID:-local}.csv"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.sm,clocks.current.memory \
    --format=csv -l 2 > "${GPU_LOG}" 2> "${GPU_LOG%.csv}.err" &
  GPU_MONITOR_PID=$!
  cleanup_gpu_monitor() {
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  }
  trap cleanup_gpu_monitor EXIT
  echo "GPU samples (every 2 seconds): ${GPU_LOG}"
else
  echo "GPU sampling unavailable: nvidia-smi was not found." >&2
fi
source "${REPO_DIR}/runs/counter/common.sh"
