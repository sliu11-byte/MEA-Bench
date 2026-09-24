#!/usr/bin/env bash

set -euo pipefail
REPO_DIR="${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
cd "${REPO_DIR}"
command -v module >/dev/null 2>&1 && module purge || true
command -v module >/dev/null 2>&1 && module load conda/25.7.0 || true
command -v module >/dev/null 2>&1 && module load cuda/12.4.1 || true
command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV:-research}" || true
if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi
export OMP_NUM_THREADS="${CPU_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export MEA_GENERATION_DIAGNOSTICS=1
export STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
PROBE_OUT="${STORAGE_ROOT}/outputs/teacher_diagnostics/${RUN_ID:-local}"
mkdir -p "${PROBE_OUT}"
echo "Diagnostics: ${PROBE_OUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi > "${PROBE_OUT}/gpu_initial.txt"
nvidia-smi topo -m > "${PROBE_OUT}/gpu_topology.txt" 2>&1 || true
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.sm,clocks.current.memory \
  --format=csv -l 2 > "${PROBE_OUT}/gpu.csv" 2> "${PROBE_OUT}/gpu.err" &
GPU_MONITOR_PID=$!
cleanup_gpu_monitor() {
  kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup_gpu_monitor EXIT
PROBE_ARGS=()
if [[ -n "${PROBE_QUERY_PLAN:-}" ]]; then
  PROBE_ARGS+=(--query-plan-jsonl "${PROBE_QUERY_PLAN}")
fi
python3 -m runs.counter.diagnose_teacher \
  --teacher-model "${TEACHER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}" \
  --output-dir "${PROBE_OUT}" \
  --num-queries "${PROBE_NUM_QUERIES:-3}" \
  --max-tokens "${PROBE_MAX_TOKENS:-256}" \
  "${PROBE_ARGS[@]}" 2>&1 | tee "${PROBE_OUT}/generation.log"
