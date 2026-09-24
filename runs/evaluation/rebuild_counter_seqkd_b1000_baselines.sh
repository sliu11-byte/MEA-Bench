#!/usr/bin/env bash

set -euo pipefail
SUBMIT_DIR="${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
REPO_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
while [[ "${REPO_DIR}" != / && ! -f "${REPO_DIR}/evaluation/defense_eval/rebuild_counter_detector_baselines.py" ]]; do
  REPO_DIR="$(dirname "${REPO_DIR}")"
done
[[ -f "${REPO_DIR}/evaluation/defense_eval/rebuild_counter_detector_baselines.py" ]] || {
  echo "Could not locate repository from ${SUBMIT_DIR}" >&2
  exit 2
}
cd "${REPO_DIR}"
mkdir -p logs

command -v module >/dev/null 2>&1 && module purge || true
command -v module >/dev/null 2>&1 && module load conda/25.7.0 cuda/12.8.1 || true
command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV:-research}" || true
export STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORAGE_ROOT}/cache}"
export TORCH_HOME="${TORCH_HOME:-${STORAGE_ROOT}/cache/torch}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS:-8}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi

ARGS=(--storage-root "${STORAGE_ROOT}"
      --detector-seed "${DETECTOR_SEED:-42}"
      --detector-max-queries "${DETECTOR_MAX_QUERIES:-1000}"
      --detector-max-new-tokens "${DETECTOR_MAX_NEW_TOKENS:-124}"
      --detector-temperature "${DETECTOR_TEMPERATURE:-0.7}")
[[ -n "${COUNTER_EVAL_OUTPUT_ROOT:-}" ]] && ARGS+=(--evaluation-root "${COUNTER_EVAL_OUTPUT_ROOT}")

python -m evaluation.defense_eval.rebuild_counter_detector_baselines "${ARGS[@]}"
