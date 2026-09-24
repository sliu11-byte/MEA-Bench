#!/usr/bin/env bash

set -euo pipefail
SUBMIT_DIR="${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
REPO_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
while [[ "${REPO_DIR}" != / && ! -f "${REPO_DIR}/evaluation/attack_eval/evaluate_attack_four_metrics.py" ]]; do
  REPO_DIR="$(dirname "${REPO_DIR}")"
done
[[ -f "${REPO_DIR}/evaluation/attack_eval/evaluate_attack_four_metrics.py" ]] || {
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
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORAGE_ROOT}/cache}"
export TORCH_HOME="${TORCH_HOME:-${STORAGE_ROOT}/cache/torch}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS:-8}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

REUSE_ROOT="${ATTACK_B100_OLD_EVAL_ROOT:-${STORAGE_ROOT}/results/attack_eval/b100}"
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  OUTPUT="${ATTACK_B100_M1_V2_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b100_m1_v2_smoke_${EVAL_LIMIT}}"
else
  OUTPUT="${ATTACK_B100_M1_V2_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b100_m1_v2}"
fi
ATTACK_ROOT="${ATTACK_ROOT:-${STORAGE_ROOT}/inputs/attack_runs/b100}"
[[ -d "${ATTACK_ROOT}" ]] || {
  echo "Missing local B=100 attack results: ${ATTACK_ROOT}" >&2
  echo 'Set ATTACK_ROOT to the directory containing the completed attack runs.' >&2
  exit 2
}
ARGS=(--attack-root "${ATTACK_ROOT}" --output-root "${OUTPUT}" --m1-only --reuse-gsm8k-root "${REUSE_ROOT}" --exclude-attack soda
      --expected-teacher-model meta-llama/Llama-3.3-70B-Instruct
      --expected-student-model meta-llama/Llama-3.1-8B-Instruct)
[[ -n "${EVAL_LIMIT:-}" ]] && ARGS+=(--limit "${EVAL_LIMIT}")

python -c 'import torch, transformers, peft, datasets, yaml, huggingface_hub'
echo "Recomputing M1 v2 for SeqKD, QEDKS, Model Leeching, LoRD, and GAD."
echo "Reusing GSM8K from: ${REUSE_ROOT}"
echo "Output: ${OUTPUT}"
python -m evaluation.attack_eval.evaluate_attack_four_metrics "${ARGS[@]}"
