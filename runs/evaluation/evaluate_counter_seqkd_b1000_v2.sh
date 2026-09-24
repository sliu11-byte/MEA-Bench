#!/usr/bin/env bash

set -euo pipefail
SUBMIT_DIR="${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
REPO_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
while [[ "${REPO_DIR}" != / && ! -f "${REPO_DIR}/evaluation/defense_eval/evaluate_counter_seqkd.py" ]]; do
  REPO_DIR="$(dirname "${REPO_DIR}")"
done
[[ -f "${REPO_DIR}/evaluation/defense_eval/evaluate_counter_seqkd.py" ]] || {
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
if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi

PROMPTS="${HELDOUT_PROMPTS_JSONL:-${STORAGE_ROOT}/outputs/heldout_queries/heldout_prompts.jsonl}"
TEACHER="${HELDOUT_TEACHER_OUTPUTS_JSONL:-${STORAGE_ROOT}/outputs/heldout_queries/heldout_teacher_outputs.jsonl}"
REUSE_ROOT="${COUNTER_OLD_EVAL_ROOT:-${STORAGE_ROOT}/results/counter_eval/seqkd_b1000}"
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  OUTPUT="${COUNTER_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/counter_eval/seqkd_b1000_m1_v2_smoke_${EVAL_LIMIT}}"
else
  OUTPUT="${COUNTER_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/counter_eval/seqkd_b1000_m1_v2}"
fi
[[ -d "${REUSE_ROOT}/models" ]] || { echo "Old Counter output is missing: ${REUSE_ROOT}" >&2; exit 2; }
ARGS=(--storage-root "${STORAGE_ROOT}" --output-root "${OUTPUT}"
      --prompts-jsonl "${PROMPTS}" --teacher-jsonl "${TEACHER}"
      --reuse-generation-root "${REUSE_ROOT}")
[[ -n "${HELDOUT_TEACHER_MANIFEST:-}" ]] && ARGS+=(--teacher-manifest "${HELDOUT_TEACHER_MANIFEST}")
[[ -n "${EVAL_LIMIT:-}" ]] && ARGS+=(--limit "${EVAL_LIMIT}")
[[ "${EVAL_PREFLIGHT_ONLY:-0}" == 1 ]] && ARGS+=(--preflight)

python -c 'import torch, transformers, accelerate, peft, datasets, yaml, bert_score'
echo "Reusing valid generations from: ${REUSE_ROOT}"
echo "M1 v2 output: ${OUTPUT}"
python -m evaluation.defense_eval.evaluate_counter_seqkd "${ARGS[@]}"
