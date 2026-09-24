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

RUN_POINTER="${SODA_ATTACK_RUN_POINTER:-${STORAGE_ROOT}/outputs/staged_attacks/soda_b100_latest.txt}"
if [[ -z "${SODA_ATTACK_RUN_DIR:-}" ]]; then
  [[ -s "${RUN_POINTER}" ]] || { echo "Missing SODA run pointer: ${RUN_POINTER}" >&2; exit 2; }
  SODA_ATTACK_RUN_DIR="$(head -n 1 "${RUN_POINTER}")"
fi
[[ -s "${SODA_ATTACK_RUN_DIR}/attack_manifest.json" ]] || {
  echo "SODA training has not produced a completed attack manifest: ${SODA_ATTACK_RUN_DIR}" >&2
  exit 2
}

JOB_RECORD="${SODA_ATTACK_JOB_RECORD:-${STORAGE_ROOT}/outputs/job_ids/staged_soda_b100_latest.env}"
if [[ -z "${SODA_ATTACK_JOB_IDS:-}" && -s "${JOB_RECORD}" ]]; then
  set -a
  source "${JOB_RECORD}"
  set +a
  SODA_ATTACK_JOB_IDS="${STAGED_ATTACK_ENDPOINT_JOB_ID:-} ${STAGED_ATTACK_TRAIN_JOB_ID:-}"
fi
read -r -a COST_JOB_IDS <<< "${SODA_ATTACK_JOB_IDS:-}"
(( ${#COST_JOB_IDS[@]} > 0 )) || {
  echo 'Set SODA_ATTACK_JOB_IDS to the allocation job IDs used by this SODA run.' >&2
  echo 'They appear as "Endpoint job:" and "Training job:" in the staged attack log.' >&2
  exit 2
}
if (( ${#COST_JOB_IDS[@]} < 2 )) && [[ "${ALLOW_SINGLE_SODA_COST_JOB:-0}" != 1 ]]; then
  echo "Expected both endpoint and training job IDs, got: ${COST_JOB_IDS[*]}" >&2
  echo 'For a deliberate RESUME_PREPARED run, set ALLOW_SINGLE_SODA_COST_JOB=1.' >&2
  exit 2
fi

PROMPTS="${HELDOUT_PROMPTS_JSONL:-${REPO_DIR}/evaluation/data/heldout/heldout_prompts.jsonl}"
TEACHER="${HELDOUT_TEACHER_OUTPUTS_JSONL:-${STORAGE_ROOT}/outputs/heldout_queries/heldout_teacher_outputs.jsonl}"
[[ -s "${PROMPTS}" && -s "${TEACHER}" ]] || { echo 'Missing held-out prompts or teacher outputs.' >&2; exit 2; }
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  OUTPUT="${SODA_ATTACK_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b100_soda_corrected_smoke_${EVAL_LIMIT}}"
else
  OUTPUT="${SODA_ATTACK_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b100_soda_corrected}"
fi
ARGS=(--attack-root "${SODA_ATTACK_RUN_DIR}" --attack soda --output-root "${OUTPUT}"
      --eval-prompts-jsonl "${PROMPTS}" --teacher-jsonl "${TEACHER}"
      --expected-teacher-model meta-llama/Llama-3.3-70B-Instruct
      --expected-student-model meta-llama/Llama-3.1-8B-Instruct
      --allocation-job-ids "${COST_JOB_IDS[@]}")
[[ -n "${EVAL_LIMIT:-}" ]] && ARGS+=(--limit "${EVAL_LIMIT}")

python -c 'import torch, transformers, peft, datasets, yaml, bert_score'
echo "SODA run: ${SODA_ATTACK_RUN_DIR}"
echo "M6 allocation jobs: ${COST_JOB_IDS[*]}"
echo "Output: ${OUTPUT}"
python -m evaluation.attack_eval.evaluate_attack_four_metrics "${ARGS[@]}"
