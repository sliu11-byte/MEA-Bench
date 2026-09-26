#!/usr/bin/env bash

set -euo pipefail
SUBMIT_DIR="${REPO_BASE_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/evaluation/attack_eval/evaluate_attack_four_metrics.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey/evaluation/attack_eval/evaluate_attack_four_metrics.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey"
else
  echo "Could not locate repository from submit dir: ${SUBMIT_DIR}" >&2
  echo 'Submit from the repository root.' >&2
  exit 2
fi
cd "${REPO_DIR}"
mkdir -p logs
export STORAGE_ROOT="${STORAGE_ROOT:-${REPO_DIR}}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORAGE_ROOT}/cache}"
export TORCH_HOME="${TORCH_HOME:-${STORAGE_ROOT}/cache/torch}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS:-8}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
PROMPTS="${HELDOUT_PROMPTS_JSONL:-${REPO_DIR}/evaluation/data/heldout/heldout_prompts.jsonl}"
TEACHER="${HELDOUT_TEACHER_OUTPUTS_JSONL:-${STORAGE_ROOT}/outputs/heldout_queries/heldout_teacher_outputs.jsonl}"
OUTPUT="${ATTACK_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b100}"
[[ -s "$PROMPTS" && -s "$TEACHER" ]] || { echo 'Missing held-out prompts or teacher outputs.' >&2; exit 2; }
if ! python -c 'import torch, transformers, peft, huggingface_hub, bert_score, datasets, yaml'; then
  echo "Evaluation dependency check failed in ${CONDA_ENV:-research}." >&2
  echo 'For missing bert_score, run in that environment: python -m pip install bert-score==0.3.13' >&2
  exit 2
fi
mkdir -p "$OUTPUT" "$HF_HOME" "$TORCH_HOME"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${STORAGE_ROOT}/cache/huggingface/datasets}"
ATTACK_ROOT="${ATTACK_ROOT:-${STORAGE_ROOT}/inputs/attack_runs/b100}"
[[ -d "${ATTACK_ROOT}" ]] || {
  echo "Missing local B=100 attack results: ${ATTACK_ROOT}" >&2
  echo 'Set ATTACK_ROOT to the directory containing the completed attack runs.' >&2
  exit 2
}
ARGS=(--attack-root "$ATTACK_ROOT" --output-root "$OUTPUT" --eval-prompts-jsonl "$PROMPTS" --teacher-jsonl "$TEACHER"
      --expected-teacher-model meta-llama/Llama-3.3-70B-Instruct
      --expected-student-model meta-llama/Llama-3.1-8B-Instruct)
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  ARGS+=(--limit "$EVAL_LIMIT")
  [[ -n "${ATTACK_EVAL_OUTPUT_ROOT:-}" ]] || ARGS+=(--output-root "${OUTPUT}_smoke_${EVAL_LIMIT}")
fi
if [[ -n "${EVAL_ATTACK:-}" ]]; then
  ARGS+=(--attack "$EVAL_ATTACK")
fi
echo "Output: $OUTPUT"
echo 'Metrics: six-task macro ACC, BERTScore, Rep-4, historical allocated GPU-hours.'
python -m evaluation.attack_eval.evaluate_attack_four_metrics "${ARGS[@]}"
