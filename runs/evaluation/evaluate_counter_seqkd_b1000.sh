#!/usr/bin/env bash

set -euo pipefail
cd "${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
[[ -f evaluation/defense_eval/evaluate_counter_seqkd.py ]] || { echo 'Submit from repository root.' >&2; exit 2; }
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
if ! python -c 'import torch, transformers, accelerate, peft, datasets, yaml, bert_score'; then
  echo 'Check research dependencies; missing BERTScore: python -m pip install bert-score==0.3.13' >&2
  exit 2
fi
PROMPTS="${HELDOUT_PROMPTS_JSONL:-${STORAGE_ROOT}/outputs/heldout_queries/heldout_prompts.jsonl}"
TEACHER="${HELDOUT_TEACHER_OUTPUTS_JSONL:-${STORAGE_ROOT}/outputs/heldout_queries/heldout_teacher_outputs.jsonl}"
OUTPUT="${COUNTER_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/counter_eval/seqkd_b1000}"
ARGS=(--storage-root "$STORAGE_ROOT" --output-root "$OUTPUT" --prompts-jsonl "$PROMPTS" --teacher-jsonl "$TEACHER")
if [[ -n "${HELDOUT_TEACHER_MANIFEST:-}" ]]; then
  ARGS+=(--teacher-manifest "$HELDOUT_TEACHER_MANIFEST")
fi
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  ARGS+=(--limit "$EVAL_LIMIT")
  [[ -n "${COUNTER_EVAL_OUTPUT_ROOT:-}" ]] || ARGS+=(--output-root "${OUTPUT}_smoke_${EVAL_LIMIT}")
fi
[[ "${EVAL_PREFLIGHT_ONLY:-0}" == 1 ]] && ARGS+=(--preflight)
python -m evaluation.defense_eval.evaluate_counter_seqkd "${ARGS[@]}"
