#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_DIR:-$(pwd)}"
[[ -f evaluation/defense_eval/evaluate.py ]] || { echo 'Submit from repository root.' >&2; exit 2; }
if command -v module >/dev/null 2>&1; then
  command -v module >/dev/null 2>&1 && module load conda/25.7.0 cuda/12.8.1 || true || true
fi
if command -v conda >/dev/null 2>&1; then
  command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV:-research}" || true
fi
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
if ! python -c 'import torch, transformers, accelerate, peft, datasets, yaml, huggingface_hub, bert_score'; then
  echo 'Check research dependencies; for bert_score: python -m pip install bert-score==0.3.13' >&2
  exit 2
fi
PROMPTS="${HELDOUT_PROMPTS_JSONL:-$(pwd)/evaluation/data/heldout/heldout_prompts.jsonl}"
OUTPUT="${DEFENSE_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/defense_eval/b1000}"
CHECKPOINT_ROOT="${DEFENSE_CHECKPOINT_ROOT:-${STORAGE_ROOT}/inputs/defense_checkpoints}"
ARGS=(--output-root "$OUTPUT" --prompts-jsonl "$PROMPTS" --checkpoint-root "$CHECKPOINT_ROOT")
if [[ "${EVAL_M1_ONLY:-0}" == 1 ]]; then
  ARGS+=(--m1-only)
  [[ -n "${REUSE_GSM8K_ROOT:-}" ]] && ARGS+=(--reuse-gsm8k-root "$REUSE_GSM8K_ROOT")
  [[ -n "${REUSE_SODA_GSM8K_ROOT:-}" ]] && ARGS+=(--reuse-soda-gsm8k-root "$REUSE_SODA_GSM8K_ROOT")
fi
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  ARGS+=(--limit "$EVAL_LIMIT")
  [[ -n "${DEFENSE_EVAL_OUTPUT_ROOT:-}" ]] || ARGS+=(--output-root "${OUTPUT}_smoke_${EVAL_LIMIT}")
fi
[[ -s "$PROMPTS" ]] || { echo "Missing prompts: $PROMPTS" >&2; exit 2; }
