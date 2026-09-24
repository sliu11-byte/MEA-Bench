#!/usr/bin/env bash

set -euo pipefail
cd "${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
[[ -f evaluation/defense_eval/evaluate_duffin.py ]] || { echo 'Submit from repository root.' >&2; exit 2; }
command -v module >/dev/null 2>&1 && module load conda/25.7.0 cuda/12.8.1 || true
command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV:-research}" || true
export STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${STORAGE_ROOT}/cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${STORAGE_ROOT}/cache/torch}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${CPU_THREADS:-8}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi
python -c 'import torch, transformers, accelerate, peft, datasets, huggingface_hub'
export SODA_FULL_OUTPUT_ROOT="${SODA_FULL_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/defenses/soda_b1000_seqkd_warmup}"
[[ -s "${SODA_FULL_OUTPUT_ROOT}/clean/attack_manifest.json" ]] || {
  echo "Missing corrected clean SODA: ${SODA_FULL_OUTPUT_ROOT}/clean/attack_manifest.json" >&2
  exit 2
}
OUTPUT="${DUFFIN_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/duffin_eval/b1000_v2_corrected_soda}"
LIMIT="${DUFFIN_MAX_PROBES:-1000}"
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  LIMIT="$EVAL_LIMIT"
  [[ -n "${DUFFIN_EVAL_OUTPUT_ROOT:-}" ]] || OUTPUT="${OUTPUT}_smoke_${EVAL_LIMIT}"
fi
python -m evaluation.defense_eval.evaluate_duffin --output-root "$OUTPUT" \
  --max-probes "$LIMIT" --max-new-tokens "${DUFFIN_MAX_NEW_TOKENS:-1024}" \
  --min-valid-rate "${DUFFIN_MIN_VALID_RATE:-0.9}" \
  --batch-size "${DUFFIN_BATCH_SIZE:-4}" \
  --probe-seed "${DUFFIN_PROBE_SEED:-42}"
