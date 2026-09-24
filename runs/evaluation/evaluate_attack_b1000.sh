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

PROMPTS="${HELDOUT_PROMPTS_JSONL:-${STORAGE_ROOT}/outputs/heldout_queries/heldout_prompts.jsonl}"
TEACHER="${HELDOUT_TEACHER_OUTPUTS_JSONL:-${STORAGE_ROOT}/outputs/heldout_queries/heldout_teacher_outputs.jsonl}"
TEACHER_MANIFEST="${HELDOUT_TEACHER_MANIFEST:-${TEACHER%.jsonl}.manifest.json}"
[[ -s "${PROMPTS}" && -s "${TEACHER}" ]] || {
  echo "Missing Llama held-out prompts or teacher outputs." >&2
  exit 2
}
python - "${TEACHER_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected = "meta-llama/Llama-3.3-70B-Instruct"
if payload.get("teacher_model") != expected or payload.get("prompt_count") != payload.get("completed_count"):
    raise SystemExit(f"Invalid or incomplete Llama teacher reference: {path}")
print(f"Validated Llama teacher reference: {path}")
PY
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  OUTPUT="${ATTACK_B1000_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b1000_smoke_${EVAL_LIMIT}}"
else
  OUTPUT="${ATTACK_B1000_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b1000}"
fi
ATTACK_B1000_ROOT="${ATTACK_B1000_ROOT:-${STORAGE_ROOT}/inputs/attack_runs/b1000}"
[[ -d "${ATTACK_B1000_ROOT}" ]] || {
  echo "Missing local B=1000 attack results: ${ATTACK_B1000_ROOT}" >&2
  echo 'Set ATTACK_B1000_ROOT to the directory containing the completed attack runs.' >&2
  exit 2
}
ARGS=(--budget 1000 --attack-root "${ATTACK_B1000_ROOT}" --output-root "${OUTPUT}"
      --eval-prompts-jsonl "${PROMPTS}" --teacher-jsonl "${TEACHER}"
      --expected-teacher-model meta-llama/Llama-3.3-70B-Instruct
      --expected-student-model meta-llama/Llama-3.1-8B-Instruct)
if [[ -n "${ATTACK_B1000_COST_RECORDS:-}" ]]; then
  ARGS+=(--cost-records "${ATTACK_B1000_COST_RECORDS}")
else
  ARGS+=(--skip-m6)
fi
[[ -n "${EVAL_LIMIT:-}" ]] && ARGS+=(--limit "${EVAL_LIMIT}")
[[ -n "${EVAL_ATTACK:-}" ]] && ARGS+=(--attack "${EVAL_ATTACK}")

python -c 'import torch, transformers, peft, datasets, yaml, huggingface_hub, bert_score'
echo "Attack B=1000 output: ${OUTPUT}"
if [[ -z "${ATTACK_B1000_COST_RECORDS:-}" ]]; then
  echo "M6 pending: set ATTACK_B1000_COST_RECORDS when allocation records are available."
fi
python -m evaluation.attack_eval.evaluate_attack_four_metrics "${ARGS[@]}"
