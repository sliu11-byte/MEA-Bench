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

export STORAGE_ROOT="${STORAGE_ROOT:-${REPO_DIR}}"
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
TEACHER_MANIFEST="${HELDOUT_TEACHER_MANIFEST:-${TEACHER%.jsonl}.manifest.json}"
[[ -s "${PROMPTS}" && -s "${TEACHER}" && -s "${TEACHER_MANIFEST}" ]] || {
  echo "Missing Llama held-out prompts, teacher outputs, or teacher manifest." >&2
  exit 2
}
python - "${TEACHER_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = "meta-llama/Llama-3.3-70B-Instruct"
if payload.get("teacher_model") != expected or payload.get("prompt_count") != payload.get("completed_count"):
    raise SystemExit(f"Invalid or incomplete Llama teacher reference: {sys.argv[1]}")
PY

OUTPUT="${ATTACK_B10000_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b10000}"
ROOT="${ATTACK_B10000_ROOT:-${STORAGE_ROOT}/inputs/attack_runs/b10000}"
[[ -d "${ROOT}" ]] || {
  echo "Missing local B=10000 attack results: ${ROOT}" >&2
  echo 'Set ATTACK_B10000_ROOT to the directory containing the completed attack runs.' >&2
  exit 2
}
ARGS=(--budget 10000 --output-root "${OUTPUT}" --attack-root "${ROOT}"
      --attack "${EVAL_ATTACK:?Set EVAL_ATTACK to one completed method}"
      --eval-prompts-jsonl "${PROMPTS}" --teacher-jsonl "${TEACHER}"
      --expected-teacher-model meta-llama/Llama-3.3-70B-Instruct
      --expected-student-model meta-llama/Llama-3.1-8B-Instruct
      --skip-m6)
[[ -n "${EVAL_LIMIT:-}" ]] && ARGS+=(--limit "${EVAL_LIMIT}")

python -c 'import torch, transformers, peft, datasets, yaml, huggingface_hub, bert_score'
echo "Attack B=10000 method: ${EVAL_ATTACK}"
echo "Output: ${OUTPUT}"
python -m evaluation.attack_eval.evaluate_attack_four_metrics "${ARGS[@]}"
