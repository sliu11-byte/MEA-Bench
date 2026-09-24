#!/usr/bin/env bash

set -euo pipefail

SUBMIT_DIR="${REPO_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey"
else
  echo "Submit from the repository root." >&2
  exit 2
fi
cd "${REPO_DIR}"
mkdir -p logs

: "${ATTACK:?Set ATTACK to qedks, model_leeching, or soda}"
case "${ATTACK}" in
  qedks|model_leeching|soda) ;;
  *) echo "Unsupported staged training attack: ${ATTACK}" >&2; exit 2 ;;
esac

STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
BUDGET_VALUE="${BUDGET:-1000}"
case "${ATTACK}" in
  qedks) BUDGET_VALUE="${QEDKS_BUDGET:-${BUDGET_VALUE}}" ;;
  model_leeching) BUDGET_VALUE="${MODEL_LEECHING_BUDGET:-${BUDGET_VALUE}}" ;;
  soda) BUDGET_VALUE="${SODA_BUDGET:-${BUDGET_VALUE}}" ;;
esac
POINTER_PATH="${STAGED_RUN_POINTER:-${STORAGE_ROOT}/outputs/staged_attacks/${ATTACK}_b${BUDGET_VALUE}_latest.txt}"
if [[ ! -s "${POINTER_PATH}" ]]; then
  echo "Prepared-run pointer is missing: ${POINTER_PATH}" >&2
  exit 1
fi

export PREPARED_RUN_DIR="$(head -n 1 "${POINTER_PATH}")"
export ATTACKS="${ATTACK}"
export ATTACK_INDEX=0
export ATTACK_EXECUTION_STAGE=train
export ATTACK_CUDA_VISIBLE_DEVICES="${ATTACK_CUDA_VISIBLE_DEVICES:-0}"
export ONLINE_ATTACK_TEACHER_MODE=external
export SODA_STUDENT_MODE=external

echo "Training prepared ${ATTACK} run: ${PREPARED_RUN_DIR}"
bash attacks/scripts/run_attacks.sh
