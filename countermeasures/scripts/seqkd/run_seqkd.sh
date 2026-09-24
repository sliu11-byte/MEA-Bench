#!/usr/bin/env bash

set -euo pipefail

SUBMIT_DIR="${REPO_BASE_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/countermeasures/scripts/seqkd/run_seqkd.sh" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/scripts/seqkd/run_seqkd.sh" && "$(basename "${SUBMIT_DIR}")" == "countermeasures" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/run_seqkd.sh" && "$(basename "${SUBMIT_DIR}")" == "seqkd" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey/countermeasures/scripts/seqkd/run_seqkd.sh" ]]; then
  REPO_DIR="${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey"
else
  echo "Could not locate A-Benchmark-for-Model-distillation-survey from submit dir: ${SUBMIT_DIR}" >&2
  exit 1
fi
cd "$REPO_DIR"

export COUNTERMEASURE="${COUNTERMEASURE:-dipper}"
case "$COUNTERMEASURE" in
  dipper|translation) ;;
  *) echo "Unknown COUNTERMEASURE=${COUNTERMEASURE}" >&2; exit 2 ;;
esac

export METHOD="${METHOD:-ginsew}"
case "$METHOD" in
  ginsew|radioactivity|adfp) ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac

export BUDGET="${BUDGET:-1000}"
DEFAULT_STORAGE_ROOT="${REPO_DIR}"
if [[ -d "/path/to/storage/${USER}" ]]; then
  DEFAULT_STORAGE_ROOT="/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey"
elif [[ -d "/home/${USER}/project/A-Benchmark-for-Model-distillation-survey" ]]; then
  DEFAULT_STORAGE_ROOT="/home/${USER}/project/A-Benchmark-for-Model-distillation-survey"
fi
export STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/countermeasures/seqkd_b${BUDGET}/${COUNTERMEASURE}}"
export LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs/countermeasures/seqkd_b${BUDGET}/${COUNTERMEASURE}}"

cat <<EOF
SeqKD countermeasure formal configuration:
  REPO_DIR=${REPO_DIR}
  STORAGE_ROOT=${STORAGE_ROOT}
  METHOD=${METHOD}
  COUNTERMEASURE=${COUNTERMEASURE}
  COUNTERMEASURE_LEX=${COUNTERMEASURE_LEX:-40}
  COUNTERMEASURE_ORDER=${COUNTERMEASURE_ORDER:-0}
  BUDGET=${BUDGET}
  TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen2.5-72B-Instruct}
  STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen2.5-7B}
  QUERY_POOL=${QUERY_POOL:-auto}
  OUTPUT_ROOT=${OUTPUT_ROOT}
  LOG_ROOT=${LOG_ROOT}
EOF

bash "countermeasures/scripts/seqkd/run_${METHOD}_seqkd.sh"
