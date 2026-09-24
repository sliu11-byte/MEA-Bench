#!/usr/bin/env bash

set -euo pipefail

command -v module >/dev/null 2>&1 && module purge || true
command -v module >/dev/null 2>&1 && module load conda/25.7.0 || true
command -v module >/dev/null 2>&1 && module load cuda/12.8.1 || true

# Override when submitting if needed, e.g.:
#   CONDA_ENV=research STUDENT_MODEL=meta-llama/Llama-3.2-1B-Instruct bash attacks/scripts/smoke_seqkd.sh
command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV:-research}" || true
if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi

export OMP_NUM_THREADS="${CPU_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

SUBMIT_DIR="${REPO_BASE_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/run_attack.py" && "$(basename "${SUBMIT_DIR}")" == "scripts" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey"
else
  echo "Could not locate A-Benchmark-for-Model-distillation-survey from submit dir: ${SUBMIT_DIR}" >&2
  echo "Submit from the repository root or from attacks/scripts." >&2
  exit 1
fi

cd "${REPO_DIR}"

# Keep Hugging Face and run outputs off home when Blue storage is available.
DEFAULT_STORAGE_ROOT="/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey"
export STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

# This is a real end-to-end smoke run, not a dry-run. The Stage-1 implementation
# currently accepts budgets 100/1000/10000, so 100 is the smallest valid run.
export ATTACK="${ATTACK:-seqkd}"
export BUDGET="${BUDGET:-100}"
export OUTPUT_DIR="${OUTPUT_DIR:-${STORAGE_ROOT}/outputs/attacks_smoke}"
export QUERY_POOL="${QUERY_POOL:-auto}"
export QUERY_ORDERING="${QUERY_ORDERING:-auto}"
export STAGE1_CONFIG="${STAGE1_CONFIG:-attacks/configs/stage1_budget.yaml}"
export SEED="${SEED:-20260701}"

# Self-contained smoke default: local HF teacher + local HF student with a small model.
# For endpoint-backed teacher testing, override TEACHER_BACKEND=vllm_openai plus endpoint vars.
export TEACHER_BACKEND="${TEACHER_BACKEND:-local_hf}"
export TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
export TEACHER_ENDPOINT_URL="${TEACHER_ENDPOINT_URL:-http://127.0.0.1:8000/v1}"
export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL}}"
export TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"

export STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
export STUDENT_ENDPOINT_URL="${STUDENT_ENDPOINT_URL:-http://127.0.0.1:8001/v1}"
export STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL:-${STUDENT_MODEL}}"
export STUDENT_API_KEY="${STUDENT_API_KEY:-EMPTY}"

COMMON_ARGS=(
  --attack "${ATTACK}"
  --budget "${BUDGET}"
  --query-pool "${QUERY_POOL}"
  --query-ordering "${QUERY_ORDERING}"
  --stage1-config "${STAGE1_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --teacher-backend "${TEACHER_BACKEND}"
  --teacher-model "${TEACHER_MODEL}"
  --teacher-endpoint-url "${TEACHER_ENDPOINT_URL}"
  --teacher-request-model "${TEACHER_REQUEST_MODEL}"
  --teacher-api-key "${TEACHER_API_KEY}"
  --student-model "${STUDENT_MODEL}"
  --student-endpoint-url "${STUDENT_ENDPOINT_URL}"
  --student-request-model "${STUDENT_REQUEST_MODEL}"
  --student-api-key "${STUDENT_API_KEY}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

echo "Smoke run configuration:"
echo "  REPO_DIR=${REPO_DIR}"
echo "  STORAGE_ROOT=${STORAGE_ROOT}"
echo "  HF_HOME=${HF_HOME}"
echo "  ATTACK=${ATTACK}"
echo "  BUDGET=${BUDGET}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  QUERY_POOL=${QUERY_POOL}"
echo "  QUERY_ORDERING=${QUERY_ORDERING}"
echo "  TEACHER_BACKEND=${TEACHER_BACKEND}"
echo "  TEACHER_MODEL=${TEACHER_MODEL}"
echo "  TEACHER_ENDPOINT_URL=${TEACHER_ENDPOINT_URL}"
echo "  STUDENT_MODEL=${STUDENT_MODEL}"
echo "  DRY_RUN=${DRY_RUN:-0}"

python3 attacks/scripts/run_attack.py "${COMMON_ARGS[@]}"
