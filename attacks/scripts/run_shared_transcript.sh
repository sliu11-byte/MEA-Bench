#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi

export OMP_NUM_THREADS="${CPU_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

SUBMIT_DIR="${REPO_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/build_shared_teacher_transcript.py" && "$(basename "${SUBMIT_DIR}")" == "scripts" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey"
else
  echo "Could not locate A-Benchmark-for-Model-distillation-survey from submit dir: ${SUBMIT_DIR}" >&2
  exit 1
fi
cd "${REPO_DIR}"

DEFAULT_STORAGE_ROOT="${REPO_DIR}"
export STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"

FULL_RUN_ENV="${FULL_RUN_ENV:-}"
if [[ -n "${FULL_RUN_ENV}" && -f "${FULL_RUN_ENV}" ]]; then
  set -a
  source "${FULL_RUN_ENV}"
  set +a
fi

export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}" "${SHARED_TRANSCRIPT_ROOT:-${STORAGE_ROOT}/outputs/shared_teacher_transcripts}"

python3 attacks/scripts/check_attack_env.py --require-trl --strict-versions

TRANSCRIPT_BUDGET="${TRANSCRIPT_BUDGET:-${BUDGET:-1000}}"
SHARED_TRANSCRIPT_ROOT="${SHARED_TRANSCRIPT_ROOT:-${STORAGE_ROOT}/outputs/shared_teacher_transcripts}"
LATEST_PATH="${LATEST_SHARED_TRANSCRIPT_PATH:-${SHARED_TRANSCRIPT_ROOT}/latest_bundle_dir.txt}"

ARGS=(
  --budget "${TRANSCRIPT_BUDGET}"
  --query-pool "${QUERY_POOL:-auto}"
  --query-ordering "${QUERY_ORDERING:-auto}"
  --stage1-config "${STAGE1_CONFIG:-attacks/configs/formal_stage1_budget.yaml}"
  --output-dir "${SHARED_TRANSCRIPT_ROOT}"
  --latest-path "${LATEST_PATH}"
  --teacher-backend "${TEACHER_BACKEND:-vllm_openai}"
  --teacher-model "${TEACHER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
  --teacher-endpoint-url "${TEACHER_ENDPOINT_URL:-http://127.0.0.1:8000/v1}"
  --teacher-request-model "${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}}"
  --teacher-api-key "${TEACHER_API_KEY:-EMPTY}"
  --teacher-max-new-tokens "${STAGE1_TEACHER_MAX_NEW_TOKENS:-1536}"
  --student-model "${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
  --seed "${SEED:-20260701}"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  ARGS+=(--validate-only)
fi

echo "Shared teacher transcript build:"
echo "  REPO_DIR=${REPO_DIR}"
echo "  STORAGE_ROOT=${STORAGE_ROOT}"
echo "  SHARED_TRANSCRIPT_ROOT=${SHARED_TRANSCRIPT_ROOT}"
echo "  LATEST_PATH=${LATEST_PATH}"
echo "  BUDGET=${TRANSCRIPT_BUDGET}"
echo "  QUERY_POOL=${QUERY_POOL:-auto}"
echo "  STAGE1_CONFIG=${STAGE1_CONFIG:-attacks/configs/formal_stage1_budget.yaml}"
echo "  TEACHER_ENDPOINT_URL=${TEACHER_ENDPOINT_URL:-http://127.0.0.1:8000/v1}"

python3 attacks/scripts/build_shared_teacher_transcript.py "${ARGS[@]}"
