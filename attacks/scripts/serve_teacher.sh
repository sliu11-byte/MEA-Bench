#!/usr/bin/env bash

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module purge || true
  command -v module >/dev/null 2>&1 && module load conda/25.7.0 cuda/12.8.1 || true || true
fi
if command -v conda >/dev/null 2>&1; then conda activate "${CONDA_ENV:-research}"; fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi

export OMP_NUM_THREADS="${CPU_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

find_free_port() {
  python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
}


SUBMIT_DIR="${REPO_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/serve_teacher.sh" && "$(basename "${SUBMIT_DIR}")" == "scripts" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey"
else
  echo "Could not locate A-Benchmark-for-Model-distillation-survey from submit dir: ${SUBMIT_DIR}" >&2
  exit 2
fi
cd "${REPO_DIR}"

DEFAULT_STORAGE_ROOT="/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey"
export STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"

FULL_RUN_ENV="${FULL_RUN_ENV:-${REPO_DIR}/attacks/configs/full_run_hpg.env}"
if [[ -f "${FULL_RUN_ENV}" ]]; then
  set -a
  source "${FULL_RUN_ENV}"
  set +a
fi

export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}" "${STORAGE_ROOT}/outputs/vllm_teacher"

TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL}}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-}"
if [[ -z "${VLLM_PORT}" ]]; then
  VLLM_PORT="$(find_free_port)"
fi
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8096}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
ENDPOINT_HOST="${VLLM_ADVERTISE_HOST:-$(hostname -f 2>/dev/null || hostname)}"
ENDPOINT_URL="http://${ENDPOINT_HOST}:${VLLM_PORT}/v1"
ENDPOINT_ENV_PATH="${VLLM_ENDPOINT_ENV_PATH:-${STORAGE_ROOT}/outputs/vllm_teacher/teacher_endpoint.env}"
mkdir -p "$(dirname "${ENDPOINT_ENV_PATH}")"

ENDPOINT_ENV_TMP="${ENDPOINT_ENV_PATH}.tmp.${RUN_ID:-$$}"
cat > "${ENDPOINT_ENV_TMP}" <<EOF
export TEACHER_BACKEND="vllm_openai"
export TEACHER_MODEL="${TEACHER_MODEL}"
export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL}"
export TEACHER_ENDPOINT_URL="${ENDPOINT_URL}"
export TEACHER_API_KEY="${VLLM_API_KEY}"
export TEACHER_VLLM_JOB_ID="${RUN_ID:-}"
export STAGE2_TEACHER_BACKEND="vllm_openai"
export STAGE2_TEACHER_MODEL_PATH="${TEACHER_MODEL}"
export STAGE2_TEACHER_MODEL_NAME="${TEACHER_REQUEST_MODEL}"
export STAGE2_TEACHER_BASE_URL="${ENDPOINT_URL}"
export STAGE2_TEACHER_API_KEY="${VLLM_API_KEY}"
EOF
mv -f "${ENDPOINT_ENV_TMP}" "${ENDPOINT_ENV_PATH}"

cat <<EOF
vLLM teacher server configuration:
  REPO_DIR=${REPO_DIR}
  STORAGE_ROOT=${STORAGE_ROOT}
  MODEL=${TEACHER_MODEL}
  SERVED_MODEL_NAME=${TEACHER_REQUEST_MODEL}
  HOST=${VLLM_HOST}
  PORT=${VLLM_PORT}
  ENDPOINT_URL=${ENDPOINT_URL}
  ENDPOINT_ENV_PATH=${ENDPOINT_ENV_PATH}
  TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}
  GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}
  MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}
  DTYPE=${VLLM_DTYPE}

After this log says the server is ready, run:
  source ${ENDPOINT_ENV_PATH}
  bash attacks/scripts/run_shared_transcript.sh
EOF

env -u VLLM_PORT python3 -m vllm.entrypoints.openai.api_server \
  --model "${TEACHER_MODEL}" \
  --served-model-name "${TEACHER_REQUEST_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --api-key "${VLLM_API_KEY}" \
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --dtype "${VLLM_DTYPE}" \
  ${VLLM_EXTRA_ARGS:-}
