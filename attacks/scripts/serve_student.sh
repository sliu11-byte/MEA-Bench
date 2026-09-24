#!/usr/bin/env bash

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module purge || true
  command -v module >/dev/null 2>&1 && module load conda/25.7.0 cuda/12.8.1 || true || true
fi
if command -v conda >/dev/null 2>&1; then conda activate "${CONDA_ENV:-research}"; fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi

export OMP_NUM_THREADS="${CPU_THREADS:-4}"
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
elif [[ -f "${SUBMIT_DIR}/serve_student.sh" && "$(basename "${SUBMIT_DIR}")" == "scripts" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey"
else
  echo "Could not locate A-Benchmark-for-Model-distillation-survey from submit dir: ${SUBMIT_DIR}" >&2
  exit 1
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
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}" "${STORAGE_ROOT}/outputs/vllm_student"

STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL:-${STUDENT_MODEL}}"
VLLM_MODEL="${STUDENT_MODEL}"
LORA_ARGS=()
if [[ -n "${WARMUP_MODEL:-}" && -f "${WARMUP_MODEL}/adapter_config.json" ]]; then
  VLLM_MODEL="$(python3 - "${WARMUP_MODEL}/adapter_config.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["base_model_name_or_path"])
PY
)"
  STUDENT_REQUEST_MODEL="${SODA_WARMUP_REQUEST_MODEL:-soda-warmup}"
  LORA_ARGS=(--enable-lora --lora-modules "${STUDENT_REQUEST_MODEL}=${WARMUP_MODEL}")
elif [[ -n "${WARMUP_MODEL:-}" ]]; then
  VLLM_MODEL="${WARMUP_MODEL}"
  STUDENT_REQUEST_MODEL="${SODA_WARMUP_REQUEST_MODEL:-soda-warmup}"
fi
STUDENT_VLLM_HOST="${STUDENT_VLLM_HOST:-0.0.0.0}"
STUDENT_VLLM_PORT="${STUDENT_VLLM_PORT:-}"
if [[ -z "${STUDENT_VLLM_PORT}" ]]; then
  STUDENT_VLLM_PORT="$(find_free_port)"
fi
STUDENT_VLLM_TENSOR_PARALLEL_SIZE="${STUDENT_VLLM_TENSOR_PARALLEL_SIZE:-1}"
STUDENT_VLLM_GPU_MEMORY_UTILIZATION="${STUDENT_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
STUDENT_VLLM_MAX_MODEL_LEN="${STUDENT_VLLM_MAX_MODEL_LEN:-4096}"
STUDENT_VLLM_DTYPE="${STUDENT_VLLM_DTYPE:-bfloat16}"
STUDENT_VLLM_API_KEY="${STUDENT_VLLM_API_KEY:-${STUDENT_API_KEY:-EMPTY}}"
ENDPOINT_HOST="${STUDENT_VLLM_ADVERTISE_HOST:-$(hostname -f 2>/dev/null || hostname)}"
ENDPOINT_URL="http://${ENDPOINT_HOST}:${STUDENT_VLLM_PORT}/v1"
ENDPOINT_ENV_PATH="${STUDENT_VLLM_ENDPOINT_ENV_PATH:-${STORAGE_ROOT}/outputs/vllm_student/student_endpoint.env}"
mkdir -p "$(dirname "${ENDPOINT_ENV_PATH}")"

ENDPOINT_ENV_TMP="${ENDPOINT_ENV_PATH}.tmp.${RUN_ID:-$$}"
cat > "${ENDPOINT_ENV_TMP}" <<EOF
export STUDENT_MODEL="${STUDENT_MODEL}"
export WARMUP_MODEL="${WARMUP_MODEL:-}"
export STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL}"
export STUDENT_ENDPOINT_URL="${ENDPOINT_URL}"
export STUDENT_API_KEY="${STUDENT_VLLM_API_KEY}"
export STUDENT_VLLM_JOB_ID="${RUN_ID:-}"
EOF
mv -f "${ENDPOINT_ENV_TMP}" "${ENDPOINT_ENV_PATH}"

cat <<EOF
vLLM student server configuration:
  REPO_DIR=${REPO_DIR}
  STORAGE_ROOT=${STORAGE_ROOT}
  MODEL=${VLLM_MODEL}
  WARMUP_MODEL=${WARMUP_MODEL:-}
  SERVED_MODEL_NAME=${STUDENT_REQUEST_MODEL}
  HOST=${STUDENT_VLLM_HOST}
  PORT=${STUDENT_VLLM_PORT}
  ENDPOINT_URL=${ENDPOINT_URL}
  ENDPOINT_ENV_PATH=${ENDPOINT_ENV_PATH}
  TENSOR_PARALLEL_SIZE=${STUDENT_VLLM_TENSOR_PARALLEL_SIZE}
  GPU_MEMORY_UTILIZATION=${STUDENT_VLLM_GPU_MEMORY_UTILIZATION}
  MAX_MODEL_LEN=${STUDENT_VLLM_MAX_MODEL_LEN}
  DTYPE=${STUDENT_VLLM_DTYPE}

After this log says the server is ready, run:
  source ${ENDPOINT_ENV_PATH}
  bash attacks/scripts/run_attacks.sh
EOF

env -u VLLM_PORT python3 -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL}" \
  --served-model-name "${STUDENT_REQUEST_MODEL}" \
  --host "${STUDENT_VLLM_HOST}" \
  --port "${STUDENT_VLLM_PORT}" \
  --api-key "${STUDENT_VLLM_API_KEY}" \
  --tensor-parallel-size "${STUDENT_VLLM_TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${STUDENT_VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${STUDENT_VLLM_MAX_MODEL_LEN}" \
  --dtype "${STUDENT_VLLM_DTYPE}" \
  "${LORA_ARGS[@]}" \
  ${STUDENT_VLLM_EXTRA_ARGS:-}
