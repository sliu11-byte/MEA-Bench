#!/usr/bin/env bash

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module purge || true
  command -v module >/dev/null 2>&1 && module load conda/25.7.0 cuda/12.8.1 || true || true
fi
if command -v conda >/dev/null 2>&1; then
  command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV:-research}" || true
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

export OMP_NUM_THREADS="${CPU_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

SUBMIT_DIR="${REPO_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/run_attack.py" && "$(basename "${SUBMIT_DIR}")" == "scripts" ]]; then
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
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

find_free_port() {
  python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
}

python3 attacks/scripts/check_attack_env.py --require-trl --strict-versions

ATTACK_LIST=(${ATTACKS:-seqkd lord soda qedks model_leeching gad})
TASK_ID="${ATTACK_INDEX:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#ATTACK_LIST[@]} )); then
  echo "ATTACK_INDEX=${TASK_ID} is outside ATTACKS length ${#ATTACK_LIST[@]}" >&2
  exit 1
fi
ATTACK="${ATTACK_LIST[$TASK_ID]}"

DEFAULT_BUDGET="${BUDGET:-1000}"
case "${ATTACK}" in
  seqkd) ATTACK_BUDGET="${SEQKD_BUDGET:-${DEFAULT_BUDGET}}" ;;
  lord) ATTACK_BUDGET="${LORD_BUDGET:-${DEFAULT_BUDGET}}" ;;
  soda) ATTACK_BUDGET="${SODA_BUDGET:-${DEFAULT_BUDGET}}" ;;
  qedks) ATTACK_BUDGET="${QEDKS_BUDGET:-${DEFAULT_BUDGET}}" ;;
  model_leeching) ATTACK_BUDGET="${MODEL_LEECHING_BUDGET:-${DEFAULT_BUDGET}}" ;;
  gad) ATTACK_BUDGET="${GAD_BUDGET:-${DEFAULT_BUDGET}}" ;;
  *) echo "Unknown attack: ${ATTACK}" >&2; exit 1 ;;
esac

export OUTPUT_DIR="${OUTPUT_DIR:-${STORAGE_ROOT}/outputs/attacks_full}"
export QUERY_POOL="${QUERY_POOL:-auto}"
export QUERY_ORDERING="${QUERY_ORDERING:-auto}"
export STAGE1_CONFIG="${STAGE1_CONFIG:-attacks/configs/formal_stage1_budget.yaml}"
export SEED="${SEED:-20260701}"

# Formal runs assume the teacher endpoint is already running. This avoids colocating
# a large teacher vLLM server with student training in the same GPU allocation.
export TEACHER_BACKEND="${TEACHER_BACKEND:-vllm_openai}"
export TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
export TEACHER_ENDPOINT_URL="${TEACHER_ENDPOINT_URL:-http://127.0.0.1:8000/v1}"
export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL}}"
export TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.0}"
export TEACHER_TOP_P="${TEACHER_TOP_P:-1.0}"
export TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-512}"
export TEACHER_ENDPOINT_ENV="${VLLM_ENDPOINT_ENV_PATH:-${STORAGE_ROOT}/outputs/vllm_teacher/teacher_endpoint.env}"
export WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-604800}"
export POLL_SECONDS="${POLL_SECONDS:-20}"
export AUTO_START_TEACHER_FOR_ONLINE_ATTACKS="${AUTO_START_TEACHER_FOR_ONLINE_ATTACKS:-1}"
export ONLINE_ATTACK_TEACHER_MODE="${ONLINE_ATTACK_TEACHER_MODE:-local}"
export AUTO_CANCEL_ATTACK_STARTED_TEACHER="${AUTO_CANCEL_ATTACK_STARTED_TEACHER:-0}"
export ONLINE_ATTACK_TEACHER_JOB_ID_FILE="${ONLINE_ATTACK_TEACHER_JOB_ID_FILE:-${STORAGE_ROOT}/outputs/job_ids/online_teacher_${ATTACK_BUDGET}.jobid}"
ATTACK_STARTED_TEACHER_JOB_ID=""
LOCAL_TEACHER_PID=""
LOCAL_STUDENT_PID=""
ATTACK_CUDA_VISIBLE_DEVICES="${ATTACK_CUDA_VISIBLE_DEVICES:-}"
export SODA_STUDENT_MODE="${SODA_STUDENT_MODE:-local}"
export ATTACK_EXECUTION_STAGE="${ATTACK_EXECUTION_STAGE:-all}"
export PREPARED_RUN_DIR="${PREPARED_RUN_DIR:-}"

check_openai_endpoint_once() {
  local url="$1"
  local api_key="$2"
  curl -fsS -H "Authorization: Bearer ${api_key}" "${url%/}/models" >/dev/null
}

wait_for_file() {
  local path="$1"
  local label="$2"
  local waited=0
  until [[ -s "${path}" ]]; do
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for ${label}: ${path}" >&2
      exit 1
    fi
    sleep "${POLL_SECONDS}"
    waited=$((waited + POLL_SECONDS))
    echo "Waiting for ${label} (${waited}s): ${path}"
  done
}

wait_for_endpoint_health() {
  local url="$1"
  local api_key="$2"
  local label="$3"
  local waited=0
  until check_openai_endpoint_once "${url}" "${api_key}" >/dev/null 2>&1; do
    if [[ -n "${LOCAL_TEACHER_PID:-}" ]] && ! kill -0 "${LOCAL_TEACHER_PID}" >/dev/null 2>&1; then
      echo "${label} endpoint process exited before becoming healthy: pid ${LOCAL_TEACHER_PID}" >&2
      exit 1
    fi
    if [[ -n "${LOCAL_STUDENT_PID:-}" ]] && ! kill -0 "${LOCAL_STUDENT_PID}" >/dev/null 2>&1; then
      echo "${label} endpoint process exited before becoming healthy: pid ${LOCAL_STUDENT_PID}" >&2
      exit 1
    fi
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for ${label} endpoint health: ${url}" >&2
      exit 1
    fi
    sleep "${POLL_SECONDS}"
    waited=$((waited + POLL_SECONDS))
    echo "Waiting for ${label} endpoint health (${waited}s): ${url}"
  done
}

load_teacher_endpoint_env_if_present() {
  if [[ -s "${TEACHER_ENDPOINT_ENV}" ]]; then
    set -a
    source "${TEACHER_ENDPOINT_ENV}"
    set +a
    export TEACHER_ENDPOINT_URL="${TEACHER_ENDPOINT_URL}"
    export TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"
    export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL}}"
  fi
}

start_local_teacher_for_online_attack() {
  local local_port
  local teacher_cuda_devices
  local endpoint_env_path

  local_port="${LOCAL_TEACHER_PORT:-}"
  if [[ -z "${local_port}" ]]; then
    local_port="$(find_free_port)"
  fi
  teacher_cuda_devices="${LOCAL_TEACHER_CUDA_VISIBLE_DEVICES:-0,1}"
  endpoint_env_path="${TEACHER_ENDPOINT_ENV}"

  export TEACHER_ENDPOINT_URL="http://127.0.0.1:${local_port}/v1"
  export TEACHER_API_KEY="${TEACHER_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
  export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL}}"
  export ATTACK_CUDA_VISIBLE_DEVICES="${ATTACK_CUDA_VISIBLE_DEVICES:-2}"
  mkdir -p "$(dirname "${endpoint_env_path}")"
  cat > "${endpoint_env_path}" <<EOF
export TEACHER_BACKEND="vllm_openai"
export TEACHER_MODEL="${TEACHER_MODEL}"
export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL}"
export TEACHER_ENDPOINT_URL="${TEACHER_ENDPOINT_URL}"
export TEACHER_API_KEY="${TEACHER_API_KEY}"
export STAGE2_TEACHER_BACKEND="vllm_openai"
export STAGE2_TEACHER_MODEL_PATH="${TEACHER_MODEL}"
export STAGE2_TEACHER_MODEL_NAME="${TEACHER_REQUEST_MODEL}"
export STAGE2_TEACHER_BASE_URL="${TEACHER_ENDPOINT_URL}"
export STAGE2_TEACHER_API_KEY="${TEACHER_API_KEY}"
EOF

  echo "Starting local teacher vLLM inside this ${ATTACK} job."
  echo "  teacher_cuda_visible_devices=${teacher_cuda_devices}"
  echo "  attack_cuda_visible_devices=${ATTACK_CUDA_VISIBLE_DEVICES}"
  echo "  teacher_endpoint_url=${TEACHER_ENDPOINT_URL}"

  env -u VLLM_PORT CUDA_VISIBLE_DEVICES="${teacher_cuda_devices}" python3 -m vllm.entrypoints.openai.api_server \
    --model "${TEACHER_MODEL}" \
    --served-model-name "${TEACHER_REQUEST_MODEL}" \
    --host "${LOCAL_TEACHER_HOST:-127.0.0.1}" \
    --port "${local_port}" \
    --api-key "${TEACHER_API_KEY}" \
    --tensor-parallel-size "${LOCAL_TEACHER_TENSOR_PARALLEL_SIZE:-2}" \
    --gpu-memory-utilization "${LOCAL_TEACHER_GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-model-len "${LOCAL_TEACHER_MAX_MODEL_LEN:-${VLLM_MAX_MODEL_LEN:-8096}}" \
    --dtype "${LOCAL_TEACHER_DTYPE:-${VLLM_DTYPE:-bfloat16}}" \
    ${LOCAL_TEACHER_EXTRA_ARGS:-${VLLM_EXTRA_ARGS:-}} &
  LOCAL_TEACHER_PID="$!"
  echo "Started local teacher vLLM pid: ${LOCAL_TEACHER_PID}"
  wait_for_endpoint_health "${TEACHER_ENDPOINT_URL}" "${TEACHER_API_KEY}" teacher
  echo "Local teacher endpoint is healthy for ${ATTACK}: ${TEACHER_ENDPOINT_URL}"
}

ensure_teacher_for_online_attack() {
  if [[ "${ATTACK_EXECUTION_STAGE}" == "train" ]]; then
    return 0
  fi
  case "${ATTACK}" in
    qedks|model_leeching) ;;
    *) return 0 ;;
  esac
  if [[ "${TEACHER_BACKEND}" != "vllm_openai" ]]; then
    return 0
  fi
  if [[ "${ONLINE_ATTACK_TEACHER_MODE}" == "local" ]]; then
    rm -f "${TEACHER_ENDPOINT_ENV}"
    start_local_teacher_for_online_attack
    return 0
  fi
  load_teacher_endpoint_env_if_present
  if check_openai_endpoint_once "${TEACHER_ENDPOINT_URL}" "${TEACHER_API_KEY:-EMPTY}" >/dev/null 2>&1; then
    echo "Teacher endpoint is healthy for ${ATTACK}: ${TEACHER_ENDPOINT_URL}"
    return 0
  fi
  if [[ "${AUTO_START_TEACHER_FOR_ONLINE_ATTACKS}" != "1" ]]; then
    echo "Teacher endpoint is not healthy for ${ATTACK}: ${TEACHER_ENDPOINT_URL}" >&2
    echo "AUTO_START_TEACHER_FOR_ONLINE_ATTACKS=0, not starting teacher vLLM." >&2
    exit 1
  fi

  rm -f "${TEACHER_ENDPOINT_ENV}"
  echo "Teacher endpoint is not healthy for ${ATTACK}; starting a local vLLM process."
  start_local_teacher_for_online_attack
}

cleanup_attack_started_teacher() {
  if [[ -n "${LOCAL_STUDENT_PID}" ]]; then
    echo "Stopping local student vLLM pid: ${LOCAL_STUDENT_PID}"
    kill "${LOCAL_STUDENT_PID}" >/dev/null 2>&1 || true
    wait "${LOCAL_STUDENT_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${LOCAL_TEACHER_PID}" ]]; then
    echo "Stopping local teacher vLLM pid: ${LOCAL_TEACHER_PID}"
    kill "${LOCAL_TEACHER_PID}" >/dev/null 2>&1 || true
    wait "${LOCAL_TEACHER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_attack_started_teacher EXIT

ensure_teacher_for_online_attack

export STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export STUDENT_ENDPOINT_URL="${STUDENT_ENDPOINT_URL:-http://127.0.0.1:8001/v1}"
export STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL:-${STUDENT_MODEL}}"
export STUDENT_API_KEY="${STUDENT_API_KEY:-EMPTY}"
export STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-0.7}"
export STUDENT_TOP_P="${STUDENT_TOP_P:-1.0}"
export STUDENT_MAX_TOKENS="${STUDENT_MAX_TOKENS:-1536}"
export WARMUP_MODEL="${WARMUP_MODEL:-}"
export STUDENT_ENDPOINT_ENV="${STUDENT_VLLM_ENDPOINT_ENV_PATH:-${STORAGE_ROOT}/outputs/vllm_student/student_endpoint.env}"

load_student_endpoint_env_if_present() {
  if [[ -s "${STUDENT_ENDPOINT_ENV}" ]]; then
    set -a
    source "${STUDENT_ENDPOINT_ENV}"
    set +a
    export STUDENT_ENDPOINT_URL="${STUDENT_ENDPOINT_URL}"
    export STUDENT_API_KEY="${STUDENT_API_KEY:-EMPTY}"
    export STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL:-${STUDENT_MODEL}}"
  fi
}

start_local_student_for_soda() {
  local local_port
  local student_cuda_devices
  local endpoint_env_path

  local_port="${LOCAL_STUDENT_PORT:-}"
  if [[ -z "${local_port}" ]]; then
    local_port="$(find_free_port)"
  fi
  student_cuda_devices="${LOCAL_STUDENT_CUDA_VISIBLE_DEVICES:-0}"
  endpoint_env_path="${STUDENT_ENDPOINT_ENV}"

  export STUDENT_ENDPOINT_URL="http://127.0.0.1:${local_port}/v1"
  export STUDENT_API_KEY="${STUDENT_API_KEY:-${STUDENT_VLLM_API_KEY:-EMPTY}}"
  export STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL:-${STUDENT_MODEL}}"
  export ATTACK_CUDA_VISIBLE_DEVICES="${ATTACK_CUDA_VISIBLE_DEVICES:-1}"
  mkdir -p "$(dirname "${endpoint_env_path}")"
  cat > "${endpoint_env_path}" <<EOF
export STUDENT_MODEL="${STUDENT_MODEL}"
export STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL}"
export STUDENT_ENDPOINT_URL="${STUDENT_ENDPOINT_URL}"
export STUDENT_API_KEY="${STUDENT_API_KEY}"
EOF

  echo "Starting local student vLLM inside this SODA job."
  echo "  student_cuda_visible_devices=${student_cuda_devices}"
  echo "  attack_cuda_visible_devices=${ATTACK_CUDA_VISIBLE_DEVICES}"
  echo "  student_endpoint_url=${STUDENT_ENDPOINT_URL}"

  env -u VLLM_PORT CUDA_VISIBLE_DEVICES="${student_cuda_devices}" python3 -m vllm.entrypoints.openai.api_server \
    --model "${STUDENT_MODEL}" \
    --served-model-name "${STUDENT_REQUEST_MODEL}" \
    --host "${LOCAL_STUDENT_HOST:-127.0.0.1}" \
    --port "${local_port}" \
    --api-key "${STUDENT_API_KEY}" \
    --tensor-parallel-size "${LOCAL_STUDENT_TENSOR_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization "${LOCAL_STUDENT_GPU_MEMORY_UTILIZATION:-0.85}" \
    --max-model-len "${LOCAL_STUDENT_MAX_MODEL_LEN:-${STUDENT_VLLM_MAX_MODEL_LEN:-4096}}" \
    --dtype "${LOCAL_STUDENT_DTYPE:-${STUDENT_VLLM_DTYPE:-bfloat16}}" \
    ${LOCAL_STUDENT_EXTRA_ARGS:-${STUDENT_VLLM_EXTRA_ARGS:-}} &
  LOCAL_STUDENT_PID="$!"
  echo "Started local student vLLM pid: ${LOCAL_STUDENT_PID}"
  wait_for_endpoint_health "${STUDENT_ENDPOINT_URL}" "${STUDENT_API_KEY}" student
  echo "Local student endpoint is healthy for SODA: ${STUDENT_ENDPOINT_URL}"
}

ensure_student_for_soda() {
  if [[ "${ATTACK_EXECUTION_STAGE}" == "train" ]]; then
    return 0
  fi
  if [[ "${ATTACK}" != "soda" ]]; then
    return 0
  fi
  if [[ "${SODA_STUDENT_MODE}" == "local" ]]; then
    rm -f "${STUDENT_ENDPOINT_ENV}"
    start_local_student_for_soda
    return 0
  fi
  load_student_endpoint_env_if_present
  if check_openai_endpoint_once "${STUDENT_ENDPOINT_URL}" "${STUDENT_API_KEY:-EMPTY}" >/dev/null 2>&1; then
    echo "Student endpoint is healthy for SODA: ${STUDENT_ENDPOINT_URL}"
    return 0
  fi
  echo "Student endpoint is not healthy for SODA: ${STUDENT_ENDPOINT_URL}" >&2
  echo "SODA_STUDENT_MODE=${SODA_STUDENT_MODE}; start student vLLM separately or set SODA_STUDENT_MODE=local." >&2
  exit 1
}

ensure_student_for_soda

COMMON_ARGS=(
  --attack "${ATTACK}"
  --budget "${ATTACK_BUDGET}"
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
  --teacher-temperature "${TEACHER_TEMPERATURE}"
  --teacher-top-p "${TEACHER_TOP_P}"
  --teacher-max-tokens "${TEACHER_MAX_TOKENS}"
  --student-model "${STUDENT_MODEL}"
  --student-endpoint-url "${STUDENT_ENDPOINT_URL}"
  --student-request-model "${STUDENT_REQUEST_MODEL}"
  --student-api-key "${STUDENT_API_KEY}"
  --student-temperature "${STUDENT_TEMPERATURE}"
  --student-top-p "${STUDENT_TOP_P}"
  --student-max-tokens "${STUDENT_MAX_TOKENS}"
  --execution-stage "${ATTACK_EXECUTION_STAGE}"
)
if [[ "${ATTACK_EXECUTION_STAGE}" == "train" ]]; then
  if [[ -z "${PREPARED_RUN_DIR}" ]]; then
    echo "ATTACK_EXECUTION_STAGE=train requires PREPARED_RUN_DIR." >&2
    exit 2
  fi
  COMMON_ARGS+=(--prepared-run-dir "${PREPARED_RUN_DIR}")
fi

if [[ -n "${WARMUP_MODEL}" ]]; then
  COMMON_ARGS+=(--warmup-model "${WARMUP_MODEL}")
fi

if [[ "${ATTACK_BF16:-1}" == "1" ]]; then
  COMMON_ARGS+=(--bf16)
else
  COMMON_ARGS+=(--no-bf16)
fi
if [[ "${ATTACK_USE_LORA:-1}" == "1" ]]; then
  COMMON_ARGS+=(--use-lora)
else
  COMMON_ARGS+=(--no-use-lora)
fi
if [[ "${ATTACK_GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  COMMON_ARGS+=(--gradient-checkpointing)
else
  COMMON_ARGS+=(--no-gradient-checkpointing)
fi
COMMON_ARGS+=(
  --lora-r "${ATTACK_LORA_R:-16}"
  --lora-alpha "${ATTACK_LORA_ALPHA:-32}"
  --lora-dropout "${ATTACK_LORA_DROPOUT:-0.05}"
)

USES_SHARED_TRANSCRIPT=0
case " ${SHARED_TRANSCRIPT_ATTACKS:-seqkd lord soda gad} " in
  *" ${ATTACK} "*) USES_SHARED_TRANSCRIPT=1 ;;
esac
if [[ "${USES_SHARED_TRANSCRIPT}" == "1" ]]; then
  if [[ -n "${SHARED_TRANSCRIPT_DIR:-}" ]]; then
    COMMON_ARGS+=(--transcript-dir "${SHARED_TRANSCRIPT_DIR}")
  elif [[ "${REQUIRE_SHARED_TRANSCRIPT:-0}" == "1" ]]; then
    echo "${ATTACK} is configured to require SHARED_TRANSCRIPT_DIR, but it is empty." >&2
    echo "Build it first with attacks/scripts/run_shared_transcript.sh, then export SHARED_TRANSCRIPT_DIR." >&2
    exit 1
  fi
fi

METHOD_ARGS=()
case "${ATTACK}" in
  soda)
    METHOD_ARGS+=(
      --soda-learning-rate "${SODA_LEARNING_RATE:-5e-6}"
      --soda-epochs "${SODA_EPOCHS:-1.0}"
      --soda-per-device-train-batch-size "${SODA_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
      --soda-gradient-accumulation-steps "${SODA_GRADIENT_ACCUMULATION_STEPS:-32}"
      --soda-max-length "${SODA_MAX_LENGTH:-3588}"
      --soda-max-prompt-length "${SODA_MAX_PROMPT_LENGTH:-1028}"
      --soda-max-grad-norm "${SODA_MAX_GRAD_NORM:-1.0}"
      --soda-nonfinite-gradient-retries "${SODA_NONFINITE_GRADIENT_RETRIES:-3}"
    )
    if [[ -n "${SODA_STUDENT_NEGATIVES_JSONL:-}" ]]; then
      METHOD_ARGS+=(--soda-student-negatives-jsonl "${SODA_STUDENT_NEGATIVES_JSONL}")
    fi
    if [[ -n "${SODA_PREFERENCES_JSONL:-}" ]]; then
      METHOD_ARGS+=(--soda-preferences-jsonl "${SODA_PREFERENCES_JSONL}")
    fi
    ;;
  qedks)
    METHOD_ARGS+=(
      --qedks-max-followups-per-answer "${QEDKS_MAX_FOLLOWUPS_PER_ANSWER:-8}"
      --qedks-learning-rate "${QEDKS_LEARNING_RATE:-2e-8}"
      --qedks-epochs "${QEDKS_EPOCHS:-2.0}"
      --qedks-lora-r "${QEDKS_LORA_R:-16}"
      --qedks-lora-alpha "${QEDKS_LORA_ALPHA:-32}"
      --qedks-lora-dropout "${QEDKS_LORA_DROPOUT:-0.05}"
      --qedks-per-device-train-batch-size "${QEDKS_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
      --qedks-gradient-accumulation-steps "${QEDKS_GRADIENT_ACCUMULATION_STEPS:-16}"
    )
    if [[ "${QEDKS_USE_PPL_SCHEDULE:-0}" == "1" ]]; then
      METHOD_ARGS+=(--qedks-use-ppl-schedule --qedks-ppl-device "${QEDKS_PPL_DEVICE:-cuda}")
    fi
    ;;
  model_leeching)
    METHOD_ARGS+=(
      --model-leeching-learning-rate "${MODEL_LEECHING_LEARNING_RATE:-2e-8}"
      --model-leeching-epochs "${MODEL_LEECHING_EPOCHS:-2.0}"
      --model-leeching-lora-r "${MODEL_LEECHING_LORA_R:-16}"
      --model-leeching-lora-alpha "${MODEL_LEECHING_LORA_ALPHA:-32}"
      --model-leeching-lora-dropout "${MODEL_LEECHING_LORA_DROPOUT:-0.05}"
      --model-leeching-per-device-train-batch-size "${MODEL_LEECHING_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
      --model-leeching-gradient-accumulation-steps "${MODEL_LEECHING_GRADIENT_ACCUMULATION_STEPS:-16}"
    )
    ;;
  gad)
    METHOD_ARGS+=(
      --gad-group-size "${GAD_GROUP_SIZE:-8}"
      --gad-kl-beta "${GAD_KL_BETA:-0.001}"
      --gad-learning-rate "${GAD_LEARNING_RATE:-1e-6}"
      --gad-discriminator-learning-rate "${GAD_DISCRIMINATOR_LEARNING_RATE:-1e-6}"
      --gad-warmup-learning-rate "${GAD_WARMUP_LEARNING_RATE:-5e-6}"
      --gad-warmup-epochs "${GAD_WARMUP_EPOCHS:-1.0}"
      --gad-epochs "${GAD_EPOCHS:-2.0}"
      --gad-discriminator-warmup-steps "${GAD_DISCRIMINATOR_WARMUP_STEPS:-10}"
      --gad-max-steps "${GAD_MAX_STEPS:--1}"
      --gad-per-device-train-batch-size "${GAD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
      --gad-gradient-accumulation-steps "${GAD_GRADIENT_ACCUMULATION_STEPS:-1}"
      --gad-max-seq-length "${GAD_MAX_SEQ_LENGTH:-3588}"
      --gad-max-prompt-length "${GAD_MAX_PROMPT_LENGTH:-2088}"
      --gad-max-response-length "${GAD_MAX_RESPONSE_LENGTH:-1536}"
      --gad-temperature "${GAD_TEMPERATURE:-0.8}"
      --gad-top-p "${GAD_TOP_P:-1.0}"
      --gad-memory-log-interval "${GAD_MEMORY_LOG_INTERVAL:-25}"
    )
    if [[ -n "${GAD_GENERATOR_DEVICE:-}" ]]; then
      METHOD_ARGS+=(--gad-generator-device "${GAD_GENERATOR_DEVICE}")
    fi
    if [[ -n "${GAD_DISCRIMINATOR_DEVICE:-}" ]]; then
      METHOD_ARGS+=(--gad-discriminator-device "${GAD_DISCRIMINATOR_DEVICE}")
    fi
    if [[ "${GAD_OFFLOAD_INACTIVE_MODELS:-0}" == "1" ]]; then
      METHOD_ARGS+=(--gad-offload-inactive-models)
    else
      METHOD_ARGS+=(--no-gad-offload-inactive-models)
    fi
    ;;
esac

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  METHOD_ARGS+=(--dry-run)
fi

echo "Full attack run configuration:"
echo "  REPO_DIR=${REPO_DIR}"
echo "  STORAGE_ROOT=${STORAGE_ROOT}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  TASK_ID=${TASK_ID}"
echo "  ATTACK=${ATTACK}"
echo "  BUDGET=${ATTACK_BUDGET}"
echo "  QUERY_POOL=${QUERY_POOL}"
echo "  TEACHER_BACKEND=${TEACHER_BACKEND}"
echo "  TEACHER_ENDPOINT_URL=${TEACHER_ENDPOINT_URL}"
echo "  TEACHER_REQUEST_MODEL=${TEACHER_REQUEST_MODEL}"
echo "  TEACHER_ENDPOINT_ENV=${TEACHER_ENDPOINT_ENV}"
echo "  AUTO_START_TEACHER_FOR_ONLINE_ATTACKS=${AUTO_START_TEACHER_FOR_ONLINE_ATTACKS}"
echo "  ONLINE_ATTACK_TEACHER_MODE=${ONLINE_ATTACK_TEACHER_MODE}"
echo "  LOCAL_TEACHER_PID=${LOCAL_TEACHER_PID:-}"
echo "  ATTACK_CUDA_VISIBLE_DEVICES=${ATTACK_CUDA_VISIBLE_DEVICES:-}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "  ATTACK_STARTED_TEACHER_JOB_ID=${ATTACK_STARTED_TEACHER_JOB_ID:-}"
echo "  ONLINE_ATTACK_TEACHER_JOB_ID_FILE=${ONLINE_ATTACK_TEACHER_JOB_ID_FILE}"
echo "  STUDENT_MODEL=${STUDENT_MODEL}"
echo "  STUDENT_ENDPOINT_URL=${STUDENT_ENDPOINT_URL}"
echo "  SODA_STUDENT_MODE=${SODA_STUDENT_MODE}"
echo "  LOCAL_STUDENT_PID=${LOCAL_STUDENT_PID:-}"
echo "  WARMUP_MODEL=${WARMUP_MODEL}"
echo "  ATTACK_BF16=${ATTACK_BF16:-1}"
echo "  ATTACK_USE_LORA=${ATTACK_USE_LORA:-1}"
echo "  ATTACK_GRADIENT_CHECKPOINTING=${ATTACK_GRADIENT_CHECKPOINTING:-1}"
echo "  GAD_OFFLOAD_INACTIVE_MODELS=${GAD_OFFLOAD_INACTIVE_MODELS:-0}"
echo "  GAD_MEMORY_LOG_INTERVAL=${GAD_MEMORY_LOG_INTERVAL:-25}"
echo "  GAD_GENERATOR_DEVICE=${GAD_GENERATOR_DEVICE:-}"
echo "  GAD_DISCRIMINATOR_DEVICE=${GAD_DISCRIMINATOR_DEVICE:-}"
echo "  SHARED_TRANSCRIPT_DIR=${SHARED_TRANSCRIPT_DIR:-}"
echo "  USES_SHARED_TRANSCRIPT=${USES_SHARED_TRANSCRIPT}"
echo "  DRY_RUN=${DRY_RUN:-0}"
echo "  ATTACK_EXECUTION_STAGE=${ATTACK_EXECUTION_STAGE}"
echo "  PREPARED_RUN_DIR=${PREPARED_RUN_DIR}"

if [[ -n "${ATTACK_CUDA_VISIBLE_DEVICES}" ]]; then
  CUDA_VISIBLE_DEVICES="${ATTACK_CUDA_VISIBLE_DEVICES}" python3 attacks/scripts/run_attack.py "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}"
else
  python3 attacks/scripts/run_attack.py "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}"
fi
