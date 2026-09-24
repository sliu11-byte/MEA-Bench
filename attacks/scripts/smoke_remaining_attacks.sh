#!/usr/bin/env bash

set -euo pipefail

command -v module >/dev/null 2>&1 && module purge || true
command -v module >/dev/null 2>&1 && module load conda/25.7.0 || true
command -v module >/dev/null 2>&1 && module load cuda/12.8.1 || true

# Override when submitting if needed, e.g.:
#   CONDA_ENV=research ATTACKS="lord model_leeching" bash attacks/scripts/smoke_remaining_attacks.sh
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

DEFAULT_STORAGE_ROOT="/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey"
export STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

python3 attacks/scripts/check_attack_env.py --require-trl --require-vllm --strict-versions

export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/attacks_remaining_smoke}"
mkdir -p "${OUTPUT_ROOT}"

# Start one small OpenAI-compatible vLLM endpoint for methods that require endpoint access.
export START_VLLM="${START_VLLM:-1}"
export VLLM_MODEL="${VLLM_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-mea_smoke_qwen25_05b}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
export VLLM_LOG="${VLLM_LOG:-${OUTPUT_ROOT}/vllm_server.log}"

wait_for_vllm() {
  python3 - "$VLLM_BASE_URL" <<'PY'
import sys
import time
import urllib.request

base_url = sys.argv[1].rstrip("/")
last_error = None
for _ in range(180):
    try:
        with urllib.request.urlopen(base_url + "/models", timeout=5) as response:
            if response.status < 500:
                print(f"vLLM endpoint is ready: {base_url}")
                raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(5)
raise SystemExit(f"Timed out waiting for vLLM endpoint {base_url}: {last_error}")
PY
}

VLLM_PID=""
if [[ "${START_VLLM}" == "1" ]]; then
  echo "Starting vLLM smoke endpoint: ${VLLM_MODEL} as ${VLLM_SERVED_NAME} on ${VLLM_BASE_URL}"
  env -u VLLM_PORT python3 -m vllm.entrypoints.openai.api_server \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --model "${VLLM_MODEL}" \
    --served-model-name "${VLLM_SERVED_NAME}" \
    --tensor-parallel-size 1 \
    --dtype auto \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code \
    >"${VLLM_LOG}" 2>&1 &
  VLLM_PID="$!"
  trap 'if [[ -n "${VLLM_PID}" ]]; then kill "${VLLM_PID}" 2>/dev/null || true; fi' EXIT
  wait_for_vllm
else
  echo "Using existing vLLM endpoint: ${VLLM_BASE_URL}"
fi

export QUERY_POOL="${QUERY_POOL:-auto}"
export QUERY_ORDERING="${QUERY_ORDERING:-auto}"
export STAGE1_CONFIG="${STAGE1_CONFIG:-attacks/configs/stage1_budget.yaml}"
export SEED="${SEED:-20260701}"
export TEACHER_MODEL="${TEACHER_MODEL:-${VLLM_MODEL}}"
export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${VLLM_SERVED_NAME}}"
export TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"
export STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
export STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL:-${VLLM_SERVED_NAME}}"
export STUDENT_API_KEY="${STUDENT_API_KEY:-EMPTY}"
export WARMUP_MODEL="${WARMUP_MODEL:-}"

# LoRD/SODA reuse Stage-1 transcript code and currently require one of 100/1000/10000.
export LORD_BUDGET="${LORD_BUDGET:-100}"
export SODA_BUDGET="${SODA_BUDGET:-100}"
# QEDKS and Model Leeching can be smaller for smoke tests.
export QEDKS_BUDGET="${QEDKS_BUDGET:-20}"
export MODEL_LEECHING_BUDGET="${MODEL_LEECHING_BUDGET:-20}"
export ATTACKS="${ATTACKS:-lord model_leeching qedks soda}"

run_attack() {
  local attack="$1"
  local budget="$2"
  echo "[RUN] attack=${attack} budget=${budget}"
  python3 attacks/scripts/run_attack.py \
    --attack "${attack}" \
    --budget "${budget}" \
    --query-pool "${QUERY_POOL}" \
    --query-ordering "${QUERY_ORDERING}" \
    --stage1-config "${STAGE1_CONFIG}" \
    --output-dir "${OUTPUT_ROOT}" \
    --seed "${SEED}" \
    --teacher-backend vllm_openai \
    --teacher-model "${TEACHER_MODEL}" \
    --teacher-endpoint-url "${VLLM_BASE_URL}" \
    --teacher-request-model "${TEACHER_REQUEST_MODEL}" \
    --teacher-api-key "${TEACHER_API_KEY}" \
    --teacher-temperature "${TEACHER_TEMPERATURE:-0.0}" \
    --teacher-top-p "${TEACHER_TOP_P:-1.0}" \
    --teacher-max-tokens "${TEACHER_MAX_TOKENS:-128}" \
    --student-model "${STUDENT_MODEL}" \
    --student-endpoint-url "${VLLM_BASE_URL}" \
    --student-request-model "${STUDENT_REQUEST_MODEL}" \
    --student-api-key "${STUDENT_API_KEY}" \
    --student-temperature "${STUDENT_TEMPERATURE:-0.7}" \
    --student-top-p "${STUDENT_TOP_P:-1.0}" \
    --student-max-tokens "${STUDENT_MAX_TOKENS:-256}" \
    --warmup-model "${WARMUP_MODEL}" \
    --soda-epochs "${SODA_EPOCHS:-0.2}" \
    --soda-gradient-accumulation-steps "${SODA_GRADIENT_ACCUMULATION_STEPS:-8}" \
    --soda-max-length "${SODA_MAX_LENGTH:-1024}" \
    --soda-max-prompt-length "${SODA_MAX_PROMPT_LENGTH:-512}" \
    --qedks-epochs "${QEDKS_EPOCHS:-0.2}" \
    --qedks-gradient-accumulation-steps "${QEDKS_GRADIENT_ACCUMULATION_STEPS:-8}" \
    --model-leeching-epochs "${MODEL_LEECHING_EPOCHS:-0.2}" \
    --model-leeching-gradient-accumulation-steps "${MODEL_LEECHING_GRADIENT_ACCUMULATION_STEPS:-8}"
  echo "[DONE] attack=${attack}"
}

echo "Remaining smoke configuration:"
echo "  REPO_DIR=${REPO_DIR}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  ATTACKS=${ATTACKS}"
echo "  VLLM_BASE_URL=${VLLM_BASE_URL}"
echo "  VLLM_MODEL=${VLLM_MODEL}"
echo "  STUDENT_MODEL=${STUDENT_MODEL}"
echo "  WARMUP_MODEL=${WARMUP_MODEL}"

if [[ " ${ATTACKS} " == *" soda "* && -z "${WARMUP_MODEL}" ]]; then
  echo "[WARMUP] No SODA warmup supplied; running SeqKD budget=${SODA_BUDGET} first."
  run_attack seqkd "${SODA_BUDGET}"
  WARMUP_MODEL="$(find "${OUTPUT_ROOT}/seqkd" -path "*/budget_${SODA_BUDGET}/seqkd/checkpoint-final" -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "${WARMUP_MODEL}" ]]; then
    echo "SeqKD smoke completed without a discoverable checkpoint-final directory." >&2
    exit 1
  fi
  export WARMUP_MODEL
  echo "[WARMUP] SODA will reuse ${WARMUP_MODEL}"
fi

for attack in ${ATTACKS}; do
  case "${attack}" in
    lord) run_attack lord "${LORD_BUDGET}" ;;
    model_leeching) run_attack model_leeching "${MODEL_LEECHING_BUDGET}" ;;
    qedks) run_attack qedks "${QEDKS_BUDGET}" ;;
    soda) run_attack soda "${SODA_BUDGET}" ;;
    gad) echo "[SKIP] use attacks/scripts/smoke_gad.sh for GAD smoke." ;;
    *) echo "Unknown attack for remaining smoke: ${attack}" >&2; exit 1 ;;
  esac
  echo
done
