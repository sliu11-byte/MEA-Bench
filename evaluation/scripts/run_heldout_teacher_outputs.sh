#!/usr/bin/env bash

set -euo pipefail

SUBMIT_DIR="${REPO_BASE_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/evaluation/scripts/build_heldout_teacher_outputs.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey/evaluation/scripts/build_heldout_teacher_outputs.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey"
else
  echo "Could not locate A-Benchmark-for-Model-distillation-survey from submit dir: ${SUBMIT_DIR}" >&2
  echo "Submit from the repository root." >&2
  exit 2
fi
cd "${REPO_DIR}"
mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module purge || true
  command -v module >/dev/null 2>&1 && module load conda/25.7.0 || true || true
  command -v module >/dev/null 2>&1 && module load cuda/12.8.1 || true || true
fi
if [[ -f /usr/local/anaconda3/etc/profile.d/conda.sh ]]; then
  source /usr/local/anaconda3/etc/profile.d/conda.sh
fi
if command -v conda >/dev/null 2>&1; then
  command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV:-research}" || true
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
export STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HELDOUT_DIR="${HELDOUT_DIR:-${STORAGE_ROOT}/outputs/heldout_queries}"
export HELDOUT_PROMPTS_JSONL="${HELDOUT_PROMPTS_JSONL:-${REPO_DIR}/evaluation/data/heldout/heldout_prompts.jsonl}"
export HELDOUT_TEACHER_OUTPUTS_JSONL="${HELDOUT_TEACHER_OUTPUTS_JSONL:-${HELDOUT_DIR}/heldout_teacher_outputs.jsonl}"
export HELDOUT_TEACHER_MANIFEST="${HELDOUT_TEACHER_MANIFEST:-${HELDOUT_DIR}/heldout_teacher_outputs.manifest.json}"

# Default: direct local Hugging Face generation on allocated GPUs.
# Set HELDOUT_TEACHER_BACKEND=openai to use an already-running OpenAI-compatible endpoint instead.
HELDOUT_TEACHER_BACKEND="${HELDOUT_TEACHER_BACKEND:-local_hf}"
if [[ "${HELDOUT_TEACHER_BACKEND}" == "openai" ]]; then
  TEACHER_ENDPOINT_ENV_PATH="${TEACHER_ENDPOINT_ENV_PATH:-${VLLM_ENDPOINT_ENV_PATH:-${STORAGE_ROOT}/outputs/vllm_teacher/teacher_endpoint.env}}"
  if [[ -f "${TEACHER_ENDPOINT_ENV_PATH}" ]]; then
    set -a
    source "${TEACHER_ENDPOINT_ENV_PATH}"
    set +a
  fi
fi

TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL}}"
TEACHER_ENDPOINT_URL="${TEACHER_ENDPOINT_URL:-${STAGE2_TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}}"
TEACHER_API_KEY="${TEACHER_API_KEY:-${STAGE2_TEACHER_API_KEY:-EMPTY}}"
TEACHER_MODE="${TEACHER_MODE:-chat}"
TEACHER_TORCH_DTYPE="${TEACHER_TORCH_DTYPE:-bfloat16}"
TEACHER_DEVICE_MAP="${TEACHER_DEVICE_MAP:-auto}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-1536}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.0}"
TEACHER_TOP_P="${TEACHER_TOP_P:-1.0}"
SEED="${SEED:-20260701}"
HELDOUT_LIMIT="${HELDOUT_LIMIT:-}"
HELDOUT_BUILD_IF_MISSING="${HELDOUT_BUILD_IF_MISSING:-1}"

mkdir -p "${HELDOUT_DIR}"

if [[ ! -f "${HELDOUT_PROMPTS_JSONL}" ]]; then
  if [[ "${HELDOUT_BUILD_IF_MISSING}" != "1" ]]; then
    echo "Missing heldout prompts: ${HELDOUT_PROMPTS_JSONL}" >&2
    exit 2
  fi
  BUILD_ARGS=(--output-dir "${HELDOUT_DIR}")
  if [[ -n "${HELDOUT_QUERY_POOL:-}" ]]; then
    BUILD_ARGS+=(--query-pool "${HELDOUT_QUERY_POOL}")
  fi
  if [[ -n "${HELDOUT_MAX_TRAIN_BUDGET:-}" ]]; then
    BUILD_ARGS+=(--max-train-budget "${HELDOUT_MAX_TRAIN_BUDGET}")
  fi
  if [[ -n "${HELDOUT_BLOCK_SIZE:-}" ]]; then
    BUILD_ARGS+=(--block-size "${HELDOUT_BLOCK_SIZE}")
  fi
  if [[ -n "${HELDOUT_NUM_BLOCKS:-}" ]]; then
    BUILD_ARGS+=(--num-blocks "${HELDOUT_NUM_BLOCKS}")
  fi
  if [[ -n "${HELDOUT_START_BLOCK:-}" ]]; then
    BUILD_ARGS+=(--start-block "${HELDOUT_START_BLOCK}")
  fi
  if [[ -n "${HELDOUT_BLOCK_IDS:-}" ]]; then
    BUILD_ARGS+=(--block-ids "${HELDOUT_BLOCK_IDS}")
  fi
  python3 -m evaluation.scripts.build_heldout_queries "${BUILD_ARGS[@]}"
fi

cat <<EOF
Heldout teacher-output job:
  REPO_DIR=${REPO_DIR}
  STORAGE_ROOT=${STORAGE_ROOT}
  HELDOUT_PROMPTS_JSONL=${HELDOUT_PROMPTS_JSONL}
  HELDOUT_TEACHER_OUTPUTS_JSONL=${HELDOUT_TEACHER_OUTPUTS_JSONL}
  TEACHER_MODEL=${TEACHER_MODEL}
  TEACHER_REQUEST_MODEL=${TEACHER_REQUEST_MODEL}
  HELDOUT_TEACHER_BACKEND=${HELDOUT_TEACHER_BACKEND}
  TEACHER_ENDPOINT_URL=${TEACHER_ENDPOINT_URL}
  TEACHER_TORCH_DTYPE=${TEACHER_TORCH_DTYPE}
  TEACHER_DEVICE_MAP=${TEACHER_DEVICE_MAP}
  TEACHER_MODE=${TEACHER_MODE}
  TEACHER_MAX_TOKENS=${TEACHER_MAX_TOKENS}
  HELDOUT_LIMIT=${HELDOUT_LIMIT:-<none>}
EOF

ARGS=(
  --prompts-jsonl "${HELDOUT_PROMPTS_JSONL}"
  --output-jsonl "${HELDOUT_TEACHER_OUTPUTS_JSONL}"
  --manifest-output "${HELDOUT_TEACHER_MANIFEST}"
  --teacher-model "${TEACHER_MODEL}"
  --backend "${HELDOUT_TEACHER_BACKEND}"
  --mode "${TEACHER_MODE}"
  --max-tokens "${TEACHER_MAX_TOKENS}"
  --temperature "${TEACHER_TEMPERATURE}"
  --top-p "${TEACHER_TOP_P}"
  --seed "${SEED}"
)
if [[ "${HELDOUT_TEACHER_BACKEND}" == "openai" ]]; then
  ARGS+=(
    --request-model "${TEACHER_REQUEST_MODEL}"
    --base-url "${TEACHER_ENDPOINT_URL}"
    --api-key "${TEACHER_API_KEY}"
  )
else
  ARGS+=(
    --torch-dtype "${TEACHER_TORCH_DTYPE}"
    --device-map "${TEACHER_DEVICE_MAP}"
  )
fi
if [[ -n "${HELDOUT_LIMIT}" ]]; then
  ARGS+=(--limit "${HELDOUT_LIMIT}")
fi

exec python3 -m evaluation.scripts.build_heldout_teacher_outputs "${ARGS[@]}"
