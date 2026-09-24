#!/usr/bin/env bash

set -euo pipefail

command -v module >/dev/null 2>&1 && module purge || true
command -v module >/dev/null 2>&1 && module load conda/25.7.0 || true
command -v module >/dev/null 2>&1 && module load cuda/12.8.1 || true

# Small real GAD smoke test. Override when submitting if needed, e.g.:
#   CONDA_ENV=research GAD_MAX_STEPS=2 GAD_GROUP_SIZE=2 bash attacks/scripts/smoke_gad.sh
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

python3 attacks/scripts/check_attack_env.py --require-trl --strict-versions

export BUDGET="${BUDGET:-100}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/attacks_gad_smoke}"
export QUERY_POOL="${QUERY_POOL:-auto}"
export QUERY_ORDERING="${QUERY_ORDERING:-auto}"
export STAGE1_CONFIG="${STAGE1_CONFIG:-attacks/configs/stage1_budget.yaml}"
export SEED="${SEED:-20260701}"

# Self-contained default: local HF teacher/student, no vLLM server needed.
export TEACHER_BACKEND="${TEACHER_BACKEND:-local_hf}"
export TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL}}"
export TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"
export TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-64}"

export STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
# GAD uses --warmup-model as the discriminator-D base model. Default keeps G/D same scale.
export WARMUP_MODEL="${WARMUP_MODEL:-${STUDENT_MODEL}}"

# Keep GAD smoke intentionally tiny. Full defaults are K=8, max_response=1536, max_steps ~= budget * epochs.
export GAD_GROUP_SIZE="${GAD_GROUP_SIZE:-2}"
export GAD_MAX_STEPS="${GAD_MAX_STEPS:-1}"
export GAD_DISCRIMINATOR_WARMUP_STEPS="${GAD_DISCRIMINATOR_WARMUP_STEPS:-1}"
export GAD_WARMUP_EPOCHS="${GAD_WARMUP_EPOCHS:-0.1}"
export GAD_EPOCHS="${GAD_EPOCHS:-0.1}"
export GAD_MAX_SEQ_LENGTH="${GAD_MAX_SEQ_LENGTH:-1024}"
export GAD_MAX_PROMPT_LENGTH="${GAD_MAX_PROMPT_LENGTH:-512}"
export GAD_MAX_RESPONSE_LENGTH="${GAD_MAX_RESPONSE_LENGTH:-64}"
export GAD_PER_DEVICE_TRAIN_BATCH_SIZE="${GAD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GAD_GRADIENT_ACCUMULATION_STEPS="${GAD_GRADIENT_ACCUMULATION_STEPS:-1}"
export GAD_LEARNING_RATE="${GAD_LEARNING_RATE:-1e-6}"
export GAD_DISCRIMINATOR_LEARNING_RATE="${GAD_DISCRIMINATOR_LEARNING_RATE:-1e-6}"
export GAD_WARMUP_LEARNING_RATE="${GAD_WARMUP_LEARNING_RATE:-5e-6}"
export GAD_KL_BETA="${GAD_KL_BETA:-0.001}"
export GAD_TEMPERATURE="${GAD_TEMPERATURE:-0.8}"
export GAD_TOP_P="${GAD_TOP_P:-1.0}"

DEVICE_ARGS=()
if [[ -n "${GAD_GENERATOR_DEVICE:-}" ]]; then
  DEVICE_ARGS+=(--gad-generator-device "${GAD_GENERATOR_DEVICE}")
fi
if [[ -n "${GAD_DISCRIMINATOR_DEVICE:-}" ]]; then
  DEVICE_ARGS+=(--gad-discriminator-device "${GAD_DISCRIMINATOR_DEVICE}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARG=(--dry-run)
else
  DRY_RUN_ARG=()
fi

echo "GAD smoke configuration:"
echo "  REPO_DIR=${REPO_DIR}"
echo "  STORAGE_ROOT=${STORAGE_ROOT}"
echo "  HF_HOME=${HF_HOME}"
echo "  BUDGET=${BUDGET}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  QUERY_POOL=${QUERY_POOL}"
echo "  TEACHER_BACKEND=${TEACHER_BACKEND}"
echo "  TEACHER_MODEL=${TEACHER_MODEL}"
echo "  STUDENT_MODEL=${STUDENT_MODEL}"
echo "  WARMUP_MODEL=${WARMUP_MODEL}"
echo "  GAD_GROUP_SIZE=${GAD_GROUP_SIZE}"
echo "  GAD_MAX_STEPS=${GAD_MAX_STEPS}"
echo "  GAD_DISCRIMINATOR_WARMUP_STEPS=${GAD_DISCRIMINATOR_WARMUP_STEPS}"
echo "  GAD_MAX_RESPONSE_LENGTH=${GAD_MAX_RESPONSE_LENGTH}"
echo "  GAD_GENERATOR_DEVICE=${GAD_GENERATOR_DEVICE:-}"
echo "  GAD_DISCRIMINATOR_DEVICE=${GAD_DISCRIMINATOR_DEVICE:-}"
echo "  DRY_RUN=${DRY_RUN:-0}"

python3 attacks/scripts/run_attack.py   --attack gad   --budget "${BUDGET}"   --query-pool "${QUERY_POOL}"   --query-ordering "${QUERY_ORDERING}"   --stage1-config "${STAGE1_CONFIG}"   --output-dir "${OUTPUT_DIR}"   --seed "${SEED}"   --teacher-backend "${TEACHER_BACKEND}"   --teacher-model "${TEACHER_MODEL}"   --teacher-request-model "${TEACHER_REQUEST_MODEL}"   --teacher-api-key "${TEACHER_API_KEY}"   --teacher-max-tokens "${TEACHER_MAX_TOKENS}"   --student-model "${STUDENT_MODEL}"   --warmup-model "${WARMUP_MODEL}"   --gad-group-size "${GAD_GROUP_SIZE}"   --gad-kl-beta "${GAD_KL_BETA}"   --gad-learning-rate "${GAD_LEARNING_RATE}"   --gad-discriminator-learning-rate "${GAD_DISCRIMINATOR_LEARNING_RATE}"   --gad-warmup-learning-rate "${GAD_WARMUP_LEARNING_RATE}"   --gad-warmup-epochs "${GAD_WARMUP_EPOCHS}"   --gad-epochs "${GAD_EPOCHS}"   --gad-discriminator-warmup-steps "${GAD_DISCRIMINATOR_WARMUP_STEPS}"   --gad-max-steps "${GAD_MAX_STEPS}"   --gad-per-device-train-batch-size "${GAD_PER_DEVICE_TRAIN_BATCH_SIZE}"   --gad-gradient-accumulation-steps "${GAD_GRADIENT_ACCUMULATION_STEPS}"   --gad-max-seq-length "${GAD_MAX_SEQ_LENGTH}"   --gad-max-prompt-length "${GAD_MAX_PROMPT_LENGTH}"   --gad-max-response-length "${GAD_MAX_RESPONSE_LENGTH}"   --gad-temperature "${GAD_TEMPERATURE}"   --gad-top-p "${GAD_TOP_P}"   "${DEVICE_ARGS[@]}"   "${DRY_RUN_ARG[@]}"
