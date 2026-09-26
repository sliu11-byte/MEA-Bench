#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${COUNTERMEASURE:-}" || -z "${METHOD:-}" || -z "${ATTACK:-}" || -z "${BUDGET:-}" ]]; then
  echo "COUNTERMEASURE, METHOD, ATTACK, and BUDGET must be set before sourcing runs/counter/common.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"
mkdir -p logs
export RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_DIR}/logs/${RUN_NAME:-counter}_${RUN_ID:-local}}"
mkdir -p "${RUN_LOG_DIR}"
echo "Stage logs: ${RUN_LOG_DIR}; stage output is also streamed into this job's main log."

if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi

export OMP_NUM_THREADS="${CPU_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

cat <<EOF
GPU allocation:
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
EOF
python3 - <<'PY'
try:
    import torch
    print(f"  torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count={torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"  torch.cuda.device[{idx}]={torch.cuda.get_device_name(idx)}")
except Exception as exc:
    print(f"  torch cuda inspection failed: {exc}")
PY

case "${COUNTERMEASURE}" in
  dipper|translation) ;;
  *) echo "Unknown COUNTERMEASURE=${COUNTERMEASURE}" >&2; exit 2 ;;
esac

case "${METHOD}" in
  ginsew|radioactivity|adfp) ;;
  *) echo "Counter experiments only cover ginsew, radioactivity, and adfp. Got METHOD=${METHOD}" >&2; exit 2 ;;
esac

export STORAGE_ROOT="${STORAGE_ROOT:-${REPO_DIR}}"

FULL_RUN_ENV="${FULL_RUN_ENV:-}"
if [[ -n "${FULL_RUN_ENV}" && -f "${FULL_RUN_ENV}" ]]; then
  set -a
  source "${FULL_RUN_ENV}"
  set +a
fi

export COUNTERMEASURE
export METHOD
export ATTACK
export BUDGET
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

export TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
export TEACHER_ENDPOINT_URL="${TEACHER_ENDPOINT_URL:-http://127.0.0.1:4000/v1}"
export TEACHER_BASE_URL="${TEACHER_BASE_URL:-${TEACHER_ENDPOINT_URL}}"
export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL}}"
export TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"
export TEACHER_MODE="${TEACHER_MODE:-chat}"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.0}"
export TEACHER_TOP_P="${TEACHER_TOP_P:-1.0}"
export TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-1536}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${TEACHER_MAX_TOKENS}}"

export STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export STUDENT_ENDPOINT_URL="${STUDENT_ENDPOINT_URL:-http://127.0.0.1:4001/v1}"
export STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL:-${STUDENT_MODEL}}"
export STUDENT_API_KEY="${STUDENT_API_KEY:-EMPTY}"
export STUDENT_MODE="${STUDENT_MODE:-chat}"
export STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-0.7}"
export STUDENT_TOP_P="${STUDENT_TOP_P:-1.0}"
export STUDENT_MAX_TOKENS="${STUDENT_MAX_TOKENS:-1536}"
export WARMUP_MODEL="${WARMUP_MODEL:-${STUDENT_MODEL}}"

export QUERY_POOL="${QUERY_POOL:-auto}"
export QUERY_ORDERING="${QUERY_ORDERING:-auto}"
export STAGE1_CONFIG="${STAGE1_CONFIG:-attacks/configs/formal_stage1_budget.yaml}"
export DEVICE="${DEVICE:-cuda}"
export PROXY_MODEL="${PROXY_MODEL:-${STUDENT_MODEL}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/countermeasures/${ATTACK}_b${BUDGET}/${COUNTERMEASURE}}"
export LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/countermeasures/${ATTACK}_b${BUDGET}/${COUNTERMEASURE}}"
export RUN_OUT="${RUN_OUT:-${OUTPUT_ROOT}/${METHOD}}"
mkdir -p "${RUN_OUT}" "${LOG_ROOT}/${METHOD}"

export COUNTERMEASURE_LEX="${COUNTERMEASURE_LEX:-40}"
export COUNTERMEASURE_ORDER="${COUNTERMEASURE_ORDER:-0}"
export COUNTERMEASURE_SENT_INTERVAL="${COUNTERMEASURE_SENT_INTERVAL:-3}"
export COUNTERMEASURE_WITH_CONTEXT="${COUNTERMEASURE_WITH_CONTEXT:-0}"
export COUNTERMEASURE_MODEL_NAME="${COUNTERMEASURE_MODEL_NAME:-}"
export COUNTERMEASURE_TOKENIZER_NAME="${COUNTERMEASURE_TOKENIZER_NAME:-}"
export COUNTERMEASURE_DEVICE="${COUNTERMEASURE_DEVICE:-}"
export COUNTERMEASURE_TIMEOUT="${COUNTERMEASURE_TIMEOUT:-3600}"
# The local Transformers oracle serializes generation. Concurrent client
# requests only accumulate queue time inside the proxy timeout.
export ATTACK_QUERY_CONCURRENCY="${COUNTER_ATTACK_QUERY_CONCURRENCY:-1}"
export QEDKS_REQUEST_TIMEOUT_SECONDS="${QEDKS_REQUEST_TIMEOUT_SECONDS:-3900}"
if [[ -z "${COUNTERMEASURE_DEVICE}" && "${METHOD}" == "adfp" ]]; then
  # ADFP's teacher and proxy normally occupy cuda:0; its Slurm jobs request a
  # second GPU so the response rewriter can coexist with them.
  export COUNTERMEASURE_DEVICE="cuda:1"
fi

python3 attacks/scripts/check_attack_env.py --require-trl --strict-versions

case "${METHOD}" in
  ginsew)
    DEFENSE_CONFIG="${DEFENSE_CONFIG:-{\"fraction\":0.5,\"strength\":2.0,\"freq\":16,\"eps\":0.2}}"
    METHOD_ARGS=(--defense-config "${DEFENSE_CONFIG}")
    ;;
  radioactivity)
    DEFENSE_CONFIG="${DEFENSE_CONFIG:-{\"method\":\"maryland\",\"ngram\":4,\"seed\":0,\"seeding\":\"hash\",\"hash_key\":35317,\"gamma\":0.25,\"delta\":2.0,\"scoring_method\":\"v2\"}}"
    METHOD_ARGS=(--defense-config "${DEFENSE_CONFIG}")
    ;;
  adfp)
    DEFENSE_CONFIG="${DEFENSE_CONFIG:-{\"gamma\":0.5,\"window_size\":2,\"strength_lambda\":140.0}}"
    METHOD_ARGS=(--proxy-model "${ADFP_PROXY_MODEL:-${PROXY_MODEL}}" --defense-config "${DEFENSE_CONFIG}")
    ;;
esac

COUNTERMEASURE_ARGS=(
  --countermeasure "${COUNTERMEASURE}"
  --countermeasure-timeout "${COUNTERMEASURE_TIMEOUT}"
  --countermeasure-lex "${COUNTERMEASURE_LEX}"
  --countermeasure-order "${COUNTERMEASURE_ORDER}"
  --countermeasure-sent-interval "${COUNTERMEASURE_SENT_INTERVAL}"
)
if [[ "${COUNTER_BASELINES:-1}" == "1" ]]; then
  COUNTERMEASURE_ARGS+=(--counter-baselines)
fi
if [[ "${PREPARE_COUNTER_BASELINES:-0}" == "1" ]]; then
  COUNTERMEASURE_ARGS+=(--prepare-counter-baselines)
fi
if [[ "${REQUIRE_COUNTER_BASELINES:-0}" == "1" ]]; then
  COUNTERMEASURE_ARGS+=(--require-counter-baselines)
fi
if [[ "${COUNTERMEASURE_WITH_CONTEXT}" == "1" || "${COUNTERMEASURE_WITH_CONTEXT}" == "true" ]]; then
  COUNTERMEASURE_ARGS+=(--countermeasure-with-context)
fi
if [[ -n "${COUNTERMEASURE_MODEL_NAME}" ]]; then
  COUNTERMEASURE_ARGS+=(--countermeasure-model-name "${COUNTERMEASURE_MODEL_NAME}")
fi
if [[ -n "${COUNTERMEASURE_TOKENIZER_NAME}" ]]; then
  COUNTERMEASURE_ARGS+=(--countermeasure-tokenizer-name "${COUNTERMEASURE_TOKENIZER_NAME}")
fi
if [[ -n "${COUNTERMEASURE_DEVICE}" ]]; then
  COUNTERMEASURE_ARGS+=(--countermeasure-device "${COUNTERMEASURE_DEVICE}")
fi

ATTACK_ARGS=(
  --student-endpoint-url "${STUDENT_ENDPOINT_URL}"
  --student-request-model "${STUDENT_REQUEST_MODEL}"
  --student-api-key "${STUDENT_API_KEY}"
  --student-mode "${STUDENT_MODE}"
  --student-temperature "${STUDENT_TEMPERATURE}"
  --student-top-p "${STUDENT_TOP_P}"
  --student-max-tokens "${STUDENT_MAX_TOKENS}"
  --warmup-model "${WARMUP_MODEL}"
  --bf16
  --use-lora
  --gradient-checkpointing
  --lora-r "${ATTACK_LORA_R:-16}"
  --lora-alpha "${ATTACK_LORA_ALPHA:-32}"
  --lora-dropout "${ATTACK_LORA_DROPOUT:-0.05}"
)

case "${ATTACK}" in
  seqkd|lord) ;;
  soda)
    ATTACK_ARGS+=(
      --soda-learning-rate "${SODA_LEARNING_RATE:-5e-6}"
      --soda-epochs "${SODA_EPOCHS:-1.0}"
      --soda-per-device-train-batch-size "${SODA_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
      --soda-gradient-accumulation-steps "${SODA_GRADIENT_ACCUMULATION_STEPS:-32}"
      --soda-max-length "${SODA_MAX_LENGTH:-3544}"
      --soda-max-prompt-length "${SODA_MAX_PROMPT_LENGTH:-1024}"
    )
    ;;
  qedks)
    ATTACK_ARGS+=(
      --qedks-max-followups-per-answer "${QEDKS_MAX_FOLLOWUPS_PER_ANSWER:-4}"
      --qedks-learning-rate "${QEDKS_LEARNING_RATE:-2e-4}"
      --qedks-epochs "${QEDKS_EPOCHS:-2.0}"
      --qedks-lora-r "${QEDKS_LORA_R:-16}"
      --qedks-lora-alpha "${QEDKS_LORA_ALPHA:-32}"
      --qedks-lora-dropout "${QEDKS_LORA_DROPOUT:-0.05}"
      --qedks-per-device-train-batch-size "${QEDKS_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
      --qedks-gradient-accumulation-steps "${QEDKS_GRADIENT_ACCUMULATION_STEPS:-16}"
    )
    if [[ "${QEDKS_USE_PPL_SCHEDULE:-0}" == "1" ]]; then
      ATTACK_ARGS+=(--qedks-use-ppl-schedule --qedks-ppl-device "${QEDKS_PPL_DEVICE:-cuda}")
    fi
    ;;
  model_leeching)
    ATTACK_ARGS+=(
      --model-leeching-learning-rate "${MODEL_LEECHING_LEARNING_RATE:-2e-4}"
      --model-leeching-epochs "${MODEL_LEECHING_EPOCHS:-2.0}"
      --model-leeching-lora-r "${MODEL_LEECHING_LORA_R:-16}"
      --model-leeching-lora-alpha "${MODEL_LEECHING_LORA_ALPHA:-32}"
      --model-leeching-lora-dropout "${MODEL_LEECHING_LORA_DROPOUT:-0.05}"
      --model-leeching-per-device-train-batch-size "${MODEL_LEECHING_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
      --model-leeching-gradient-accumulation-steps "${MODEL_LEECHING_GRADIENT_ACCUMULATION_STEPS:-16}"
    )
    ;;
  gad)
    ATTACK_ARGS+=(
      --gad-group-size "${GAD_GROUP_SIZE:-4}"
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
      --gad-max-seq-length "${GAD_MAX_SEQ_LENGTH:-3544}"
      --gad-max-prompt-length "${GAD_MAX_PROMPT_LENGTH:-2044}"
      --gad-max-response-length "${GAD_MAX_RESPONSE_LENGTH:-1536}"
      --gad-temperature "${GAD_TEMPERATURE:-0.4}"
      --gad-top-p "${GAD_TOP_P:-1.0}"
      --gad-memory-log-interval "${GAD_MEMORY_LOG_INTERVAL:-25}"
      --gad-generator-device "${GAD_GENERATOR_DEVICE:-cuda:0}"
      --gad-discriminator-device "${GAD_DISCRIMINATOR_DEVICE:-cuda:1}"
      --no-gad-offload-inactive-models
    )
    ;;
  *)
    echo "Unknown ATTACK=${ATTACK}" >&2
    exit 2
    ;;
esac

cat <<EOF
Countermeasure experiment:
  COUNTERMEASURE=${COUNTERMEASURE}
  DEFENSE=${METHOD}
  ATTACK=${ATTACK}
  BUDGET=${BUDGET}
  OUTPUT_DIR=${RUN_OUT}
  LOG_ROOT=${LOG_ROOT}
  TEACHER_MODEL=${TEACHER_MODEL}
  STUDENT_MODEL=${STUDENT_MODEL}
  ATTACK_QUERY_CONCURRENCY=${ATTACK_QUERY_CONCURRENCY}
  COUNTERMEASURE_TIMEOUT=${COUNTERMEASURE_TIMEOUT}
  QEDKS_REQUEST_TIMEOUT_SECONDS=${QEDKS_REQUEST_TIMEOUT_SECONDS}
EOF

python3 -m "defenses.${METHOD}.runner" \
  --attack "${ATTACK}" \
  --budget "${BUDGET}" \
  --query-pool "${QUERY_POOL}" \
  --query-ordering "${QUERY_ORDERING}" \
  --stage1-config "${STAGE1_CONFIG}" \
  --teacher-model "${TEACHER_MODEL}" \
  --student-model "${STUDENT_MODEL}" \
  --teacher-mode "${TEACHER_MODE}" \
  --teacher-max-tokens "${MAX_NEW_TOKENS}" \
  --teacher-temperature "${TEACHER_TEMPERATURE}" \
  --teacher-top-p "${TEACHER_TOP_P}" \
  --detector-max-queries "${DETECTOR_MAX_QUERIES:-1000}" \
  --detector-max-new-tokens "${DETECTOR_MAX_NEW_TOKENS:-124}" \
  --detector-temperature "${DETECTOR_TEMPERATURE:-0.7}" \
  --detector-seed "${DETECTOR_SEED:-42}" \
  --device "${DEVICE}" \
  --output-dir "${RUN_OUT}" \
  "${COUNTERMEASURE_ARGS[@]}" \
  "${METHOD_ARGS[@]}" \
  -- "${ATTACK_ARGS[@]}"
