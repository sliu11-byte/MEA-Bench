#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${METHOD:-}" || -z "${ATTACK:-}" || -z "${BUDGET:-}" ]]; then
  echo "METHOD, ATTACK, and BUDGET must be set before sourcing runs/defense/common.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"
mkdir -p logs

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

export OMP_NUM_THREADS="${CPU_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

export STORAGE_ROOT="${STORAGE_ROOT:-${REPO_DIR}}"

# The paper defense benchmark uses the Qwen teacher/student pair for every
# attack and defense condition.
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-7B}"

FULL_RUN_ENV="${FULL_RUN_ENV:-}"
if [[ -n "${FULL_RUN_ENV}" && -f "${FULL_RUN_ENV}" ]]; then
  set -a
  source "${FULL_RUN_ENV}"
  set +a
fi

export METHOD
export ATTACK
export BUDGET
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

DEFAULT_TEACHER_MODEL="Qwen/Qwen2.5-72B-Instruct"
DEFAULT_STUDENT_MODEL="Qwen/Qwen2.5-7B"
export TEACHER_MODEL="${TEACHER_MODEL:-${DEFAULT_TEACHER_MODEL}}"
export TEACHER_ENDPOINT_URL="${TEACHER_ENDPOINT_URL:-http://127.0.0.1:4000/v1}"
export TEACHER_BASE_URL="${TEACHER_BASE_URL:-${TEACHER_ENDPOINT_URL}}"
export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL}}"
export TEACHER_API_KEY="${TEACHER_API_KEY:-EMPTY}"
export TEACHER_MODE="${TEACHER_MODE:-chat}"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.0}"
export TEACHER_TOP_P="${TEACHER_TOP_P:-1.0}"
export TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-1536}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${TEACHER_MAX_TOKENS}}"

export STUDENT_MODEL="${STUDENT_MODEL:-${DEFAULT_STUDENT_MODEL}}"
export STUDENT_ENDPOINT_URL="${STUDENT_ENDPOINT_URL:-http://127.0.0.1:4001/v1}"
export STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL:-${STUDENT_MODEL}}"
export STUDENT_API_KEY="${STUDENT_API_KEY:-EMPTY}"
export STUDENT_MODE="${STUDENT_MODE:-chat}"
export STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-0.7}"
export STUDENT_TOP_P="${STUDENT_TOP_P:-1.0}"
export STUDENT_MAX_TOKENS="${STUDENT_MAX_TOKENS:-1536}"
export WARMUP_MODEL="${WARMUP_MODEL:-}"

export QUERY_POOL="${QUERY_POOL:-auto}"
export QUERY_ORDERING="${QUERY_ORDERING:-auto}"
export STAGE1_CONFIG="${STAGE1_CONFIG:-attacks/configs/formal_stage1_budget.yaml}"
export SEED="${SEED:-20260701}"
export DEVICE="${DEVICE:-cuda}"
export PROXY_MODEL="${PROXY_MODEL:-${STUDENT_MODEL}}"
export DEFENSE_EXECUTION_MODE="${DEFENSE_EXECUTION_MODE:-auto}"
export ADS_BATCH_SIZE="${ADS_BATCH_SIZE:-4}"
export ADS_ALLOW_LAM_BATCH="${ADS_ALLOW_LAM_BATCH:-0}"
export ADS_GRAD_MODEL_DTYPE="${ADS_GRAD_MODEL_DTYPE:-auto}"
export ADS_GRAD_DTYPE="${ADS_GRAD_DTYPE:-float32}"
export ADFP_BATCH_SIZE="${ADFP_BATCH_SIZE:-4}"
if [[ "${TEACHER_MODEL}" != "Qwen/Qwen2.5-72B-Instruct" || "${STUDENT_MODEL}" != "Qwen/Qwen2.5-7B" ]]; then
  echo "The paper defense profile requires TEACHER_MODEL=Qwen/Qwen2.5-72B-Instruct and STUDENT_MODEL=Qwen/Qwen2.5-7B" >&2
  exit 2
fi
if [[ "${ATTACK}" == "soda" ]]; then
  export OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/defenses/${ATTACK}_b${BUDGET}_seqkd_warmup}"
else
  export OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/defenses/${ATTACK}_b${BUDGET}}"
fi
export LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/defenses/${ATTACK}_b${BUDGET}}"
export DEFENSE_OUT="${DEFENSE_OUT:-${OUTPUT_ROOT}/${METHOD}}"
mkdir -p "${DEFENSE_OUT}" "${LOG_ROOT}/${METHOD}"

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
    DEFENSE_CONFIG="${DEFENSE_CONFIG:-$(printf '{\"gamma\":0.5,\"window_size\":2,\"strength_lambda\":140.0,\"batch_size\":%s}' "${ADFP_BATCH_SIZE}")}"
    METHOD_ARGS=(
      --proxy-model "${ADFP_PROXY_MODEL:-${PROXY_MODEL}}"
      --adfp-batch-size "${ADFP_BATCH_SIZE}"
      --defense-config "${DEFENSE_CONFIG}"
    )
    ;;
  ads)
    if [[ "${ATTACK}" == "soda" ]]; then
      # Replay uses the pinned SeqKD transcript, so ADS gradient preparation
      # and a second teacher generation are unnecessary.
      METHOD_ARGS=(--no-detector)
    else
    ADS_WORK_DIR="${ADS_WORK_DIR:-${OUTPUT_ROOT}/ads/prep}"
    ADS_HOLDOUT="${ADS_HOLDOUT:-2440}"
    ADS_GRAD_MAX_LENGTH="${ADS_GRAD_MAX_LENGTH:-512}"
    mkdir -p "${ADS_WORK_DIR}"
    GRAD_PATH="${GRAD_PATH:-${ADS_WORK_DIR}/student_grads.pt}"
    if [[ ! -f "${GRAD_PATH}" ]]; then
      HOLDOUT_PATH="${ADS_HOLDOUT_TRANSCRIPT:-${ADS_WORK_DIR}/holdout_teacher.jsonl}"
      if [[ ! -f "${HOLDOUT_PATH}" ]]; then
        python3 - "$BUDGET" "$ADS_HOLDOUT" "$QUERY_POOL" "$ADS_WORK_DIR" "$TEACHER_MODEL" "$PROXY_MODEL" "$DEVICE" "$MAX_NEW_TOKENS" "$ADS_BATCH_SIZE" "$HOLDOUT_PATH" <<'PY'
from pathlib import Path
import sys

from attacks.core.hf_query_pool import hf_query_pool_for_budget, resolve_query_pool_and_ordering
from defenses.ads.generator import ADSGenerator
from defenses.core.io_utils import load_queries, write_jsonl

budget = int(sys.argv[1])
holdout = int(sys.argv[2])
query_pool = sys.argv[3]
out_dir = Path(sys.argv[4])
teacher_model = sys.argv[5]
proxy_model = sys.argv[6]
device = sys.argv[7]
max_new_tokens = int(sys.argv[8])
batch_size = int(sys.argv[9])
holdout_path = Path(sys.argv[10])

query_pool_spec = hf_query_pool_for_budget(budget + holdout) if query_pool == "auto" else query_pool
query_pool_path, _ = resolve_query_pool_and_ordering(query_pool_spec, "auto")
queries = load_queries(query_pool_path, max_queries=budget + holdout)[budget:budget + holdout]
if len(queries) < holdout:
    raise SystemExit(f"Need {holdout} holdout queries, got {len(queries)} from {query_pool_path}")

out_dir.mkdir(parents=True, exist_ok=True)
gen = ADSGenerator(teacher_model, proxy_student=proxy_model, device=device)
records = gen.generate(
    queries,
    config={"lam": 0.0, "tau": 0.9, "top_p": 0.95, "max_new_tokens": max_new_tokens, "batch_size": batch_size},
)
write_jsonl(records, holdout_path)
print({"holdout_transcript": str(holdout_path), "num_records": len(records)})
PY
      fi
      python3 -m defenses.ads.save_grad \
        --proxy_student "${PROXY_MODEL}" \
        --holdout_transcript "${HOLDOUT_PATH}" \
        --output_path "${GRAD_PATH}" \
        --max_length "${ADS_GRAD_MAX_LENGTH}" \
        --device "${DEVICE}" \
        --model_dtype "${ADS_GRAD_MODEL_DTYPE}" \
        --grad_dtype "${ADS_GRAD_DTYPE}"
    fi
    if [[ "$ADS_ALLOW_LAM_BATCH" == "1" || "$ADS_ALLOW_LAM_BATCH" == "true" ]]; then
      ADS_ALLOW_LAM_BATCH_JSON=true
    else
      ADS_ALLOW_LAM_BATCH_JSON=false
    fi
    DEFENSE_CONFIG="${DEFENSE_CONFIG:-$(printf '{\"lam\":0.1,\"eps\":0.01,\"tau\":0.9,\"top_p\":0.95,\"max_new_tokens\":%s,\"batch_size\":%s,\"allow_lam_batch\":%s,\"proxy_student\":\"%s\",\"grad_path\":\"%s\"}' "$MAX_NEW_TOKENS" "$ADS_BATCH_SIZE" "$ADS_ALLOW_LAM_BATCH_JSON" "$PROXY_MODEL" "$GRAD_PATH")}"
    METHOD_ARGS=(--proxy-model "${PROXY_MODEL}" --ads-batch-size "${ADS_BATCH_SIZE}" --grad-path "${GRAD_PATH}" --defense-config "${DEFENSE_CONFIG}" --no-detector)
    fi
    ;;
  trace_rewriting)
    export REWRITER_BACKEND="${REWRITER_BACKEND:-openai_compatible}"
    export REWRITER_BASE_URL="${REWRITER_BASE_URL:-${TEACHER_BASE_URL}}"
    export REWRITER_REQUEST_MODEL="${REWRITER_REQUEST_MODEL:-${REWRITER_MODEL:-${TEACHER_MODEL}}}"
    DEFENSE_CONFIG="${DEFENSE_CONFIG:-$(printf '{\"rewrite_strategy\":\"optimized_prompt_official\",\"teacher_max_new_tokens\":%s,\"max_new_tokens\":%s,\"temperature\":0.6,\"top_p\":0.95}' "$MAX_NEW_TOKENS" "$MAX_NEW_TOKENS")}"
    METHOD_ARGS=(
      --teacher-base-url "${TEACHER_BASE_URL}"
      --teacher-request-model "${TEACHER_REQUEST_MODEL}"
      --rewriter-backend "${REWRITER_BACKEND}"
      --rewriter-base-url "${REWRITER_BASE_URL}"
      --rewriter-request-model "${REWRITER_REQUEST_MODEL}"
      --rewriter-api-key "${REWRITER_API_KEY:-EMPTY}"
      --rewriter-model "${REWRITER_MODEL:-${TEACHER_MODEL}}"
      --defense-config "${DEFENSE_CONFIG}"
      --no-detector
    )
    ;;
  doge)
    if [[ "${ATTACK}" == "soda" ]]; then
      # Replay needs neither a new DOGe teacher head nor new oracle responses.
      METHOD_ARGS=(--no-detector)
    else
    DOGE_WORK_DIR="${DOGE_WORK_DIR:-${OUTPUT_ROOT}/doge/prep}"
    DOGE_CHECKPOINT="${DOGE_CHECKPOINT:-${DOGE_WORK_DIR}/doge_teacher}"
    DOGE_TRAIN_FILE="${DOGE_TRAIN_FILE:-${DOGE_WORK_DIR}/doge_train.jsonl}"
    DOGE_TRAIN_EXAMPLES="${DOGE_TRAIN_EXAMPLES:-${BUDGET}}"
    mkdir -p "${DOGE_WORK_DIR}"
    if [[ ! -f "${DOGE_TRAIN_FILE}" ]]; then
      python3 - "$BUDGET" "$DOGE_TRAIN_EXAMPLES" "$QUERY_POOL" "$DOGE_TRAIN_FILE" "$TEACHER_BASE_URL" "$TEACHER_REQUEST_MODEL" "$TEACHER_API_KEY" "$MAX_NEW_TOKENS" "$TEACHER_TEMPERATURE" "$TEACHER_TOP_P" <<'PY'
from pathlib import Path
import sys

from attacks.core.hf_query_pool import hf_query_pool_for_budget, resolve_query_pool_and_ordering
from defenses.core.generation import generate_openai_chat_responses
from defenses.core.io_utils import load_queries, write_jsonl

budget = int(sys.argv[1])
num_examples = int(sys.argv[2])
query_pool = sys.argv[3]
train_file = Path(sys.argv[4])
base_url = sys.argv[5]
model = sys.argv[6]
api_key = sys.argv[7]
max_new_tokens = int(sys.argv[8])
temperature = float(sys.argv[9])
top_p = float(sys.argv[10])

needed = max(budget, num_examples)
query_pool_spec = hf_query_pool_for_budget(needed) if query_pool == "auto" else query_pool
query_pool_path, _ = resolve_query_pool_and_ordering(query_pool_spec, "auto")
queries = load_queries(query_pool_path, max_queries=num_examples)
if len(queries) < num_examples:
    raise SystemExit(f"Need {num_examples} DOGe training queries, got {len(queries)} from {query_pool_path}")
responses = generate_openai_chat_responses(
    [q["query"] for q in queries],
    base_url=base_url,
    model=model,
    api_key=api_key,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
)
rows = []
for q, response in zip(queries, responses):
    rows.append({
        "query_id": q.get("query_id"),
        "query": q["query"],
        "teacher_response": response,
        "teacher_model": model,
        "source": "doge_auto_teacher_collection",
    })
train_file.parent.mkdir(parents=True, exist_ok=True)
write_jsonl(rows, train_file)
print(train_file)
PY
    fi
    if [[ ! -f "${DOGE_CHECKPOINT}/lm_head.pt" ]]; then
      python3 -m defenses.doge.run train \
        --teacher-model "${TEACHER_MODEL}" \
        --proxy-model "${DOGE_PROXY_MODEL:-${TEACHER_MODEL}}" \
        --train-file "${DOGE_TRAIN_FILE}" \
        --output-dir "${DOGE_CHECKPOINT}" \
        --anti-kd-coef "${DOGE_ANTI_KD_COEF:-3e-5}" \
        --kd-temperature "${DOGE_KD_TEMPERATURE:-2.0}" \
        --learning-rate "${DOGE_LEARNING_RATE:-5e-5}" \
        --weight-decay "${DOGE_WEIGHT_DECAY:-0.01}" \
        --batch-size "${DOGE_BATCH_SIZE:-1}" \
        --gradient-accumulation-steps "${DOGE_GRADIENT_ACCUMULATION_STEPS:-16}" \
        --num-train-epochs "${DOGE_NUM_TRAIN_EPOCHS:-2}" \
        ${DOGE_MAX_STEPS:+--max-steps "$DOGE_MAX_STEPS"} \
        --max-length "${DOGE_MAX_LENGTH:-2044}"
    fi
    METHOD_ARGS=(--doge-checkpoint "${DOGE_CHECKPOINT}" --no-detector)
    fi
    ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

RUNNER_REUSE_ARGS=()

ATTACK_ARGS=(
  --student-endpoint-url "${STUDENT_ENDPOINT_URL}"
  --student-request-model "${STUDENT_REQUEST_MODEL}"
  --student-api-key "${STUDENT_API_KEY}"
  --student-mode "${STUDENT_MODE}"
  --student-temperature "${STUDENT_TEMPERATURE}"
  --student-top-p "${STUDENT_TOP_P}"
  --student-max-tokens "${STUDENT_MAX_TOKENS}"
  --bf16
  --use-lora
  --gradient-checkpointing
  --lora-r "${ATTACK_LORA_R:-16}"
  --lora-alpha "${ATTACK_LORA_ALPHA:-32}"
  --lora-dropout "${ATTACK_LORA_DROPOUT:-0.05}"
)
if [[ -n "${WARMUP_MODEL}" ]]; then
  ATTACK_ARGS+=(--warmup-model "${WARMUP_MODEL}")
fi

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
      --soda-max-grad-norm "${SODA_MAX_GRAD_NORM:-1.0}"
      --soda-nonfinite-gradient-retries "${SODA_NONFINITE_GRADIENT_RETRIES:-3}"
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
Formal defense experiment:
  METHOD=${METHOD}
  ATTACK=${ATTACK}
  BUDGET=${BUDGET}
  REPO_DIR=${REPO_DIR}
  STORAGE_ROOT=${STORAGE_ROOT}
  OUTPUT_DIR=${DEFENSE_OUT}
  LOG_ROOT=${LOG_ROOT}
  TEACHER_MODEL=${TEACHER_MODEL}
  STUDENT_MODEL=${STUDENT_MODEL}
  TEACHER_BASE_URL=${TEACHER_BASE_URL}
  STUDENT_ENDPOINT_URL=${STUDENT_ENDPOINT_URL}
  DEFENSE_EXECUTION_MODE=${DEFENSE_EXECUTION_MODE}
  ADS_BATCH_SIZE=${ADS_BATCH_SIZE}
  ADS_ALLOW_LAM_BATCH=${ADS_ALLOW_LAM_BATCH}
  ADS_GRAD_MODEL_DTYPE=${ADS_GRAD_MODEL_DTYPE}
  ADS_GRAD_DTYPE=${ADS_GRAD_DTYPE}
  ADFP_BATCH_SIZE=${ADFP_BATCH_SIZE}
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
  --device "${DEVICE}" \
  --output-dir "${DEFENSE_OUT}" \
  --execution-mode "${DEFENSE_EXECUTION_MODE}" \
  --ads-batch-size "${ADS_BATCH_SIZE}" \
  --adfp-batch-size "${ADFP_BATCH_SIZE}" \
  "${METHOD_ARGS[@]}" \
  -- "${ATTACK_ARGS[@]}"
