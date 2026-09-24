#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Defaults run the real attack pipelines. Set DRY_RUN=1 for manifest-only checks.
DRY_RUN=${DRY_RUN:-0}
ATTACKS=${ATTACKS:-seqkd lord soda qedks model_leeching gad}
BUDGET=${BUDGET:-100}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/attacks}
QUERY_POOL=${QUERY_POOL:-auto}
QUERY_ORDERING=${QUERY_ORDERING:-auto}
STAGE1_CONFIG=${STAGE1_CONFIG:-attacks/configs/stage1_budget.yaml}
SEED=${SEED:-20260701}

TEACHER_BACKEND=${TEACHER_BACKEND:-vllm_openai}
TEACHER_MODEL=${TEACHER_MODEL:-meta-llama/Llama-3.3-70B-Instruct}
TEACHER_ENDPOINT_URL=${TEACHER_ENDPOINT_URL:-http://127.0.0.1:8000/v1}
TEACHER_REQUEST_MODEL=${TEACHER_REQUEST_MODEL:-$TEACHER_MODEL}
TEACHER_API_KEY=${TEACHER_API_KEY:-EMPTY}
TEACHER_MODE=${TEACHER_MODE:-chat}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-0.0}
TEACHER_TOP_P=${TEACHER_TOP_P:-1.0}
TEACHER_MAX_TOKENS=${TEACHER_MAX_TOKENS:-512}

STUDENT_MODEL=${STUDENT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
STUDENT_ENDPOINT_URL=${STUDENT_ENDPOINT_URL:-http://127.0.0.1:8001/v1}
STUDENT_REQUEST_MODEL=${STUDENT_REQUEST_MODEL:-$STUDENT_MODEL}
STUDENT_API_KEY=${STUDENT_API_KEY:-EMPTY}
STUDENT_MODE=${STUDENT_MODE:-chat}
STUDENT_TEMPERATURE=${STUDENT_TEMPERATURE:-0.7}
STUDENT_TOP_P=${STUDENT_TOP_P:-1.0}
STUDENT_MAX_TOKENS=${STUDENT_MAX_TOKENS:-1536}

# SODA auto-discovers a compatible completed SeqKD checkpoint when omitted.
WARMUP_MODEL=${WARMUP_MODEL:-}

COMMON_ARGS=(
  --budget "$BUDGET"
  --query-pool "$QUERY_POOL"
  --query-ordering "$QUERY_ORDERING"
  --stage1-config "$STAGE1_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --seed "$SEED"
  --teacher-backend "$TEACHER_BACKEND"
  --teacher-model "$TEACHER_MODEL"
  --teacher-endpoint-url "$TEACHER_ENDPOINT_URL"
  --teacher-request-model "$TEACHER_REQUEST_MODEL"
  --teacher-api-key "$TEACHER_API_KEY"
  --teacher-mode "$TEACHER_MODE"
  --teacher-temperature "$TEACHER_TEMPERATURE"
  --teacher-top-p "$TEACHER_TOP_P"
  --teacher-max-tokens "$TEACHER_MAX_TOKENS"
  --student-model "$STUDENT_MODEL"
  --student-endpoint-url "$STUDENT_ENDPOINT_URL"
  --student-request-model "$STUDENT_REQUEST_MODEL"
  --student-api-key "$STUDENT_API_KEY"
  --student-mode "$STUDENT_MODE"
  --student-temperature "$STUDENT_TEMPERATURE"
  --student-top-p "$STUDENT_TOP_P"
  --student-max-tokens "$STUDENT_MAX_TOKENS"
)

if [[ -n "$WARMUP_MODEL" ]]; then
  COMMON_ARGS+=(--warmup-model "$WARMUP_MODEL")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

for attack in $ATTACKS; do
  echo "[RUN] attack=$attack budget=$BUDGET dry_run=$DRY_RUN"
  python3 attacks/scripts/run_attack.py \
    --attack "$attack" \
    "${COMMON_ARGS[@]}"
  echo "[DONE] attack=$attack"
  echo
done
