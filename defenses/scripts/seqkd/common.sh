#!/usr/bin/env bash
set -euo pipefail

resolve_repo_root() {
  local start="${1:-$PWD}"
  local dir
  dir="$(cd "$start" && pwd)"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/attacks/scripts/run_attack.py" && -d "$dir/defenses" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  echo "Could not locate A-Benchmark-for-Model-distillation-survey from $start" >&2
  return 2
}

REPO_ROOT="${REPO_ROOT:-$(resolve_repo_root "$(dirname "${BASH_SOURCE[0]}")")}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-7B}"
export PROXY_MODEL="${PROXY_MODEL:-$STUDENT_MODEL}"
export ADS_BATCH_SIZE="${ADS_BATCH_SIZE:-4}"
export ADS_ALLOW_LAM_BATCH="${ADS_ALLOW_LAM_BATCH:-0}"
export ADS_GRAD_MODEL_DTYPE="${ADS_GRAD_MODEL_DTYPE:-auto}"
export ADS_GRAD_DTYPE="${ADS_GRAD_DTYPE:-float32}"
export ADFP_BATCH_SIZE="${ADFP_BATCH_SIZE:-4}"
export BUDGET="${BUDGET:-1000}"
export QUERY_POOL="${QUERY_POOL:-auto}"
export QUERY_ORDERING="${QUERY_ORDERING:-auto}"
export STAGE1_CONFIG="${STAGE1_CONFIG:-attacks/configs/formal_stage1_budget.yaml}"
export TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-1536}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$TEACHER_MAX_TOKENS}"
export TEACHER_MODE="${TEACHER_MODE:-chat}"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.0}"
export TEACHER_TOP_P="${TEACHER_TOP_P:-1.0}"
export DETECTOR_MAX_QUERIES="${DETECTOR_MAX_QUERIES:-1000}"
export DETECTOR_MAX_NEW_TOKENS="${DETECTOR_MAX_NEW_TOKENS:-128}"
export DETECTOR_TEMPERATURE="${DETECTOR_TEMPERATURE:-0.7}"
export DEVICE="${DEVICE:-cuda}"
export COMMON_ATTACK_EXTRA="${COMMON_ATTACK_EXTRA:-}"
export DEFENSE_EXECUTION_MODE="${DEFENSE_EXECUTION_MODE:-auto}"
export COUNTERMEASURE="${COUNTERMEASURE:-none}"
export COUNTERMEASURE_LEX="${COUNTERMEASURE_LEX:-40}"
export COUNTERMEASURE_ORDER="${COUNTERMEASURE_ORDER:-0}"
export COUNTERMEASURE_SENT_INTERVAL="${COUNTERMEASURE_SENT_INTERVAL:-3}"
export COUNTERMEASURE_WITH_CONTEXT="${COUNTERMEASURE_WITH_CONTEXT:-0}"
export COUNTERMEASURE_MODEL_NAME="${COUNTERMEASURE_MODEL_NAME:-}"
export COUNTERMEASURE_TOKENIZER_NAME="${COUNTERMEASURE_TOKENIZER_NAME:-}"
export COUNTERMEASURE_DEVICE="${COUNTERMEASURE_DEVICE:-}"

DEFAULT_STORAGE_ROOT="${STORAGE_ROOT:-$REPO_ROOT}"
if [[ -z "${STORAGE_ROOT:-}" && -d "/path/to/storage/$USER" ]]; then
  DEFAULT_STORAGE_ROOT="/path/to/storage/$USER/A-Benchmark-for-Model-distillation-survey"
fi
export STORAGE_ROOT="${STORAGE_ROOT:-$DEFAULT_STORAGE_ROOT}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/defenses/seqkd_b${BUDGET}}"
DEFAULT_LOG_ROOT="$REPO_ROOT/logs/defenses/seqkd_b${BUDGET}"
if [[ -d "/home/$USER/project/A-Benchmark-for-Model-distillation-survey" ]]; then
  DEFAULT_LOG_ROOT="/home/$USER/project/A-Benchmark-for-Model-distillation-survey/logs/defenses/seqkd_b${BUDGET}"
fi
export LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"

mirror_defense_logs() {
  local method="$1"
  local out="$2"
  local dest="${LOG_ROOT}/${method}"
  mkdir -p "$dest"
  python3 - "$out" "$dest" <<'PY'
from pathlib import Path
import shutil
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
patterns = (
    "*.log",
    "*.out",
    "*.err",
    "*manifest.json",
    "*report*.json",
    "*metrics*.json",
    "*summary*.json",
)
if not src.exists():
    raise SystemExit(0)
seen = set()
for pattern in patterns:
    for path in src.rglob(pattern):
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
print(f"[logs] mirrored {len(seen)} light log/manifest files -> {dst}")
PY
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 2
  fi
}

run_seqkd_defense_runner() {
  local method="$1"
  shift
  local out="${DEFENSE_OUT:-$OUTPUT_ROOT/$method}"
  mkdir -p "$out" "$LOG_ROOT/$method"
  local countermeasure_args=(
    --countermeasure "$COUNTERMEASURE"
    --countermeasure-lex "$COUNTERMEASURE_LEX"
    --countermeasure-order "$COUNTERMEASURE_ORDER"
    --countermeasure-sent-interval "$COUNTERMEASURE_SENT_INTERVAL"
  )
  if [[ "$COUNTERMEASURE_WITH_CONTEXT" == "1" || "$COUNTERMEASURE_WITH_CONTEXT" == "true" ]]; then
    countermeasure_args+=(--countermeasure-with-context)
  fi
  if [[ -n "$COUNTERMEASURE_MODEL_NAME" ]]; then
    countermeasure_args+=(--countermeasure-model-name "$COUNTERMEASURE_MODEL_NAME")
  fi
  if [[ -n "$COUNTERMEASURE_TOKENIZER_NAME" ]]; then
    countermeasure_args+=(--countermeasure-tokenizer-name "$COUNTERMEASURE_TOKENIZER_NAME")
  fi
  if [[ -n "$COUNTERMEASURE_DEVICE" ]]; then
    countermeasure_args+=(--countermeasure-device "$COUNTERMEASURE_DEVICE")
  fi

  set +e
  python3 -m "defenses.${method}.runner" \
    --attack seqkd \
    --budget "$BUDGET" \
    --query-pool "$QUERY_POOL" \
    --query-ordering "$QUERY_ORDERING" \
    --stage1-config "$STAGE1_CONFIG" \
    --teacher-model "$TEACHER_MODEL" \
    --student-model "$STUDENT_MODEL" \
    --teacher-mode "$TEACHER_MODE" \
    --teacher-max-tokens "$MAX_NEW_TOKENS" \
    --teacher-temperature "$TEACHER_TEMPERATURE" \
    --teacher-top-p "$TEACHER_TOP_P" \
    --detector-max-queries "$DETECTOR_MAX_QUERIES" \
    --detector-max-new-tokens "$DETECTOR_MAX_NEW_TOKENS" \
    --detector-temperature "$DETECTOR_TEMPERATURE" \
    --device "$DEVICE" \
    --output-dir "$out" \
    --execution-mode "$DEFENSE_EXECUTION_MODE" \
    --ads-batch-size "$ADS_BATCH_SIZE" \
    --adfp-batch-size "$ADFP_BATCH_SIZE" \
    "${countermeasure_args[@]}" \
    "$@" \
    ${COMMON_ATTACK_EXTRA:+-- $COMMON_ATTACK_EXTRA}
  local status=$?
  set -e
  mirror_defense_logs "$method" "$out"
  return "$status"
}
