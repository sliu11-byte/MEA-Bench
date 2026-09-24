#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_env STUDENT_CHECKPOINT
DUFFIN_OUT="${DUFFIN_OUT:-$OUTPUT_ROOT/duffin}"
mkdir -p "$DUFFIN_OUT" "$LOG_ROOT/duffin"
set +e
python3 -m defenses.duffin.runner \
  --teacher_model "$TEACHER_MODEL" \
  --student_checkpoint "$STUDENT_CHECKPOINT" \
  --output_dir "$DUFFIN_OUT" \
  --max_probes "${DUFFIN_MAX_PROBES:-1000}" \
  --max_new_tokens "${DUFFIN_MAX_NEW_TOKENS:-1024}" \
  --min_valid_rate "${DUFFIN_MIN_VALID_RATE:-0.9}" \
  --batch_size "${DUFFIN_BATCH_SIZE:-4}" \
  --probe_source "${DUFFIN_PROBE_SOURCE:-mmlu_pro}" \
  --probe_categories "${DUFFIN_PROBE_CATEGORIES:-biology,business,chemistry,computer_science,math,physics}"
status=$?
set -e
mirror_defense_logs duffin "$DUFFIN_OUT"
exit "$status"
