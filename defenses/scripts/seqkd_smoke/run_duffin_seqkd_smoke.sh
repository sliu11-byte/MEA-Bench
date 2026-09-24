#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DUFFIN_OUT="${DUFFIN_OUT:-$OUTPUT_ROOT/duffin}"
mkdir -p "$DUFFIN_OUT"
python3 -m defenses.duffin.runner \
  --teacher_model "$TEACHER_MODEL" \
  --student_checkpoint "$STUDENT_MODEL" \
  --output_dir "$DUFFIN_OUT" \
  --max_probes "${DUFFIN_MAX_PROBES:-5}" \
  --max_new_tokens "${DUFFIN_MAX_NEW_TOKENS:-4}" \
  --min_valid_rate "${DUFFIN_MIN_VALID_RATE:-0.0}" \
  --probe_source "${DUFFIN_PROBE_SOURCE:-mmlu_pro}" \
  --probe_categories "${DUFFIN_PROBE_CATEGORIES:-biology}"
