#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_env TEACHER_BASE_URL
export REWRITER_BACKEND="${REWRITER_BACKEND:-openai_compatible}"
export REWRITER_BASE_URL="${REWRITER_BASE_URL:-$TEACHER_BASE_URL}"
export REWRITER_REQUEST_MODEL="${REWRITER_REQUEST_MODEL:-${REWRITER_MODEL:-$TEACHER_MODEL}}"
if [[ -z "${DEFENSE_CONFIG:-}" ]]; then
  DEFENSE_CONFIG="$(printf '{"rewrite_strategy":"optimized_prompt_official","teacher_max_new_tokens":%s,"max_new_tokens":%s,"temperature":0.6,"top_p":0.95}' "$MAX_NEW_TOKENS" "$MAX_NEW_TOKENS")"
fi
run_seqkd_defense_runner trace_rewriting \
  --teacher-base-url "$TEACHER_BASE_URL" \
  --teacher-request-model "${TEACHER_REQUEST_MODEL:-$TEACHER_MODEL}" \
  --rewriter-backend "$REWRITER_BACKEND" \
  --rewriter-base-url "$REWRITER_BASE_URL" \
  --rewriter-request-model "$REWRITER_REQUEST_MODEL" \
  --rewriter-api-key "${REWRITER_API_KEY:-EMPTY}" \
  --rewriter-model "${REWRITER_MODEL:-$TEACHER_MODEL}" \
  --defense-config "$DEFENSE_CONFIG" \
  --no-detector
