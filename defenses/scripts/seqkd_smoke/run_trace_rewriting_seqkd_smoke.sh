#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
REWRITER_MODEL="${REWRITER_MODEL:-$TEACHER_MODEL}"
if [[ -z "${DEFENSE_CONFIG:-}" ]]; then
  DEFENSE_CONFIG="$(printf '{"rewrite_strategy":"optimized_prompt_local","teacher_max_new_tokens":%s,"max_new_tokens":%s,"temperature":0.6,"top_p":0.95}' "$MAX_NEW_TOKENS" "$MAX_NEW_TOKENS")"
fi
run_seqkd_defense_runner trace_rewriting \
  --rewriter-model "$REWRITER_MODEL" \
  --defense-config "$DEFENSE_CONFIG" \
  --no-detector
