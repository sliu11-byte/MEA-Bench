#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
export TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
if [[ -z "${DEFENSE_CONFIG:-}" ]]; then
  DEFENSE_CONFIG="$(printf '{"gamma":0.5,"window_size":2,"strength_lambda":20.0,"temperature":0.7,"top_p":0.95,"max_new_tokens":%s}' "$MAX_NEW_TOKENS")"
fi
run_seqkd_defense_runner adfp \
  --proxy-model "${ADFP_PROXY_MODEL:-$PROXY_MODEL}" \
  --defense-config "$DEFENSE_CONFIG"
