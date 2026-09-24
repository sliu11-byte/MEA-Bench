#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ -z "${DEFENSE_CONFIG:-}" ]]; then
  DEFENSE_CONFIG="$(printf '{"gamma":0.25,"delta":2.0,"max_new_tokens":%s,"temperature":0.7,"top_p":0.95}' "$MAX_NEW_TOKENS")"
fi
run_seqkd_defense_runner ginsew --defense-config "$DEFENSE_CONFIG"
