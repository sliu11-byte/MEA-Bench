#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ -z "${DEFENSE_CONFIG:-}" ]]; then
  DEFENSE_CONFIG="$(printf '{"gamma":0.5,"delta":1.0,"max_new_tokens":%s,"temperature":0.8,"top_p":0.95}' "$MAX_NEW_TOKENS")"
fi
run_seqkd_defense_runner radioactivity --defense-config "$DEFENSE_CONFIG"
