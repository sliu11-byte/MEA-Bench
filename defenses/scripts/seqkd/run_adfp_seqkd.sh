#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ -z "${DEFENSE_CONFIG:-}" ]]; then
  DEFENSE_CONFIG="$(printf '{"gamma":0.5,"window_size":2,"strength_lambda":140.0,"batch_size":%s}' "$ADFP_BATCH_SIZE")"
fi
run_seqkd_defense_runner adfp \
  --proxy-model "${ADFP_PROXY_MODEL:-$PROXY_MODEL}" \
  --adfp-batch-size "$ADFP_BATCH_SIZE" \
  --defense-config "$DEFENSE_CONFIG"
