#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ -z "${DEFENSE_CONFIG:-}" ]]; then
  DEFENSE_CONFIG='{"method":"maryland","ngram":4,"seed":0,"seeding":"hash","hash_key":35317,"gamma":0.25,"delta":2.0,"scoring_method":"v2"}'
fi
run_seqkd_defense_runner radioactivity --defense-config "$DEFENSE_CONFIG"
