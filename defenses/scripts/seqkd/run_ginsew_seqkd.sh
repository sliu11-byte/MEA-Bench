#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ -z "${DEFENSE_CONFIG:-}" ]]; then
  DEFENSE_CONFIG='{"fraction":0.5,"strength":2.0,"freq":16,"eps":0.2}'
fi
run_seqkd_defense_runner ginsew --defense-config "$DEFENSE_CONFIG"
