#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
METHODS="${METHODS:-ginsew radioactivity adfp ads trace_rewriting doge duffin query_traffic}"
for method in $METHODS; do
  echo "[seqkd smoke] running $method"
  bash "$REPO_ROOT/defenses/scripts/seqkd_smoke/run_${method}_seqkd_smoke.sh"
done
