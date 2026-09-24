#!/usr/bin/env bash
set -euo pipefail
METHODS="${METHODS:-ginsew radioactivity adfp trace_rewriting}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for method in $METHODS; do
  bash "$SCRIPT_DIR/run_${method}_seqkd.sh"
done
