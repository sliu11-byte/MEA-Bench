#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHODS="${METHODS:-ginsew radioactivity adfp}"
COUNTERMEASURES="${COUNTERMEASURES:-${COUNTERMEASURE:-dipper translation}}"

for countermeasure in $COUNTERMEASURES; do
  export COUNTERMEASURE="$countermeasure"
  for method in $METHODS; do
    bash "$SCRIPT_DIR/run_${method}_seqkd.sh"
  done
done
