#!/usr/bin/env bash

set -euo pipefail
source "${REPO_DIR:-$(pwd)}/runs/evaluation/defense_common.sh"
ATTACK="${ATTACK:-${1:-}}"
case "${ATTACK}" in
  seqkd|soda|qedks) ;;
  *) echo 'Usage: ATTACK=seqkd|soda|qedks bash runs/evaluation/evaluate_defense_students.sh' >&2; exit 2 ;;
esac
python -m evaluation.defense_eval.evaluate --stage students --attack "${ATTACK}" "${ARGS[@]}"
