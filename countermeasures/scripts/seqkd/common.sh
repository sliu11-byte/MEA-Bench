#!/usr/bin/env bash
set -euo pipefail

resolve_repo_root() {
  local start="${1:-$PWD}"
  local dir
  dir="$(cd "$start" && pwd)"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/attacks/scripts/run_attack.py" && -d "$dir/defenses" && -d "$dir/countermeasures" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  echo "Could not locate A-Benchmark-for-Model-distillation-survey from $start" >&2
  return 2
}

REPO_ROOT="${REPO_ROOT:-$(resolve_repo_root "$(dirname "${BASH_SOURCE[0]}")")}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
cd "$REPO_ROOT"

export COUNTERMEASURE="${COUNTERMEASURE:-dipper}"
case "$COUNTERMEASURE" in
  dipper|translation) ;;
  none) echo "countermeasure scripts require COUNTERMEASURE=dipper or translation" >&2; exit 2 ;;
  *) echo "Unknown COUNTERMEASURE=$COUNTERMEASURE" >&2; exit 2 ;;
esac

export BUDGET="${BUDGET:-1000}"
DEFAULT_STORAGE_ROOT="${STORAGE_ROOT:-$REPO_ROOT}"
if [[ -z "${STORAGE_ROOT:-}" && -d "/path/to/storage/$USER" ]]; then
  DEFAULT_STORAGE_ROOT="/path/to/storage/$USER/A-Benchmark-for-Model-distillation-survey"
fi
export STORAGE_ROOT="${STORAGE_ROOT:-$DEFAULT_STORAGE_ROOT}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/countermeasures/seqkd_b${BUDGET}/${COUNTERMEASURE}}"
DEFAULT_LOG_ROOT="$REPO_ROOT/logs/countermeasures/seqkd_b${BUDGET}/${COUNTERMEASURE}"
if [[ -d "/home/$USER/project/A-Benchmark-for-Model-distillation-survey" ]]; then
  DEFAULT_LOG_ROOT="/home/$USER/project/A-Benchmark-for-Model-distillation-survey/logs/countermeasures/seqkd_b${BUDGET}/${COUNTERMEASURE}"
fi
export LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"

export COUNTERMEASURE_LEX="${COUNTERMEASURE_LEX:-40}"
export COUNTERMEASURE_ORDER="${COUNTERMEASURE_ORDER:-0}"
export COUNTERMEASURE_SENT_INTERVAL="${COUNTERMEASURE_SENT_INTERVAL:-3}"
export COUNTERMEASURE_WITH_CONTEXT="${COUNTERMEASURE_WITH_CONTEXT:-0}"
export COUNTERMEASURE_MODEL_NAME="${COUNTERMEASURE_MODEL_NAME:-}"
export COUNTERMEASURE_TOKENIZER_NAME="${COUNTERMEASURE_TOKENIZER_NAME:-}"
export COUNTERMEASURE_DEVICE="${COUNTERMEASURE_DEVICE:-}"

source "$REPO_ROOT/defenses/scripts/seqkd/common.sh"

run_countermeasure_seqkd_defense_runner() {
  local method="$1"
  shift
  echo "[countermeasure] method=$COUNTERMEASURE defense=$method attack=seqkd budget=$BUDGET"
  run_seqkd_defense_runner "$method" "$@"
}

run_countermeasure_seqkd_script() {
  local method="$1"
  shift || true
  echo "[countermeasure] method=$COUNTERMEASURE defense=$method attack=seqkd budget=$BUDGET"
  bash "$REPO_ROOT/defenses/scripts/seqkd/run_${method}_seqkd.sh" "$@"
}
