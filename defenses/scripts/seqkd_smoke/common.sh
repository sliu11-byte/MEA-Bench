#!/usr/bin/env bash
set -euo pipefail

resolve_repo_root() {
  local start="${1:-$PWD}"
  local dir
  dir="$(cd "$start" && pwd)"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/attacks/scripts/run_attack.py" && -d "$dir/defenses" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  echo "Could not locate A-Benchmark-for-Model-distillation-survey from $start" >&2
  return 2
}

REPO_ROOT="${REPO_ROOT:-$(resolve_repo_root "$(dirname "${BASH_SOURCE[0]}")")}"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}"
export PROXY_MODEL="${PROXY_MODEL:-$STUDENT_MODEL}"
export BUDGET="${BUDGET:-100}"
export QUERY_POOL="${QUERY_POOL:-auto}"
DEFAULT_STORAGE_ROOT="${STORAGE_ROOT:-$REPO_ROOT}"
if [[ -z "${STORAGE_ROOT:-}" && -d "/path/to/storage/$USER" ]]; then
  DEFAULT_STORAGE_ROOT="/path/to/storage/$USER/A-Benchmark-for-Model-distillation-survey"
fi
export STORAGE_ROOT="${STORAGE_ROOT:-$DEFAULT_STORAGE_ROOT}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$STORAGE_ROOT/outputs/defenses/seqkd_smoke_online}"
DEFAULT_LOG_ROOT="$REPO_ROOT/logs/defenses/seqkd_smoke_online"
if [[ -d "/home/$USER/project/A-Benchmark-for-Model-distillation-survey" ]]; then
  DEFAULT_LOG_ROOT="/home/$USER/project/A-Benchmark-for-Model-distillation-survey/logs/defenses/seqkd_smoke_online"
fi
export LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.0}"
export TEACHER_TOP_P="${TEACHER_TOP_P:-1.0}"
export DETECTOR_MAX_QUERIES="${DETECTOR_MAX_QUERIES:-5}"
export DETECTOR_MAX_NEW_TOKENS="${DETECTOR_MAX_NEW_TOKENS:-8}"
export DETECTOR_TEMPERATURE="${DETECTOR_TEMPERATURE:-0.7}"
export DEVICE="${DEVICE:-cuda}"
export COMMON_ATTACK_EXTRA="${COMMON_ATTACK_EXTRA:-}"

mirror_defense_logs() {
  local method="$1"
  local out="$2"
  local dest="${LOG_ROOT}/${method}"
  mkdir -p "$dest"
  python3 - "$out" "$dest" <<'PY'
from pathlib import Path
import shutil
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
patterns = (
    "*.log",
    "*.out",
    "*.err",
    "*manifest.json",
    "*report*.json",
    "*metrics*.json",
    "*summary*.json",
)
if not src.exists():
    raise SystemExit(0)
seen = set()
for pattern in patterns:
    for path in src.rglob(pattern):
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
print(f"[logs] mirrored {len(seen)} light log/manifest files -> {dst}")
PY
}

run_seqkd_defense_runner() {
  local method="$1"
  shift
  local out="${DEFENSE_OUT:-$OUTPUT_ROOT/$method}"
  mkdir -p "$out" "$LOG_ROOT/$method"
  set +e
  python3 -m "defenses.${method}.runner" \
    --attack seqkd \
    --budget "$BUDGET" \
    --query-pool "$QUERY_POOL" \
    --teacher-model "$TEACHER_MODEL" \
    --student-model "$STUDENT_MODEL" \
    --teacher-max-tokens "$MAX_NEW_TOKENS" \
    --teacher-temperature "$TEACHER_TEMPERATURE" \
    --teacher-top-p "$TEACHER_TOP_P" \
    --detector-max-queries "$DETECTOR_MAX_QUERIES" \
    --detector-max-new-tokens "$DETECTOR_MAX_NEW_TOKENS" \
    --detector-temperature "$DETECTOR_TEMPERATURE" \
    --device "$DEVICE" \
    --output-dir "$out" \
    "$@" \
    ${COMMON_ATTACK_EXTRA:+-- $COMMON_ATTACK_EXTRA}
  local status=$?
  set -e
  mirror_defense_logs "$method" "$out"
  return "$status"
}
