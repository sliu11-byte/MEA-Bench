#!/usr/bin/env bash
# Portable front end for the original countermeasure orchestrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

usage() {
  cat <<'EOF'
Usage: bash runs/run_adaptive_attack.sh --adaptive-attack NAME --defense NAME --attack NAME --budget N [options]

Required:
  --adaptive-attack NAME   dipper or translation
  --defense NAME           adfp, ginsew, or radioactivity
  --attack NAME            One of the six benchmark attacks
  --budget N               1000 under --profile paper

Options:
  --profile NAME           Benchmark profile (default: paper)
  --query-pool PATH|auto   Query pool (default: auto)
  --query-ordering PATH    Optional deterministic ordering file
  --teacher-endpoint URL   OpenAI-compatible teacher endpoint
  --student-endpoint URL   OpenAI-compatible student endpoint
  --output-root PATH       Artifact root (default: outputs)
  --reuse-baselines        Require previously prepared clean/defense baselines
  --dry-run                Validate and print the dispatch plan
  -h, --help               Show this help
EOF
}

ADAPTIVE_ATTACK=""
DEFENSE=""
ATTACK=""
BUDGET=""
PROFILE="paper"
QUERY_POOL="auto"
QUERY_ORDERING="auto"
TEACHER_ENDPOINT=""
STUDENT_ENDPOINT=""
OUTPUT_ROOT="outputs"
REUSE_BASELINES=0
DRY_RUN=0

while (($#)); do
  case "$1" in
    --adaptive-attack) ADAPTIVE_ATTACK="${2:?--adaptive-attack requires a value}"; shift 2 ;;
    --defense) DEFENSE="${2:?--defense requires a value}"; shift 2 ;;
    --attack) ATTACK="${2:?--attack requires a value}"; shift 2 ;;
    --budget) BUDGET="${2:?--budget requires a value}"; shift 2 ;;
    --profile) PROFILE="${2:?--profile requires a value}"; shift 2 ;;
    --query-pool) QUERY_POOL="${2:?--query-pool requires a value}"; shift 2 ;;
    --query-ordering) QUERY_ORDERING="${2:?--query-ordering requires a value}"; shift 2 ;;
    --teacher-endpoint) TEACHER_ENDPOINT="${2:?--teacher-endpoint requires a value}"; shift 2 ;;
    --student-endpoint) STUDENT_ENDPOINT="${2:?--student-endpoint requires a value}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?--output-root requires a value}"; shift 2 ;;
    --reuse-baselines) REUSE_BASELINES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${ADAPTIVE_ATTACK}" ]] || { echo "--adaptive-attack is required" >&2; exit 2; }
[[ -n "${DEFENSE}" ]] || { echo "--defense is required" >&2; exit 2; }
[[ -n "${ATTACK}" ]] || { echo "--attack is required" >&2; exit 2; }
[[ -n "${BUDGET}" ]] || { echo "--budget is required" >&2; exit 2; }

python3 - "${ADAPTIVE_ATTACK}" "${DEFENSE}" "${ATTACK}" "${BUDGET}" "${PROFILE}" "${OUTPUT_ROOT}" "${REUSE_BASELINES}" <<'PY'
import json
import sys
from benchmark.profiles import load_profile
from benchmark.registry import validate_adaptive_combination

adaptive, defense, attack, budget_text, profile_name, output_root, reuse = sys.argv[1:]
profile = load_profile(profile_name)
budget = int(budget_text)
validate_adaptive_combination(adaptive, defense, attack, budget, profile.adaptive_budgets)
print(json.dumps({
    "run_type": "adaptive_attack",
    "status": "planned",
    "adaptive_attack": adaptive,
    "defense": defense,
    "attack": attack,
    "budget": budget,
    "profile": profile_name,
    "reuse_baselines": reuse == "1",
    "output_root": output_root,
    "dispatcher": "runs/counter/common.sh",
}, indent=2, sort_keys=True))
PY

if ((DRY_RUN)); then
  exit 0
fi

export REPO_DIR
export STORAGE_ROOT="${STORAGE_ROOT:-${REPO_DIR}}"
export COUNTERMEASURE="${ADAPTIVE_ATTACK}"
export METHOD="${DEFENSE}"
export ATTACK
export BUDGET
export QUERY_POOL
export QUERY_ORDERING
export TEACHER_ENDPOINT_URL="${TEACHER_ENDPOINT:-${TEACHER_ENDPOINT_URL:-http://127.0.0.1:8000/v1}}"
export STUDENT_ENDPOINT_URL="${STUDENT_ENDPOINT:-${STUDENT_ENDPOINT_URL:-http://127.0.0.1:8001/v1}}"
OUTPUT_ROOT_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUTPUT_ROOT}")"
export OUTPUT_ROOT="${OUTPUT_ROOT_ABS}/adaptive"
export RUN_OUT="${OUTPUT_ROOT_ABS}/adaptive/${ADAPTIVE_ATTACK}/${DEFENSE}/${ATTACK}/b${BUDGET}"
export LOG_ROOT="${RUN_OUT}/logs"
if ((REUSE_BASELINES)); then
  export COUNTER_BASELINES=1
  export PREPARE_COUNTER_BASELINES=0
  export REQUIRE_COUNTER_BASELINES=1
else
  export COUNTER_BASELINES=1
  export PREPARE_COUNTER_BASELINES=1
  export REQUIRE_COUNTER_BASELINES=0
fi

source runs/counter/common.sh
