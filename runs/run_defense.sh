#!/usr/bin/env bash
# Portable front end for the original defense orchestrators.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash runs/run_defense.sh --defense NAME --attack NAME --budget N [options]
  bash runs/run_defense.sh --defense duffin|mmd|prada|seat --attack-run MANIFEST [options]

Options:
  --profile NAME           Benchmark profile (default: paper)
  --query-pool PATH|auto   Query pool for defended extraction (default: auto)
  --query-ordering PATH    Optional deterministic ordering file
  --teacher-endpoint URL   OpenAI-compatible teacher endpoint
  --student-endpoint URL   OpenAI-compatible student endpoint
  --output-root PATH       Artifact root (default: outputs)
  --execution-mode MODE    auto, offline_batch, or online
  --dry-run                Validate and print the dispatch plan
  -h, --help               Show this help
EOF
}

DEFENSE=""
ATTACK=""
BUDGET=""
ATTACK_RUN=""
PROFILE="paper"
QUERY_POOL="auto"
QUERY_ORDERING="auto"
TEACHER_ENDPOINT=""
STUDENT_ENDPOINT=""
OUTPUT_ROOT="outputs"
EXECUTION_MODE="auto"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --defense) DEFENSE="${2:?--defense requires a value}"; shift 2 ;;
    --attack) ATTACK="${2:?--attack requires a value}"; shift 2 ;;
    --budget) BUDGET="${2:?--budget requires a value}"; shift 2 ;;
    --attack-run) ATTACK_RUN="${2:?--attack-run requires a value}"; shift 2 ;;
    --profile) PROFILE="${2:?--profile requires a value}"; shift 2 ;;
    --query-pool) QUERY_POOL="${2:?--query-pool requires a value}"; shift 2 ;;
    --query-ordering) QUERY_ORDERING="${2:?--query-ordering requires a value}"; shift 2 ;;
    --teacher-endpoint) TEACHER_ENDPOINT="${2:?--teacher-endpoint requires a value}"; shift 2 ;;
    --student-endpoint) STUDENT_ENDPOINT="${2:?--student-endpoint requires a value}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?--output-root requires a value}"; shift 2 ;;
    --execution-mode) EXECUTION_MODE="${2:?--execution-mode requires a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${DEFENSE}" ]] || { echo "--defense is required" >&2; exit 2; }

if [[ -n "${ATTACK_RUN}" ]]; then
  [[ -f "${ATTACK_RUN}" ]] || { echo "Attack manifest does not exist: ${ATTACK_RUN}" >&2; exit 2; }
  readarray -t INFERRED < <(python3 - "${ATTACK_RUN}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
config = payload.get("run_config") if isinstance(payload.get("run_config"), dict) else payload.get("config", {})
print(result.get("attack") or payload.get("attack") or config.get("attack") or "")
print(result.get("budget") or payload.get("budget") or config.get("budget") or "")
PY
  )
  [[ -z "${ATTACK}" || "${ATTACK}" == "${INFERRED[0]}" ]] || {
    echo "--attack conflicts with attack manifest (${ATTACK} != ${INFERRED[0]})" >&2; exit 2;
  }
  [[ -z "${BUDGET}" || "${BUDGET}" == "${INFERRED[1]}" ]] || {
    echo "--budget conflicts with attack manifest (${BUDGET} != ${INFERRED[1]})" >&2; exit 2;
  }
  ATTACK="${INFERRED[0]}"
  BUDGET="${INFERRED[1]}"
fi

python3 - "${DEFENSE}" "${ATTACK}" "${BUDGET}" "${PROFILE}" "${ATTACK_RUN}" "${OUTPUT_ROOT}" "${EXECUTION_MODE}" <<'PY'
import json
import sys
from benchmark.profiles import load_profile
from benchmark.registry import attack_spec, defense_spec

defense, attack, budget_text, profile_name, attack_run, output_root, mode = sys.argv[1:]
profile = load_profile(profile_name)
spec = defense_spec(defense)
if not attack:
    raise SystemExit("an attack must be supplied directly or inferred from --attack-run")
attack_spec(attack)
if not budget_text:
    raise SystemExit("a budget must be supplied directly or inferred from --attack-run")
budget = int(budget_text)
if mode not in {"auto", "offline_batch", "online"}:
    raise SystemExit(f"unknown execution mode: {mode}")
if spec.category == "defended_extraction":
    allowed = tuple(int(v) for v in profile.payload["budgets"]["defended_extraction"])
    if budget not in allowed:
        raise SystemExit(f"defended extraction budget {budget} is not in profile {profile_name}: {allowed}")
    if attack_run:
        raise SystemExit("--attack-run is for result-based defenses; defended extraction runs the attack itself")
elif not attack_run:
    raise SystemExit(f"{defense} requires --attack-run")
plan = {
    "run_type": "defense",
    "status": "planned",
    "category": spec.category,
    "defense": defense,
    "attack": attack,
    "budget": budget,
    "profile": profile_name,
    "attack_manifest": attack_run or None,
    "output_root": output_root,
    "execution_mode": mode,
    "teacher_model": profile.section("defense")["teacher_model"],
    "student_model": profile.section("defense")["student_model"],
    "dispatcher": "runs/defense/common.sh" if spec.category == "defended_extraction" else "runs/defense/detector_common.sh",
}
print(json.dumps(plan, indent=2, sort_keys=True))
PY

if ((DRY_RUN)); then
  exit 0
fi

export REPO_DIR
export STORAGE_ROOT="${STORAGE_ROOT:-${REPO_DIR}}"
export METHOD="${DEFENSE}"
export ATTACK
export BUDGET
export QUERY_POOL
export QUERY_ORDERING
export DEFENSE_EXECUTION_MODE="${EXECUTION_MODE}"
readarray -t DEFENSE_MODELS < <(python3 - "${PROFILE}" <<'PY'
import sys
from benchmark.profiles import load_profile
section = load_profile(sys.argv[1]).section("defense")
print(section["teacher_model"])
print(section["student_model"])
PY
)
export TEACHER_MODEL="${DEFENSE_MODELS[0]}"
export STUDENT_MODEL="${DEFENSE_MODELS[1]}"
export TEACHER_REQUEST_MODEL="${TEACHER_REQUEST_MODEL:-${TEACHER_MODEL}}"
export STUDENT_REQUEST_MODEL="${STUDENT_REQUEST_MODEL:-${STUDENT_MODEL}}"
export TEACHER_ENDPOINT_URL="${TEACHER_ENDPOINT:-${TEACHER_ENDPOINT_URL:-http://127.0.0.1:8000/v1}}"
export STUDENT_ENDPOINT_URL="${STUDENT_ENDPOINT:-${STUDENT_ENDPOINT_URL:-http://127.0.0.1:8001/v1}}"

OUTPUT_ROOT_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUTPUT_ROOT}")"
case "${DEFENSE}" in
  ads|doge|trace_rewriting|adfp|ginsew|radioactivity)
    export OUTPUT_ROOT="${OUTPUT_ROOT_ABS}/defenses"
    export DEFENSE_OUT="${OUTPUT_ROOT_ABS}/defenses/${DEFENSE}/${ATTACK}/b${BUDGET}"
    export LOG_ROOT="${DEFENSE_OUT}/logs"
    source runs/defense/common.sh
    ;;
  duffin)
    export METHOD=duffin
    export ATTACK_MANIFEST="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${ATTACK_RUN}")"
    export OUTPUT_ROOT="${OUTPUT_ROOT_ABS}/defenses"
    export DETECTOR_OUT="${OUTPUT_ROOT_ABS}/defenses/duffin/${ATTACK}/b${BUDGET}"
    export LOG_ROOT="${DETECTOR_OUT}/logs"
    source runs/defense/detector_common.sh
    ;;
  mmd|prada|seat)
    export METHOD=query_traffic
    export QT_DETECTORS="${DEFENSE}"
    export ATTACK_MANIFEST="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${ATTACK_RUN}")"
    export OUTPUT_ROOT="${OUTPUT_ROOT_ABS}/defenses"
    export DETECTOR_OUT="${OUTPUT_ROOT_ABS}/defenses/${DEFENSE}/${ATTACK}/b${BUDGET}"
    export LOG_ROOT="${DETECTOR_OUT}/logs"
    source runs/defense/detector_common.sh
    ;;
esac
