#!/usr/bin/env bash
# Manifest-oriented front end for the existing evaluation drivers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

usage() {
  cat <<'EOF'
Usage: bash runs/run_evaluation.sh --manifest PATH [options]

Options:
  --metrics all            Reserved metric selection (default: all)
  --limit N                Smoke-test example limit
  --output-root PATH       Evaluation artifact root (default: outputs/evaluation)
  --storage-root PATH      Current root used to remap recorded outputs/ paths
  --dry-run                Validate and print the selected evaluation driver
  -h, --help               Show this help
EOF
}

MANIFEST=""
METRICS="all"
LIMIT=""
OUTPUT_ROOT="outputs/evaluation"
STORAGE_ROOT_VALUE=""
PROMPTS=""
TEACHER=""
DRY_RUN=0

while (($#)); do
  case "$1" in
    --manifest) MANIFEST="${2:?--manifest requires a value}"; shift 2 ;;
    --metrics) METRICS="${2:?--metrics requires a value}"; shift 2 ;;
    --limit) LIMIT="${2:?--limit requires a value}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?--output-root requires a value}"; shift 2 ;;
    --storage-root) STORAGE_ROOT_VALUE="${2:?--storage-root requires a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${MANIFEST}" ]] || { echo "--manifest is required" >&2; exit 2; }
[[ -f "${MANIFEST}" ]] || { echo "Manifest does not exist: ${MANIFEST}" >&2; exit 2; }
[[ "${METRICS}" == "all" ]] || {
  echo "The existing consolidated evaluators currently accept --metrics all only." >&2; exit 2;
}

export REPO_DIR
export STORAGE_ROOT="${STORAGE_ROOT_VALUE:-${STORAGE_ROOT:-${REPO_DIR}}}"
RESOLVED_JSON="$(mktemp)"
trap 'rm -f "${RESOLVED_JSON}"' EXIT
python3 -m evaluation.scripts.resolve_local_run \
  --manifest "${MANIFEST}" --storage-root "${STORAGE_ROOT}" > "${RESOLVED_JSON}"
readarray -t RUN_INFO < <(python3 - "${RESOLVED_JSON}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("kind", "attack", "budget", "run_id", "defense", "checkpoint", "report", "manifest"):
    print(payload.get(key) or "")
PY
)
RUN_TYPE="${RUN_INFO[0]}"
ATTACK="${RUN_INFO[1]}"
BUDGET="${RUN_INFO[2]}"
RUN_ID="${RUN_INFO[3]}"
DEFENSE="${RUN_INFO[4]}"
CHECKPOINT="${RUN_INFO[5]}"
REPORT="${RUN_INFO[6]}"
MANIFEST_ABS="${RUN_INFO[7]}"
[[ -n "${ATTACK}" && -n "${BUDGET}" ]] || { echo "Manifest lacks attack or budget metadata" >&2; exit 2; }

case "${RUN_TYPE}" in
  attack)
    case "${BUDGET}" in
      100) DRIVER="runs/evaluation/evaluate_attack_b100.sh" ;;
      1000) DRIVER="runs/evaluation/evaluate_attack_b1000.sh" ;;
      10000) DRIVER="runs/evaluation/evaluate_attack_b10000.sh" ;;
      *) echo "No attack evaluator is configured for budget ${BUDGET}" >&2; exit 2 ;;
    esac
    ;;
  defense) DRIVER="evaluation.defense_eval.evaluate" ;;
  adaptive_attack) DRIVER="evaluation.defense_eval.evaluate_counter_seqkd" ;;
  detector) DRIVER="local_detector_report" ;;
  *) echo "Unsupported manifest run_type: ${RUN_TYPE}" >&2; exit 2 ;;
esac

python3 - "${RUN_TYPE}" "${ATTACK}" "${BUDGET}" "${RUN_ID}" "${DRIVER}" "${OUTPUT_ROOT}" "${LIMIT}" "${DEFENSE}" <<'PY'
import json
import sys
run_type, attack, budget, run_id, driver, output_root, limit, defense = sys.argv[1:]
print(json.dumps({
    "run_type": "evaluation",
    "status": "planned",
    "source_run_type": run_type,
    "source_run_id": run_id,
    "attack": attack,
    "budget": int(budget),
    "defense": defense or None,
    "driver": driver,
    "output_root": output_root,
    "limit": int(limit) if limit else None,
}, indent=2, sort_keys=True))
PY

if ((DRY_RUN)); then
  exit 0
fi

OUTPUT_ROOT_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUTPUT_ROOT}")"
[[ -n "${LIMIT}" ]] && export EVAL_LIMIT="${LIMIT}"
EVAL_RUN_KEY="${RUN_ID}"
[[ -n "${LIMIT}" ]] && EVAL_RUN_KEY="${RUN_ID}_smoke_${LIMIT}"

PROMPTS="${PROMPTS:-${REPO_DIR}/evaluation/data/heldout/heldout_prompts.jsonl}"
if [[ ! -s "${PROMPTS}" ]]; then
  echo "Held-out prompts are missing; building the paper evaluation split automatically."
  python3 -m evaluation.scripts.build_heldout_queries \
    --output-dir "$(dirname "${PROMPTS}")" \
    --prompts-output "${PROMPTS}"
fi
export HELDOUT_PROMPTS_JSONL="${PROMPTS}"

if [[ "${RUN_TYPE}" == "attack" || "${RUN_TYPE}" == "adaptive_attack" ]]; then
  TEACHER="${TEACHER:-${STORAGE_ROOT}/outputs/heldout_queries/heldout_teacher_outputs.jsonl}"
  TEACHER_MANIFEST="${TEACHER%.jsonl}.manifest.json"
  REFERENCE_VALID=0
  if [[ -s "${TEACHER}" && -s "${TEACHER_MANIFEST}" ]]; then
    if python3 - "${TEACHER_MANIFEST}" "${PROMPTS}" "${LIMIT}" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], encoding="utf-8") as handle:
    available = sum(1 for line in handle if line.strip())
required = int(sys.argv[3]) if sys.argv[3] else available
if payload.get("teacher_model") != "meta-llama/Llama-3.3-70B-Instruct":
    raise SystemExit(1)
if int(payload.get("completed_count") or 0) < required:
    raise SystemExit(1)
PY
    then
      REFERENCE_VALID=1
    fi
  fi
  if ((REFERENCE_VALID == 0)); then
    echo "Held-out teacher outputs are missing or incomplete; generating them automatically."
    TEACHER_ARGS=(
      --prompts-jsonl "${PROMPTS}"
      --output-jsonl "${TEACHER}"
      --teacher-model meta-llama/Llama-3.3-70B-Instruct
      --backend local_hf
      --mode chat
      --torch-dtype bfloat16
      --device-map auto
      --max-tokens 1536
    )
    [[ -n "${LIMIT}" ]] && TEACHER_ARGS+=(--limit "${LIMIT}")
    python3 -m evaluation.scripts.build_heldout_teacher_outputs "${TEACHER_ARGS[@]}"
  fi
  export HELDOUT_TEACHER_OUTPUTS_JSONL="${TEACHER}"
  export HELDOUT_TEACHER_MANIFEST="${TEACHER_MANIFEST}"
fi

case "${RUN_TYPE}" in
  attack)
    # Scope discovery to exactly the selected run. Historical roots may contain
    # several completed runs for the same method and must not leak into this job.
    ATTACK_ROOT="$(dirname "${MANIFEST_ABS}")"
    export EVAL_ATTACK="${ATTACK}"
    case "${BUDGET}" in
      100)
        export ATTACK_ROOT
        export ATTACK_EVAL_OUTPUT_ROOT="${OUTPUT_ROOT_ABS}/attack/${EVAL_RUN_KEY}"
        ;;
      1000)
        export ATTACK_B1000_ROOT="${ATTACK_ROOT}"
        export ATTACK_B1000_EVAL_OUTPUT_ROOT="${OUTPUT_ROOT_ABS}/attack/${EVAL_RUN_KEY}"
        ;;
      10000)
        export ATTACK_B10000_ROOT="${ATTACK_ROOT}"
        export ATTACK_B10000_EVAL_OUTPUT_ROOT="${OUTPUT_ROOT_ABS}/attack/${EVAL_RUN_KEY}"
        ;;
    esac
    rm -f "${RESOLVED_JSON}"
    trap - EXIT
    exec bash "${DRIVER}"
    ;;
  defense)
    case "${ATTACK}" in seqkd|soda|qedks) ;; *)
      echo "The existing defense evaluator supports seqkd, soda, and qedks only; got ${ATTACK}." >&2; exit 2;;
    esac
    [[ -n "${DEFENSE}" && -n "${CHECKPOINT}" ]] || {
      echo "Defense manifest does not resolve a defense name and local checkpoint." >&2; exit 2;
    }
    EVAL_DIR="${OUTPUT_ROOT_ABS}/defense/${EVAL_RUN_KEY}"
    REFERENCE=(python3 -m evaluation.defense_eval.evaluate --stage reference
      --output-root "${EVAL_DIR}" --prompts-jsonl "${PROMPTS}")
    [[ -n "${LIMIT}" ]] && REFERENCE+=(--limit "${LIMIT}")
    if [[ ! -s "${EVAL_DIR}/reference/complete.json" ]]; then
      "${REFERENCE[@]}"
    fi
    COMMAND=(python3 -m evaluation.defense_eval.evaluate --stage students --attack "${ATTACK}"
      --defense "${DEFENSE}" --checkpoint "${CHECKPOINT}"
      --output-root "${EVAL_DIR}" --prompts-jsonl "${PROMPTS}")
    [[ -n "${LIMIT}" ]] && COMMAND+=(--limit "${LIMIT}")
    rm -f "${RESOLVED_JSON}"
    trap - EXIT
    exec "${COMMAND[@]}"
    ;;
  adaptive_attack)
    case "${ATTACK}" in seqkd|qedks) ;; *)
      echo "The existing adaptive evaluator supports seqkd and qedks only; got ${ATTACK}." >&2; exit 2;;
    esac
    PROMPTS="${PROMPTS:-${REPO_DIR}/evaluation/data/heldout/heldout_prompts.jsonl}"
    COMMAND=(python3 -m evaluation.defense_eval.evaluate_counter_seqkd
      --storage-root "${STORAGE_ROOT}" --attack "${ATTACK}"
      --output-root "${OUTPUT_ROOT_ABS}/adaptive/${EVAL_RUN_KEY}"
      --prompts-jsonl "${PROMPTS}" --teacher-jsonl "${TEACHER}"
      --teacher-manifest "${TEACHER_MANIFEST}" --manifest "${MANIFEST_ABS}")
    [[ -n "${LIMIT}" ]] && COMMAND+=(--limit "${LIMIT}")
    rm -f "${RESOLVED_JSON}"
    trap - EXIT
    exec "${COMMAND[@]}"
    ;;
  detector)
    [[ -n "${REPORT}" ]] || { echo "Detector manifest does not resolve a local report." >&2; exit 2; }
    DETECTOR_OUT="${OUTPUT_ROOT_ABS}/detector/${EVAL_RUN_KEY}"
    python3 - "${MANIFEST_ABS}" "${REPORT}" "${DETECTOR_OUT}" <<'PY'
import json
import sys
from pathlib import Path

manifest, report, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
payload = json.loads(report.read_text(encoding="utf-8"))
(output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
(output / "complete.json").write_text(json.dumps({
    "complete": True,
    "source_manifest": str(manifest.resolve()),
    "source_report": str(report.resolve()),
}, indent=2) + "\n", encoding="utf-8")
print(output / "summary.json")
PY
    ;;
esac
