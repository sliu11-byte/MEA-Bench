#!/usr/bin/env bash
# CPU-side data preparation for QEDKS, Model Leeching, or SODA.

set -euo pipefail

SUBMIT_DIR="${REPO_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey"
else
  echo "Submit from the repository root." >&2
  exit 2
fi
cd "${REPO_DIR}"
mkdir -p logs

: "${ATTACK:?Set ATTACK to qedks, model_leeching, or soda}"
case "${ATTACK}" in
  qedks|model_leeching|soda) ;;
  *) echo "Unsupported staged preparation attack: ${ATTACK}" >&2; exit 2 ;;
esac

export ATTACKS="${ATTACK}"
export ATTACK_INDEX=0
export ATTACK_EXECUTION_STAGE=prepare
export ATTACK_CUDA_VISIBLE_DEVICES=""
export ONLINE_ATTACK_TEACHER_MODE=external
export SODA_STUDENT_MODE=external
export ATTACK_QUERY_CONCURRENCY="${ATTACK_QUERY_CONCURRENCY:-8}"

if [[ "${ATTACK}" == "soda" ]]; then
  ENDPOINT_ENV="${STUDENT_VLLM_ENDPOINT_ENV_PATH:-${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}/outputs/vllm_student/student_endpoint.env}"
else
  ENDPOINT_ENV="${VLLM_ENDPOINT_ENV_PATH:-${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}/outputs/vllm_teacher/teacher_endpoint.env}"
fi
if [[ ! -s "${ENDPOINT_ENV}" ]]; then
  echo "Endpoint environment file is missing: ${ENDPOINT_ENV}" >&2
  exit 1
fi
set -a
source "${ENDPOINT_ENV}"
set +a

bash attacks/scripts/run_attacks.sh

STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
OUTPUT_DIR="${OUTPUT_DIR:-${STORAGE_ROOT}/outputs/attacks_full}"
BUDGET_VALUE="${BUDGET:-1000}"
case "${ATTACK}" in
  qedks) BUDGET_VALUE="${QEDKS_BUDGET:-${BUDGET_VALUE}}" ;;
  model_leeching) BUDGET_VALUE="${MODEL_LEECHING_BUDGET:-${BUDGET_VALUE}}" ;;
  soda) BUDGET_VALUE="${SODA_BUDGET:-${BUDGET_VALUE}}" ;;
esac
POINTER_DIR="${STORAGE_ROOT}/outputs/staged_attacks"
POINTER_PATH="${STAGED_RUN_POINTER:-${POINTER_DIR}/${ATTACK}_b${BUDGET_VALUE}_latest.txt}"
mkdir -p "${POINTER_DIR}"
python3 - "${OUTPUT_DIR}" "${ATTACK}" "${BUDGET_VALUE}" "${POINTER_PATH}" <<'PY'
import json
import sys
from pathlib import Path

output_dir, attack, budget, pointer = sys.argv[1:]
candidates = []
for manifest in (Path(output_dir) / attack).glob("*/prepare_manifest.json"):
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("result", {}).get("status") == "prepared" and int(payload.get("result", {}).get("budget", -1)) == int(budget):
        candidates.append(manifest)
if not candidates:
    raise SystemExit(f"No prepared {attack} run found for budget {budget}")
run_dir = max(candidates, key=lambda path: path.stat().st_mtime).parent.resolve()
Path(pointer).write_text(str(run_dir) + "\n", encoding="utf-8")
print(f"Prepared run: {run_dir}")
print(f"Prepared-run pointer: {pointer}")
PY
