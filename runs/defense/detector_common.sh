#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${METHOD:-}" || -z "${ATTACK:-}" || -z "${BUDGET:-}" ]]; then
  echo "METHOD, ATTACK, and BUDGET must be set before sourcing runs/defense/detector_common.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"
mkdir -p logs

if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi

export OMP_NUM_THREADS="${CPU_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

export STORAGE_ROOT="${STORAGE_ROOT:-${REPO_DIR}}"

FULL_RUN_ENV="${FULL_RUN_ENV:-}"
if [[ -n "${FULL_RUN_ENV}" && -f "${FULL_RUN_ENV}" ]]; then
  set -a
  source "${FULL_RUN_ENV}"
  set +a
fi

export METHOD
export ATTACK
export BUDGET
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

export TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
export DEVICE="${DEVICE:-cuda}"
export ATTACK_OUTPUT_ROOT="${ATTACK_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/attacks_full}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/defenses/detectors_b${BUDGET}}"
export LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/defenses/detectors_b${BUDGET}}"
export DETECTOR_OUT="${DETECTOR_OUT:-${OUTPUT_ROOT}/${METHOD}/${ATTACK}}"
mkdir -p "${DETECTOR_OUT}" "${LOG_ROOT}/${METHOD}/${ATTACK}"

find_latest_attack_manifest() {
  python3 - "$ATTACK_OUTPUT_ROOT" "$ATTACK" "$BUDGET" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
attack = sys.argv[2]
budget = int(sys.argv[3])
attack_root = root / attack
if not attack_root.exists():
    raise SystemExit(f"No attack output directory: {attack_root}")

candidates = []
for path in attack_root.glob("*/attack_manifest.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    run_config = payload.get("run_config") if isinstance(payload.get("run_config"), dict) else {}
    manifest_attack = result.get("attack") or run_config.get("attack") or payload.get("attack")
    manifest_budget = result.get("budget") or run_config.get("budget") or payload.get("budget")
    status = result.get("status") or payload.get("status")
    if manifest_attack != attack or int(manifest_budget or -1) != budget:
        continue
    if status == "dry_run":
        continue
    candidates.append(path)

if not candidates:
    raise SystemExit(f"No non-dry-run attack_manifest.json found for attack={attack} budget={budget} under {attack_root}")
print(max(candidates, key=lambda p: p.stat().st_mtime).resolve())
PY
}

if [[ -z "${ATTACK_MANIFEST:-}" && -z "${STUDENT_CHECKPOINT:-}" ]]; then
  if [[ "${METHOD}" == "query_traffic" && -n "${ATTACK_OUTPUT_SOURCE:-}" ]]; then
    :
  elif [[ -d "${ATTACK_OUTPUT_ROOT}" ]]; then
    ATTACK_MANIFEST="$(find_latest_attack_manifest)"
    export ATTACK_MANIFEST
  fi
fi

if [[ "${ATTACK}" == "soda" ]]; then
  SODA_CHECKPOINT_TO_VALIDATE="${STUDENT_CHECKPOINT:-}"
  if [[ -z "${SODA_CHECKPOINT_TO_VALIDATE}" && -n "${ATTACK_MANIFEST:-}" ]]; then
    SODA_CHECKPOINT_TO_VALIDATE="$(python3 - "${ATTACK_MANIFEST}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
print(result.get("checkpoint_dir") or payload.get("checkpoint_dir") or "")
PY
)"
  fi
  if [[ -n "${SODA_CHECKPOINT_TO_VALIDATE}" ]]; then
    echo "Validating SODA checkpoint tensors before detector execution: ${SODA_CHECKPOINT_TO_VALIDATE}"
    python3 attacks/scripts/validate_checkpoint_finite.py "${SODA_CHECKPOINT_TO_VALIDATE}"
  fi
fi

python3 attacks/scripts/check_attack_env.py --require-trl --strict-versions

case "${METHOD}" in
  duffin)
    DUFFIN_OUT="${DUFFIN_OUT:-${DETECTOR_OUT}}"
    mkdir -p "${DUFFIN_OUT}"
    COMMAND=(
      python3 -m defenses.duffin.runner
      --teacher_model "${TEACHER_MODEL}"
      --output_dir "${DUFFIN_OUT}"
      --max_probes "${DUFFIN_MAX_PROBES:-1000}"
      --max_new_tokens "${DUFFIN_MAX_NEW_TOKENS:-1024}"
      --min_valid_rate "${DUFFIN_MIN_VALID_RATE:-0.9}"
      --batch_size "${DUFFIN_BATCH_SIZE:-4}"
      --probe_source "${DUFFIN_PROBE_SOURCE:-mmlu_pro}"
      --probe_categories "${DUFFIN_PROBE_CATEGORIES:-biology,business,chemistry,computer_science,math,physics}"
      --label "${DUFFIN_LABEL:-${ATTACK}_b${BUDGET}}"
    )
    if [[ -n "${STUDENT_CHECKPOINT:-}" ]]; then
      COMMAND+=(--student_checkpoint "${STUDENT_CHECKPOINT}")
    elif [[ -n "${ATTACK_MANIFEST:-}" ]]; then
      COMMAND+=(--attack_manifest "${ATTACK_MANIFEST}")
    else
      echo "DuFFin needs STUDENT_CHECKPOINT or ATTACK_MANIFEST. Set one manually, or set ATTACK_OUTPUT_ROOT so the latest manifest can be found." >&2
      exit 2
    fi
    if [[ -n "${STUDENT_BASE_MODEL:-}" ]]; then
      COMMAND+=(--student_base_model "${STUDENT_BASE_MODEL}")
    fi
    ;;
  query_traffic)
    QT_OUT="${QT_OUT:-${DETECTOR_OUT}}"
    QT_DATA_DIR="${QT_DATA_DIR:-${QT_OUT}/data}"
    QT_DETECTORS="${QT_DETECTORS:-mmd prada seat}"
    QT_BENIGN_MODE="${QT_BENIGN_MODE:-matched_mix}"
    QT_NUM_QUERIES="${QT_NUM_QUERIES:-${BUDGET}}"
    mkdir -p "${QT_DATA_DIR}"

    if [[ -z "${BENIGN_QUERY_LOG:-}" ]]; then
      BENIGN_DIR="${QT_DATA_DIR}/benign"
      python3 -m defenses.query_traffic.prepare_benign_reference \
        --mode "${QT_BENIGN_MODE}" \
        --output_dir "${BENIGN_DIR}" \
        --num_queries "${QT_NUM_QUERIES}" \
        --attack_budget "${BUDGET}" \
        --fallback_user_from_candidate_lmsys
      BENIGN_QUERY_LOG="${BENIGN_DIR}/benign_${QT_BENIGN_MODE}.jsonl"
      export BENIGN_QUERY_LOG
    fi

    if [[ -z "${TEACHER_QUERY_LOG:-}" && -z "${ATTACK_MANIFEST:-}" ]]; then
      if [[ -z "${ATTACK_OUTPUT_SOURCE:-}" ]]; then
        echo "Query Traffic needs TEACHER_QUERY_LOG, ATTACK_MANIFEST, or ATTACK_OUTPUT_SOURCE." >&2
        echo "Set ATTACK_OUTPUT_ROOT for automatic latest-manifest lookup, or provide one of those inputs manually." >&2
        exit 2
      fi
      COLLECTED_DIR="${COLLECTED_ATTACK_QUERIES_DIR:-${QT_DATA_DIR}/collected_attack_queries}"
      python3 -m defenses.query_traffic.collect_attack_queries \
        --source-root "${ATTACK_OUTPUT_SOURCE}" \
        --output-dir "${COLLECTED_DIR}" \
        --limit "${QT_NUM_QUERIES}"
      TEACHER_QUERY_LOG="$(python3 - "${COLLECTED_DIR}/attack_query_collection_index.json" "${ATTACK}" <<'PY'
import json
import sys
from pathlib import Path
index = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
wanted = sys.argv[2]
results = index.get("results") or []
filtered = [row for row in results if row.get("attack") == wanted]
if filtered:
    results = filtered
if not results:
    raise SystemExit("No collected attack query logs in index")
print(results[0]["path"])
PY
)"
      export TEACHER_QUERY_LOG
    fi
    ;;
  *)
    echo "Unknown detector METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

cat <<EOF
Detector defense experiment:
  METHOD=${METHOD}
  ATTACK=${ATTACK}
  BUDGET=${BUDGET}
  REPO_DIR=${REPO_DIR}
  STORAGE_ROOT=${STORAGE_ROOT}
  ATTACK_OUTPUT_ROOT=${ATTACK_OUTPUT_ROOT}
  ATTACK_MANIFEST=${ATTACK_MANIFEST:-}
  STUDENT_CHECKPOINT=${STUDENT_CHECKPOINT:-}
  TEACHER_QUERY_LOG=${TEACHER_QUERY_LOG:-}
  BENIGN_QUERY_LOG=${BENIGN_QUERY_LOG:-}
  OUTPUT_DIR=${DETECTOR_OUT}
EOF

case "${METHOD}" in
  duffin)
    "${COMMAND[@]}"
    ;;
  query_traffic)
    status=0
    for detector in ${QT_DETECTORS}; do
      COMMAND=(
        python3 -m defenses.query_traffic.runner
        --detector "${detector}"
        --benign_query_log "${BENIGN_QUERY_LOG}"
        --output_dir "${QT_OUT}/${detector}"
        --max_teacher_queries "${QT_NUM_QUERIES}"
        --max_benign "${QT_NUM_QUERIES}"
        --batch_size "${QT_BATCH_SIZE:-50}"
        --null_samples "${QT_NULL_SAMPLES:-200}"
        --embedding_model "${QT_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
        --device "${DEVICE}"
        --compute_device "${DEVICE}"
        --mmd_device "${DEVICE}"
      )
      if [[ -n "${TEACHER_QUERY_LOG:-}" ]]; then
        COMMAND+=(--teacher_query_log "${TEACHER_QUERY_LOG}")
      else
        COMMAND+=(--attack_manifest "${ATTACK_MANIFEST}")
      fi
      "${COMMAND[@]}" || status=$?
    done
    exit "${status}"
    ;;
esac
