#!/usr/bin/env bash

set -euo pipefail

command -v module >/dev/null 2>&1 && module purge || true
command -v module >/dev/null 2>&1 && module load conda/25.7.0 || true
command -v module >/dev/null 2>&1 && module load cuda/12.8.1 || true
command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV:-research}" || true
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

export OMP_NUM_THREADS="${CPU_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

SUBMIT_DIR="${REPO_BASE_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/scripts/seqkd_smoke/run_ginsew_seqkd_smoke.sh" && "$(basename "${SUBMIT_DIR}")" == "defenses" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/run_ginsew_seqkd_smoke.sh" && "$(basename "${SUBMIT_DIR}")" == "seqkd_smoke" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/../../.." && pwd)"
elif [[ -f "${SUBMIT_DIR}/defenses/scripts/seqkd_smoke/run_ginsew_seqkd_smoke.sh" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey/attacks/scripts/run_attack.py" ]]; then
  REPO_DIR="${SUBMIT_DIR}/A-Benchmark-for-Model-distillation-survey"
else
  echo "Could not locate A-Benchmark-for-Model-distillation-survey from submit dir: ${SUBMIT_DIR}" >&2
  echo "Submit from the repository root or defenses/scripts/seqkd_smoke." >&2
  exit 1
fi
cd "${REPO_DIR}"

DEFAULT_STORAGE_ROOT="${REPO_DIR}"
if [[ -d "/path/to/storage/${USER}" ]]; then
  DEFAULT_STORAGE_ROOT="/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey"
elif [[ -d "/home/${USER}/project/A-Benchmark-for-Model-distillation-survey" ]]; then
  DEFAULT_STORAGE_ROOT="/home/${USER}/project/A-Benchmark-for-Model-distillation-survey"
fi
export STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}" "${STORAGE_ROOT}/outputs/defenses/seqkd_smoke_online"

python3 attacks/scripts/check_attack_env.py --require-trl --strict-versions

export METHOD="${METHOD:-ginsew}"
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}"
export PROXY_MODEL="${PROXY_MODEL:-${STUDENT_MODEL}}"
export BUDGET="${BUDGET:-100}"
export QUERY_POOL="${QUERY_POOL:-auto}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
export DETECTOR_MAX_QUERIES="${DETECTOR_MAX_QUERIES:-5}"
export DETECTOR_MAX_NEW_TOKENS="${DETECTOR_MAX_NEW_TOKENS:-8}"
export DETECTOR_TEMPERATURE="${DETECTOR_TEMPERATURE:-0.7}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/defenses/seqkd_smoke_online}"
export LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs/defenses/seqkd_smoke_online}"
export DEVICE="${DEVICE:-cuda}"

case "${METHOD}" in
  ginsew|radioactivity|adfp|ads|trace_rewriting|doge|duffin|query_traffic) ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac

cat <<EOF
SeqKD defense smoke configuration:
  REPO_DIR=${REPO_DIR}
  STORAGE_ROOT=${STORAGE_ROOT}
  HF_HOME=${HF_HOME}
  METHOD=${METHOD}
  TEACHER_MODEL=${TEACHER_MODEL}
  STUDENT_MODEL=${STUDENT_MODEL}
  BUDGET=${BUDGET}
  QUERY_POOL=${QUERY_POOL}
  OUTPUT_ROOT=${OUTPUT_ROOT}
  LOG_ROOT=${LOG_ROOT}
  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
  DETECTOR_MAX_QUERIES=${DETECTOR_MAX_QUERIES}
EOF

bash "defenses/scripts/seqkd_smoke/run_${METHOD}_seqkd_smoke.sh"
