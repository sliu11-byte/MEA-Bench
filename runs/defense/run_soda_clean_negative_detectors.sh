#!/usr/bin/env bash

set -euo pipefail
cd "${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
[[ -f defenses/adfp/detector_run.py ]] || { echo 'Submit from repository root.' >&2; exit 2; }
mkdir -p logs
command -v module >/dev/null 2>&1 && module purge || true
command -v module >/dev/null 2>&1 && module load conda/25.7.0 cuda/12.4.1 || true
command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV:-research}" || true
export STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; fi

ROOT="${SODA_FULL_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs/defenses/soda_b1000_seqkd_warmup}"
CLEAN_MANIFEST="${ROOT}/clean/attack_manifest.json"
[[ -s "${CLEAN_MANIFEST}" ]] || { echo "Missing corrected clean SODA manifest: ${CLEAN_MANIFEST}" >&2; exit 2; }

for METHOD in adfp ginsew radioactivity; do
  MANIFEST="${ROOT}/${METHOD}/defense_run_manifest.json"
  [[ -s "${MANIFEST}" ]] || { echo "Missing ${METHOD} defense manifest: ${MANIFEST}" >&2; exit 2; }
  readarray -t PATHS < <(python3 - "${MANIFEST}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))["generator_result"]
print(d["defense_artifacts_dir"])
print(d["defended_transcript_path"])
print(d["teacher_query_log_path"])
PY
  )
  OUT="${ROOT}/${METHOD}/detector_negative/${METHOD}"
  if [[ "${METHOD}" == "adfp" ]]; then
    python3 -m defenses.adfp.detector_run --fingerprint_artifacts "${PATHS[0]}" \
      --transcript "${PATHS[1]}" --attack_manifest "${CLEAN_MANIFEST}" --label negative \
      --max_contexts "${DETECTOR_MAX_QUERIES:-1000}" --output_dir "${OUT}"
  else
    python3 -m "defenses.${METHOD}.detector_run" --watermark_artifacts "${PATHS[0]}" \
      --probe_queries "${PATHS[2]}" --attack_manifest "${CLEAN_MANIFEST}" --label negative \
      --max_queries "${DETECTOR_MAX_QUERIES:-1000}" --max_new_tokens "${DETECTOR_MAX_NEW_TOKENS:-124}" \
      --temperature "${DETECTOR_TEMPERATURE:-0.7}" --output_dir "${OUT}"
  fi
done
echo "Corrected clean SODA negative detectors complete under ${ROOT}"
