#!/usr/bin/env bash
# Portable front end for the original attack orchestrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

usage() {
  cat <<'EOF'
Usage: bash runs/run_attack.sh --attack NAME --budget N [options]

Required:
  --attack NAME            seqkd, lord, soda, qedks, model_leeching, or gad
  --budget N               100, 1000, or 10000 under --profile paper

Options:
  --profile NAME           Benchmark profile (default: paper)
  --query-pool PATH|auto   Query-pool file or anonymous HF source (default: auto)
  --query-ordering PATH    Optional deterministic ordering file
  --teacher-endpoint URL   OpenAI-compatible teacher endpoint
  --student-endpoint URL   OpenAI-compatible student endpoint (used by SODA)
  --teacher-serving MODE   local or external (default: local)
  --student-serving MODE   local or external (default: local; used by SODA)
  --transcript-dir PATH    Reuse a compatible teacher transcript bundle
  --warmup-model PATH      Explicit SODA/LoRD warmup checkpoint
  --output-root PATH       Artifact root (default: outputs)
  --seed N                 Override the profile seed
  --resume                 Resume method-owned partial state when supported
  --dry-run                Validate and print the resolved plan only
  -h, --help               Show this help

Activate the desired Python/CUDA environment before invoking this script.
EOF
}

ATTACK=""
BUDGET=""
PROFILE="paper"
QUERY_POOL="auto"
QUERY_ORDERING="auto"
TEACHER_ENDPOINT=""
STUDENT_ENDPOINT=""
TEACHER_SERVING="local"
STUDENT_SERVING="local"
TRANSCRIPT_DIR=""
WARMUP_MODEL_VALUE=""
OUTPUT_ROOT="outputs"
SEED_VALUE=""
RESUME=0
DRY_RUN=0

while (($#)); do
  case "$1" in
    --attack) ATTACK="${2:?--attack requires a value}"; shift 2 ;;
    --budget) BUDGET="${2:?--budget requires a value}"; shift 2 ;;
    --profile) PROFILE="${2:?--profile requires a value}"; shift 2 ;;
    --query-pool) QUERY_POOL="${2:?--query-pool requires a value}"; shift 2 ;;
    --query-ordering) QUERY_ORDERING="${2:?--query-ordering requires a value}"; shift 2 ;;
    --teacher-endpoint) TEACHER_ENDPOINT="${2:?--teacher-endpoint requires a value}"; shift 2 ;;
    --student-endpoint) STUDENT_ENDPOINT="${2:?--student-endpoint requires a value}"; shift 2 ;;
    --teacher-serving) TEACHER_SERVING="${2:?--teacher-serving requires a value}"; shift 2 ;;
    --student-serving) STUDENT_SERVING="${2:?--student-serving requires a value}"; shift 2 ;;
    --transcript-dir) TRANSCRIPT_DIR="${2:?--transcript-dir requires a value}"; shift 2 ;;
    --warmup-model) WARMUP_MODEL_VALUE="${2:?--warmup-model requires a value}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?--output-root requires a value}"; shift 2 ;;
    --seed) SEED_VALUE="${2:?--seed requires a value}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${ATTACK}" ]] || { echo "--attack is required" >&2; exit 2; }
[[ -n "${BUDGET}" ]] || { echo "--budget is required" >&2; exit 2; }

PLAN_ARGS=(--attack "${ATTACK}" --budget "${BUDGET}" --profile "${PROFILE}"
  --query-pool "${QUERY_POOL}" --query-ordering "${QUERY_ORDERING}"
  --output-root "${OUTPUT_ROOT}" --teacher-serving "${TEACHER_SERVING}"
  --student-serving "${STUDENT_SERVING}" --dry-run)
[[ -n "${TEACHER_ENDPOINT}" ]] && PLAN_ARGS+=(--teacher-endpoint "${TEACHER_ENDPOINT}")
[[ -n "${STUDENT_ENDPOINT}" ]] && PLAN_ARGS+=(--student-endpoint "${STUDENT_ENDPOINT}")
[[ -n "${TRANSCRIPT_DIR}" ]] && PLAN_ARGS+=(--transcript-dir "${TRANSCRIPT_DIR}")
[[ -n "${WARMUP_MODEL_VALUE}" ]] && PLAN_ARGS+=(--warmup-model "${WARMUP_MODEL_VALUE}")
[[ -n "${SEED_VALUE}" ]] && PLAN_ARGS+=(--seed "${SEED_VALUE}")
((RESUME)) && PLAN_ARGS+=(--resume)

if ((DRY_RUN)); then
  exec python3 -m benchmark.cli attack "${PLAN_ARGS[@]}"
fi

# Resolve and validate the public configuration before entering the legacy
# method dispatcher. Its method implementations remain the source of truth.
PLAN_JSON="$(mktemp)"
trap 'rm -f "${PLAN_JSON}"' EXIT
python3 -m benchmark.cli attack "${PLAN_ARGS[@]}" --no-write-manifest > "${PLAN_JSON}"

eval "$(python3 - "${PLAN_JSON}" <<'PY'
import json
import shlex
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
config = plan["config"]
teacher = config["teacher"]
student = config["student"]
training = config["training"]
method = config["method"]

values = {
    "SEED": config["seed"],
    "TEACHER_BACKEND": teacher["backend"],
    "TEACHER_MODEL": teacher["model"],
    "TEACHER_ENDPOINT_URL": teacher["endpoint"],
    "TEACHER_REQUEST_MODEL": teacher["request_model"],
    "TEACHER_TEMPERATURE": teacher["temperature"],
    "TEACHER_TOP_P": teacher["top_p"],
    "TEACHER_MAX_TOKENS": teacher["max_tokens"],
    "STUDENT_MODEL": student["model"],
    "STUDENT_ENDPOINT_URL": student["endpoint"],
    "STUDENT_REQUEST_MODEL": student["request_model"],
    "STUDENT_TEMPERATURE": student["temperature"],
    "STUDENT_TOP_P": student["top_p"],
    "STUDENT_MAX_TOKENS": student["max_tokens"],
    "ATTACK_USE_LORA": int(bool(training["use_lora"])),
    "ATTACK_BF16": int(bool(training["bf16"])),
    "ATTACK_GRADIENT_CHECKPOINTING": int(bool(training["gradient_checkpointing"])),
    "ATTACK_LORA_R": training["lora_r"],
    "ATTACK_LORA_ALPHA": training["lora_alpha"],
    "ATTACK_LORA_DROPOUT": training["lora_dropout"],
}
prefixes = {
    "soda": "SODA", "qedks": "QEDKS", "model_leeching": "MODEL_LEECHING", "gad": "GAD"
}
prefix = prefixes.get(config["attack"])
if prefix:
    for key, value in method.items():
        if isinstance(value, bool):
            value = int(value)
        values[f"{prefix}_{key.upper()}"] = value
for key, value in values.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"

export REPO_DIR
export STORAGE_ROOT="${STORAGE_ROOT:-${REPO_DIR}}"
export ATTACKS="${ATTACK}"
export ATTACK_INDEX=0
export BUDGET
export QUERY_POOL
export QUERY_ORDERING
export OUTPUT_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUTPUT_ROOT}")/attacks"
export WARMUP_MODEL="${WARMUP_MODEL_VALUE}"
export SHARED_TRANSCRIPT_DIR="${TRANSCRIPT_DIR}"
export REQUIRE_SHARED_TRANSCRIPT=0
export BENCHMARK_RESUME="${RESUME}"
export ONLINE_ATTACK_TEACHER_MODE="${TEACHER_SERVING}"
export SODA_STUDENT_MODE="${STUDENT_SERVING}"
if [[ "${TEACHER_SERVING}" == "external" ]]; then
  export AUTO_START_TEACHER_FOR_ONLINE_ATTACKS=0
fi

case "${ATTACK}" in
  seqkd) export SEQKD_BUDGET="${BUDGET}" ;;
  lord) export LORD_BUDGET="${BUDGET}" ;;
  soda) export SODA_BUDGET="${BUDGET}" ;;
  qedks) export QEDKS_BUDGET="${BUDGET}" ;;
  model_leeching) export MODEL_LEECHING_BUDGET="${BUDGET}" ;;
  gad) export GAD_BUDGET="${BUDGET}" ;;
esac

# The former Slurm controller built one shared transcript before launching the
# offline attacks. In portable mode the same lifecycle is available explicitly.
if [[ "${TEACHER_SERVING}" == "local" && -z "${SHARED_TRANSCRIPT_DIR}" ]]; then
  case "${ATTACK}" in
    seqkd|lord|soda|gad)
      RUNTIME_DIR="${OUTPUT_DIR}/.runtime/${ATTACK}_b${BUDGET}_$$"
      mkdir -p "${RUNTIME_DIR}"
      export VLLM_ENDPOINT_ENV_PATH="${RUNTIME_DIR}/teacher_endpoint.env"
      export VLLM_ADVERTISE_HOST="127.0.0.1"
      export LATEST_SHARED_TRANSCRIPT_PATH="${RUNTIME_DIR}/latest_transcript.txt"
      export SHARED_TRANSCRIPT_ROOT="${OUTPUT_DIR}/shared_teacher_transcripts"
      unset TEACHER_ENDPOINT_URL TEACHER_API_KEY
      bash attacks/scripts/serve_teacher.sh >"${RUNTIME_DIR}/teacher.log" 2>&1 &
      TEACHER_PID=$!
      cleanup_teacher() {
        kill "${TEACHER_PID}" >/dev/null 2>&1 || true
        wait "${TEACHER_PID}" >/dev/null 2>&1 || true
      }
      trap cleanup_teacher EXIT
      for _ in $(seq 1 "${ENDPOINT_STARTUP_POLLS:-180}"); do
        [[ -s "${VLLM_ENDPOINT_ENV_PATH}" ]] && source "${VLLM_ENDPOINT_ENV_PATH}"
        if [[ -n "${TEACHER_ENDPOINT_URL:-}" ]] && curl -fsS -H "Authorization: Bearer ${TEACHER_API_KEY:-EMPTY}" \
          "${TEACHER_ENDPOINT_URL%/}/models" >/dev/null 2>&1; then
          break
        fi
        kill -0 "${TEACHER_PID}" >/dev/null 2>&1 || {
          echo "Local teacher exited before becoming ready; see ${RUNTIME_DIR}/teacher.log" >&2; exit 1;
        }
        sleep "${ENDPOINT_STARTUP_POLL_SECONDS:-5}"
      done
      [[ -n "${TEACHER_ENDPOINT_URL:-}" ]] && curl -fsS -H "Authorization: Bearer ${TEACHER_API_KEY:-EMPTY}" \
        "${TEACHER_ENDPOINT_URL%/}/models" >/dev/null || {
          echo "Timed out waiting for local teacher; see ${RUNTIME_DIR}/teacher.log" >&2; exit 1;
        }
      export TRANSCRIPT_BUDGET="${BUDGET}"
      bash attacks/scripts/run_shared_transcript.sh
      export SHARED_TRANSCRIPT_DIR="$(<"${LATEST_SHARED_TRANSCRIPT_PATH}")"
      cleanup_teacher
      trap - EXIT
      ;;
  esac
fi

# SODA is initialized from a matched SeqKD run. Under the default local serving
# path, create that prerequisite automatically from the exact transcript that
# will be used by SODA when no compatible completed run exists yet.
if [[ "${ATTACK}" == "soda" && -z "${WARMUP_MODEL_VALUE}" && -n "${SHARED_TRANSCRIPT_DIR}" ]]; then
  RESOLVE_SODA_ARGS=(
    --budget "${BUDGET}"
    --output-dir "${OUTPUT_DIR}"
    --student-model "${STUDENT_MODEL}"
    --query-pool "${QUERY_POOL}"
    --query-ordering "${QUERY_ORDERING}"
    --transcript-dir "${SHARED_TRANSCRIPT_DIR}"
  )
  if ! python3 attacks/scripts/resolve_soda_warmup.py "${RESOLVE_SODA_ARGS[@]}" >/dev/null 2>&1; then
    echo "No compatible SeqKD initialization was found; preparing it automatically for SODA."
    SEQKD_ARGS=(
      --attack seqkd
      --budget "${BUDGET}"
      --profile "${PROFILE}"
      --query-pool "${QUERY_POOL}"
      --query-ordering "${QUERY_ORDERING}"
      --transcript-dir "${SHARED_TRANSCRIPT_DIR}"
      --teacher-serving external
      --student-serving external
      --output-root "${OUTPUT_ROOT}"
    )
    [[ -n "${SEED_VALUE}" ]] && SEQKD_ARGS+=(--seed "${SEED_VALUE}")
    bash runs/run_attack.sh "${SEQKD_ARGS[@]}"
    python3 attacks/scripts/resolve_soda_warmup.py "${RESOLVE_SODA_ARGS[@]}" >/dev/null
  fi
fi

rm -f "${PLAN_JSON}"
trap - EXIT
exec bash attacks/scripts/run_attacks.sh
