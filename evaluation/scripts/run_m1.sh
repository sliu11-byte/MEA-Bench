#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

usage() {
  cat >&2 <<'USAGE'
Usage: evaluation/scripts/run_m1.sh <dry-run|smoke|smoke-10|pilot|full> [extra rollout args...]

Environment:
  M1_PYTHON              Python executable, default: python
  M1_CONFIG              Rollout config path, default: evaluation/configs/m1_rollout.yaml
  M1_PREFLIGHT_RUN_ID    Required for pilot unless --preflight-run-id is passed explicitly
  M1_ALLOW_FULL_RUN=1    Required for full mode

Endpoint variables are read by the config, for example:
  QWEN72_BASE_URL=http://127.0.0.1:8002/v1
  QWEN7_BASE_URL=http://127.0.0.1:8000/v1
  MISTRAL7_BASE_URL=http://127.0.0.1:8001/v1
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

MODE="$1"
shift

PYTHON_BIN="${M1_PYTHON:-python}"
CONFIG_PATH="${M1_CONFIG:-evaluation/configs/m1_rollout.yaml}"
ARGS=(--config "${CONFIG_PATH}")

case "${MODE}" in
  dry-run)
    ARGS+=(--mode dry-run)
    ;;
  smoke)
    ARGS+=(--mode smoke)
    ;;
  smoke-10)
    ARGS+=(--mode smoke --single-split-per-dataset --smoke-examples-per-split 10 --manifest-pool-size 10)
    ;;
  pilot)
    ARGS+=(--mode pilot)
    if [[ " $* " != *" --preflight-run-id "* ]]; then
      if [[ -z "${M1_PREFLIGHT_RUN_ID:-}" ]]; then
        echo "Set M1_PREFLIGHT_RUN_ID or pass --preflight-run-id before starting pilot." >&2
        exit 2
      fi
      ARGS+=(--preflight-run-id "${M1_PREFLIGHT_RUN_ID}")
    fi
    ;;
  full)
    if [[ "${M1_ALLOW_FULL_RUN:-0}" != "1" ]]; then
      echo "Full M1 is intentionally disabled. Set M1_ALLOW_FULL_RUN=1 only after explicit approval." >&2
      exit 2
    fi
    ARGS+=(--mode full --confirm-full-run)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown M1 mode: ${MODE}" >&2
    usage
    exit 2
    ;;
esac

exec "${PYTHON_BIN}" -m evaluation.rollouts.m1_capability "${ARGS[@]}" "$@"
