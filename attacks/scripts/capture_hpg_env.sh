#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-attacks/env_snapshots}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_FILE="${OUT_FILE:-${OUT_DIR}/hpg_env_${STAMP}.txt}"
mkdir -p "$(dirname "${OUT_FILE}")"

{
  echo "# HPG environment snapshot"
  echo "created_at_utc=${STAMP}"
  echo "hostname=$(hostname)"
  echo "pwd=$(pwd)"
  echo
  echo "## Modules"
  if command -v module >/dev/null 2>&1; then
    module list 2>&1 || true
  else
    echo "module command not available"
  fi
  echo
  echo "## Conda"
  echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
  command -v conda >/dev/null 2>&1 && conda info --envs || true
  echo
  echo "## Python"
  command -v python3 || true
  python3 --version || true
  echo
  echo "## GPU"
  nvidia-smi || true
  echo
  echo "## Key package versions"
  pip show transformers trl peft accelerate datasets torch vllm huggingface-hub PyYAML numpy tqdm || true
  echo
  echo "## Full pip freeze"
  pip freeze || true
  echo
  echo "## Attack env check"
  python3 attacks/scripts/check_attack_env.py --require-trl --require-vllm --strict-versions || true
} > "${OUT_FILE}"

echo "Wrote HPG environment snapshot: ${OUT_FILE}"
