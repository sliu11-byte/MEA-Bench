#!/usr/bin/env bash

set -euo pipefail
source "${REPO_BASE_DIR:-$(pwd)}/runs/evaluation/defense_common.sh"
python -m evaluation.defense_eval.evaluate --stage reference "${ARGS[@]}"
