#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_countermeasure_seqkd_script radioactivity "$@"
