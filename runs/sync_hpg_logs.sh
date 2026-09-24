#!/usr/bin/env bash
# Sync the benchmark logs directory from HiPerGator to this local checkout.
#
# Usage:
#   bash runs/sync_hpg_logs.sh [--dry-run]
#   bash runs/sync_hpg_logs.sh --host hpg.rc.ufl.edu --user anonymous_user
#   bash runs/sync_hpg_logs.sh --destination /path/to/local/logs
#
# Environment overrides:
#   HPG_HOST, HPG_USER, HPG_REPO_DIR, LOCAL_LOG_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

HPG_HOST="${HPG_HOST:-hpg.rc.ufl.edu}"
HPG_USER="${HPG_USER:-anonymous_user}"
HPG_REPO_DIR="${HPG_REPO_DIR:-/home/anonymous_user/project/A-Benchmark-for-Model-distillation-survey}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${REPO_DIR}/logs}"
DRY_RUN=0

usage() {
  sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while (( $# > 0 )); do
  case "$1" in
    --host) HPG_HOST="${2:?--host requires a value}"; shift 2 ;;
    --user) HPG_USER="${2:?--user requires a value}"; shift 2 ;;
    --remote-repo) HPG_REPO_DIR="${2:?--remote-repo requires a value}"; shift 2 ;;
    --destination) LOCAL_LOG_DIR="${2:?--destination requires a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required in WSL. Install it with: sudo apt install rsync" >&2
  exit 1
fi
if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required in WSL." >&2
  exit 1
fi

mkdir -p "${LOCAL_LOG_DIR}"
REMOTE="${HPG_USER}@${HPG_HOST}:${HPG_REPO_DIR%/}/logs/"

echo "HPG logs sync"
echo "  source:      ${REMOTE}"
echo "  destination: ${LOCAL_LOG_DIR}/"
echo "  mode:        $([[ ${DRY_RUN} == 1 ]] && echo preview || echo download)"

RSYNC_ARGS=(
  --archive
  --compress
  --partial
  --human-readable
  --itemize-changes
  --safe-links
)
if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

# No --delete: local-only files and earlier downloaded logs are preserved.
rsync "${RSYNC_ARGS[@]}" -e ssh "${REMOTE}" "${LOCAL_LOG_DIR}/"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Preview complete; no files were changed."
else
  echo "Log sync complete: ${LOCAL_LOG_DIR}"
fi
