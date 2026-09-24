#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

QT_OUT="${QT_OUT:-$OUTPUT_ROOT/query_traffic}"
QT_DATA_DIR="${QT_DATA_DIR:-$QT_OUT/data}"
QT_DETECTORS="${QT_DETECTORS:-mmd prada seat}"
QT_BENIGN_MODE="${QT_BENIGN_MODE:-matched_mix}"
QT_NUM_QUERIES="${QT_NUM_QUERIES:-$BUDGET}"
mkdir -p "$QT_DATA_DIR" "$LOG_ROOT/query_traffic"

if [[ -z "${TEACHER_QUERY_LOG:-}" ]]; then
  if [[ -z "${ATTACK_OUTPUT_SOURCE:-}" ]]; then
    echo "Pass TEACHER_QUERY_LOG, or set ATTACK_OUTPUT_SOURCE to a local attack-output directory." >&2
    exit 2
  fi
  COLLECTED_DIR="${COLLECTED_ATTACK_QUERIES_DIR:-$QT_DATA_DIR/collected_attack_queries}"
  python3 -m defenses.query_traffic.collect_attack_queries \
    --source-root "$ATTACK_OUTPUT_SOURCE" \
    --output-dir "$COLLECTED_DIR" \
    --limit "$QT_NUM_QUERIES"
  TEACHER_QUERY_LOG="$(python3 - "$COLLECTED_DIR/attack_query_collection_index.json" "${ATTACK_NAME:-seqkd}" <<'PY'
import json
import sys
from pathlib import Path
index = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
wanted = sys.argv[2]
results = index.get("results") or []
if wanted:
    filtered = [row for row in results if row.get("attack") == wanted]
    if filtered:
        results = filtered
if not results:
    raise SystemExit("No collected attack query logs in index")
print(results[0]["path"])
PY
)"
fi

if [[ -z "${BENIGN_QUERY_LOG:-}" ]]; then
  BENIGN_DIR="$QT_DATA_DIR/benign"
  python3 -m defenses.query_traffic.prepare_benign_reference \
    --mode "$QT_BENIGN_MODE" \
    --output_dir "$BENIGN_DIR" \
    --num_queries "$QT_NUM_QUERIES" \
    --attack_budget "$BUDGET" \
    --fallback_user_from_candidate_lmsys
  BENIGN_QUERY_LOG="$BENIGN_DIR/benign_${QT_BENIGN_MODE}.jsonl"
fi

set +e
status=0
for detector in $QT_DETECTORS; do
  python3 -m defenses.query_traffic.runner \
    --detector "$detector" \
    --teacher_query_log "$TEACHER_QUERY_LOG" \
    --benign_query_log "$BENIGN_QUERY_LOG" \
    --output_dir "$QT_OUT/$detector" \
    --max_teacher_queries "$QT_NUM_QUERIES" \
    --max_benign "$QT_NUM_QUERIES" \
    --batch_size "${QT_BATCH_SIZE:-50}" \
    --null_samples "${QT_NULL_SAMPLES:-200}" \
    --embedding_model "${QT_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}" \
    --device "$DEVICE" \
    --compute_device "$DEVICE" \
    --mmd_device "$DEVICE" || status=$?
done
set -e
mirror_defense_logs query_traffic "$QT_OUT"
exit "$status"
