#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
QT_OUT="${QT_OUT:-$OUTPUT_ROOT/query_traffic}"
QT_DATA_DIR="${QT_DATA_DIR:-$QT_OUT/data}"
QT_DETECTORS="${QT_DETECTORS:-prada}"
QT_BENIGN_MODE="${QT_BENIGN_MODE:-matched_mix}"
QT_NUM_QUERIES="${QT_NUM_QUERIES:-20}"
mkdir -p "$QT_DATA_DIR"
ATTACKER_LOG="${ATTACK_QUERY_LOG:-$QT_DATA_DIR/attack_queries.jsonl}"
if [[ ! -f "$ATTACKER_LOG" ]]; then
  python3 - <<PY
from pathlib import Path
import json
out = Path("$ATTACKER_LOG")
out.parent.mkdir(parents=True, exist_ok=True)
rows = []
from attacks.core.hf_query_pool import hf_query_pool_for_budget, resolve_query_pool_and_ordering
from defenses.core.io_utils import load_queries
path, _ = resolve_query_pool_and_ordering(hf_query_pool_for_budget(max(int("$BUDGET"), 100)), "auto")
rows = load_queries(path, max_queries=int("$QT_NUM_QUERIES"))
with out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(out)
PY
fi
BENIGN_DIR="$QT_DATA_DIR/benign"
python3 -m defenses.query_traffic.prepare_benign_reference \
  --mode "$QT_BENIGN_MODE" \
  --output_dir "$BENIGN_DIR" \
  --num_queries "$QT_NUM_QUERIES" \
  --attack_budget "$BUDGET" \
  --fallback_user_from_candidate_lmsys
BENIGN_LOG="$BENIGN_DIR/benign_${QT_BENIGN_MODE}.jsonl"
for detector in $QT_DETECTORS; do
  python3 -m defenses.query_traffic.runner \
    --detector "$detector" \
    --teacher_query_log "$ATTACKER_LOG" \
    --benign_query_log "$BENIGN_LOG" \
    --output_dir "$QT_OUT/$detector" \
    --max_teacher_queries "$QT_NUM_QUERIES" \
    --max_benign "$QT_NUM_QUERIES" \
    --batch_size "${QT_BATCH_SIZE:-5}" \
    --null_samples "${QT_NULL_SAMPLES:-10}" \
    --embedding_model "${QT_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}" \
    --device "$DEVICE" \
    --compute_device "$DEVICE" \
    --mmd_device "$DEVICE"
done
