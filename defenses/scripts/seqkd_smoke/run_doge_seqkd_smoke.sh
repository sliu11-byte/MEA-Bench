#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
DOGE_WORK_DIR="${DOGE_WORK_DIR:-$OUTPUT_ROOT/doge_prep}"
DOGE_CHECKPOINT="${DOGE_CHECKPOINT:-$DOGE_WORK_DIR/doge_teacher}"
DOGE_TRAIN_EXAMPLES="${DOGE_TRAIN_EXAMPLES:-8}"
mkdir -p "$DOGE_WORK_DIR"
if [[ ! -f "$DOGE_CHECKPOINT/lm_head.pt" ]]; then
  python3 - <<PY
from pathlib import Path
from attacks.core.hf_query_pool import hf_query_pool_for_budget, resolve_query_pool_and_ordering
from defenses.core.io_utils import load_queries, write_jsonl
budget = int("$BUDGET")
n = int("$DOGE_TRAIN_EXAMPLES")
query_pool = "$QUERY_POOL"
query_pool_path, _ = resolve_query_pool_and_ordering(hf_query_pool_for_budget(max(budget, 100)) if query_pool == "auto" else query_pool, "auto")
queries = load_queries(query_pool_path, max_queries=n)
rows = [{"query": q["query"], "teacher_response": "Smoke-test teacher response."} for q in queries]
out = Path("$DOGE_WORK_DIR") / "doge_train.jsonl"
write_jsonl(rows, out)
print(out)
PY
  python3 -m defenses.doge.run train \
    --teacher-model "$TEACHER_MODEL" \
    --proxy-model "${DOGE_PROXY_MODEL:-$TEACHER_MODEL}" \
    --train-file "$DOGE_WORK_DIR/doge_train.jsonl" \
    --output-dir "$DOGE_CHECKPOINT" \
    --max-steps "${DOGE_MAX_STEPS:-1}" \
    --batch-size 1 \
    --gradient-accumulation-steps 1 \
    --max-length 256
fi
run_seqkd_defense_runner doge \
  --doge-checkpoint "$DOGE_CHECKPOINT" \
  --no-detector
