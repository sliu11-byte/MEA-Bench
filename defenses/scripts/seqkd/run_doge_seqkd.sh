#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DOGE_WORK_DIR="${DOGE_WORK_DIR:-$OUTPUT_ROOT/doge/prep}"
DOGE_CHECKPOINT="${DOGE_CHECKPOINT:-$DOGE_WORK_DIR/doge_teacher}"
DOGE_TRAIN_FILE="${DOGE_TRAIN_FILE:-$DOGE_WORK_DIR/doge_train.jsonl}"
DOGE_TRAIN_EXAMPLES="${DOGE_TRAIN_EXAMPLES:-$BUDGET}"
mkdir -p "$DOGE_WORK_DIR"

if [[ ! -f "$DOGE_TRAIN_FILE" ]]; then
  require_env TEACHER_BASE_URL
  python3 - "$BUDGET" "$DOGE_TRAIN_EXAMPLES" "$QUERY_POOL" "$DOGE_TRAIN_FILE" "$TEACHER_BASE_URL" "${TEACHER_REQUEST_MODEL:-$TEACHER_MODEL}" "${TEACHER_API_KEY:-EMPTY}" "$MAX_NEW_TOKENS" "$TEACHER_TEMPERATURE" "$TEACHER_TOP_P" <<'PY'
from pathlib import Path
import sys

from attacks.core.hf_query_pool import hf_query_pool_for_budget, resolve_query_pool_and_ordering
from defenses.core.generation import generate_openai_chat_responses
from defenses.core.io_utils import load_queries, write_jsonl

budget = int(sys.argv[1])
num_examples = int(sys.argv[2])
query_pool = sys.argv[3]
train_file = Path(sys.argv[4])
base_url = sys.argv[5]
model = sys.argv[6]
api_key = sys.argv[7]
max_new_tokens = int(sys.argv[8])
temperature = float(sys.argv[9])
top_p = float(sys.argv[10])

needed = max(budget, num_examples)
query_pool_spec = hf_query_pool_for_budget(needed) if query_pool == "auto" else query_pool
query_pool_path, _ = resolve_query_pool_and_ordering(query_pool_spec, "auto")
queries = load_queries(query_pool_path, max_queries=num_examples)
if len(queries) < num_examples:
    raise SystemExit(f"Need {num_examples} DOGe training queries, got {len(queries)} from {query_pool_path}")
responses = generate_openai_chat_responses(
    [q["query"] for q in queries],
    base_url=base_url,
    model=model,
    api_key=api_key,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
)
rows = []
for q, response in zip(queries, responses):
    rows.append({
        "query_id": q.get("query_id"),
        "query": q["query"],
        "teacher_response": response,
        "teacher_model": model,
        "source": "doge_auto_teacher_collection",
    })
train_file.parent.mkdir(parents=True, exist_ok=True)
write_jsonl(rows, train_file)
print(train_file)
PY
fi

if [[ ! -f "$DOGE_CHECKPOINT/lm_head.pt" ]]; then
  python3 -m defenses.doge.run train \
    --teacher-model "$TEACHER_MODEL" \
    --proxy-model "${DOGE_PROXY_MODEL:-$TEACHER_MODEL}" \
    --train-file "$DOGE_TRAIN_FILE" \
    --output-dir "$DOGE_CHECKPOINT" \
    --anti-kd-coef "${DOGE_ANTI_KD_COEF:-3e-5}" \
    --kd-temperature "${DOGE_KD_TEMPERATURE:-2.0}" \
    --learning-rate "${DOGE_LEARNING_RATE:-5e-5}" \
    --weight-decay "${DOGE_WEIGHT_DECAY:-0.01}" \
    --batch-size "${DOGE_BATCH_SIZE:-1}" \
    --gradient-accumulation-steps "${DOGE_GRADIENT_ACCUMULATION_STEPS:-16}" \
    --num-train-epochs "${DOGE_NUM_TRAIN_EPOCHS:-2}" \
    ${DOGE_MAX_STEPS:+--max-steps "$DOGE_MAX_STEPS"} \
    --max-length "${DOGE_MAX_LENGTH:-2048}"
fi

run_seqkd_defense_runner doge \
  --doge-checkpoint "$DOGE_CHECKPOINT" \
  --no-detector
