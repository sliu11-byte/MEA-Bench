#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ADS_WORK_DIR="${ADS_WORK_DIR:-$OUTPUT_ROOT/ads/prep}"
ADS_HOLDOUT="${ADS_HOLDOUT:-2880}"
ADS_GRAD_MAX_LENGTH="${ADS_GRAD_MAX_LENGTH:-512}"
mkdir -p "$ADS_WORK_DIR"
GRAD_PATH="${GRAD_PATH:-$ADS_WORK_DIR/student_grads.pt}"

if [[ ! -f "$GRAD_PATH" ]]; then
  HOLDOUT_PATH="${ADS_HOLDOUT_TRANSCRIPT:-$ADS_WORK_DIR/holdout_teacher.jsonl}"
  if [[ ! -f "$HOLDOUT_PATH" ]]; then
    python3 - "$BUDGET" "$ADS_HOLDOUT" "$QUERY_POOL" "$ADS_WORK_DIR" "$TEACHER_MODEL" "$PROXY_MODEL" "$DEVICE" "$MAX_NEW_TOKENS" "$ADS_BATCH_SIZE" "$HOLDOUT_PATH" <<'PY'
from pathlib import Path
import sys

from attacks.core.hf_query_pool import hf_query_pool_for_budget, resolve_query_pool_and_ordering
from defenses.ads.generator import ADSGenerator
from defenses.core.io_utils import load_queries, write_jsonl

budget = int(sys.argv[1])
holdout = int(sys.argv[2])
query_pool = sys.argv[3]
out_dir = Path(sys.argv[4])
teacher_model = sys.argv[5]
proxy_model = sys.argv[6]
device = sys.argv[7]
max_new_tokens = int(sys.argv[8])
batch_size = int(sys.argv[9])
holdout_path = Path(sys.argv[10])

query_pool_spec = hf_query_pool_for_budget(budget + holdout) if query_pool == "auto" else query_pool
query_pool_path, _ = resolve_query_pool_and_ordering(query_pool_spec, "auto")
queries = load_queries(query_pool_path, max_queries=budget + holdout)[budget:budget + holdout]
if len(queries) < holdout:
    raise SystemExit(f"Need {holdout} holdout queries, got {len(queries)} from {query_pool_path}")

out_dir.mkdir(parents=True, exist_ok=True)
gen = ADSGenerator(teacher_model, proxy_student=proxy_model, device=device)
records = gen.generate(
    queries,
    config={
        "lam": 0.0,
        "tau": 0.9,
        "top_p": 0.95,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
    },
)
write_jsonl(records, holdout_path)
print({"holdout_transcript": str(holdout_path), "num_records": len(records)})
PY
  fi
  python3 -m defenses.ads.save_grad \
    --proxy_student "$PROXY_MODEL" \
    --holdout_transcript "$HOLDOUT_PATH" \
    --output_path "$GRAD_PATH" \
    --max_length "$ADS_GRAD_MAX_LENGTH" \
    --device "$DEVICE" \
    --model_dtype "$ADS_GRAD_MODEL_DTYPE" \
    --grad_dtype "$ADS_GRAD_DTYPE"
fi

if [[ -z "${DEFENSE_CONFIG:-}" ]]; then
  if [[ "$ADS_ALLOW_LAM_BATCH" == "1" || "$ADS_ALLOW_LAM_BATCH" == "true" ]]; then
    ADS_ALLOW_LAM_BATCH_JSON=true
  else
    ADS_ALLOW_LAM_BATCH_JSON=false
  fi
  DEFENSE_CONFIG="$(printf '{"lam":0.1,"eps":0.01,"tau":0.9,"top_p":0.95,"max_new_tokens":%s,"batch_size":%s,"allow_lam_batch":%s,"proxy_student":"%s","grad_path":"%s"}' "$MAX_NEW_TOKENS" "$ADS_BATCH_SIZE" "$ADS_ALLOW_LAM_BATCH_JSON" "$PROXY_MODEL" "$GRAD_PATH")"
fi

run_seqkd_defense_runner ads \
  --proxy-model "$PROXY_MODEL" \
  --ads-batch-size "$ADS_BATCH_SIZE" \
  --grad-path "$GRAD_PATH" \
  --defense-config "$DEFENSE_CONFIG" \
  --no-detector
