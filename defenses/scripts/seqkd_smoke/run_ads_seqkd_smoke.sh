#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ADS_WORK_DIR="${ADS_WORK_DIR:-$OUTPUT_ROOT/ads_prep}"
ADS_HOLDOUT="${ADS_HOLDOUT:-4}"
mkdir -p "$ADS_WORK_DIR"
GRAD_PATH="${GRAD_PATH:-$ADS_WORK_DIR/student_grads.pt}"
if [[ ! -f "$GRAD_PATH" ]]; then
  python3 - <<PY
from pathlib import Path
from attacks.core.hf_query_pool import hf_query_pool_for_budget, resolve_query_pool_and_ordering
from defenses.ads.generator import ADSGenerator
from defenses.ads.save_grad import compute_proxy_student_grads
from defenses.core.io_utils import load_queries, write_jsonl
budget = int("$BUDGET")
holdout = int("$ADS_HOLDOUT")
query_pool = "$QUERY_POOL"
query_pool_path, _ = resolve_query_pool_and_ordering(hf_query_pool_for_budget(budget + holdout) if query_pool == "auto" else query_pool, "auto")
queries = load_queries(query_pool_path, max_queries=budget + holdout)[budget:budget + holdout]
if len(queries) < holdout:
    raise SystemExit(f"Need {holdout} holdout queries, got {len(queries)}")
out_dir = Path("$ADS_WORK_DIR")
out_dir.mkdir(parents=True, exist_ok=True)
gen = ADSGenerator("$TEACHER_MODEL", proxy_student="$PROXY_MODEL", device="$DEVICE")
records = gen.generate(queries, config={"lam": 0.0, "tau": 0.7, "max_new_tokens": int("$MAX_NEW_TOKENS")})
write_jsonl(records, out_dir / "holdout_teacher.jsonl")
manifest = compute_proxy_student_grads("$PROXY_MODEL", records, output_path=Path("$GRAD_PATH"), max_length=256, device="$DEVICE")
print(manifest)
PY
fi
if [[ -z "${DEFENSE_CONFIG:-}" ]]; then
  DEFENSE_CONFIG="$(printf '{"lam":0.1,"eps":0.01,"tau":0.7,"top_p":0.95,"max_new_tokens":%s,"proxy_student":"%s","grad_path":"%s"}' "$MAX_NEW_TOKENS" "$PROXY_MODEL" "$GRAD_PATH")"
fi
run_seqkd_defense_runner ads \
  --proxy-model "$PROXY_MODEL" \
  --grad-path "$GRAD_PATH" \
  --defense-config "$DEFENSE_CONFIG" \
  --no-detector
