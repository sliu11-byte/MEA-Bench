#!/usr/bin/env bash

set -euo pipefail
cd "${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
export STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  OUTPUT="${ATTACK_B1000_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b1000_smoke_${EVAL_LIMIT}}"
else
  OUTPUT="${ATTACK_B1000_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b1000}"
fi
python3 - "${OUTPUT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
attacks = ("seqkd", "qedks", "model_leeching", "lord", "soda", "gad")
rows = []
for attack in attacks:
    path = output / f"summary_{attack}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload) != 1 or payload[0].get("attack") != attack:
        raise SystemExit(f"Invalid worker summary: {path}")
    rows.append(payload[0])
(output / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Merged {len(rows)} attack summaries -> {output / 'summary.csv'}")
PY
