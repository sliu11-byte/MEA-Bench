#!/usr/bin/env bash

set -euo pipefail
cd "${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
export STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
OUTPUT="${ATTACK_B10000_EVAL_OUTPUT_ROOT:-${STORAGE_ROOT}/results/attack_eval/b10000}"
python3 - "${OUTPUT}" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected = ("seqkd", "qedks", "model_leeching", "lord", "soda", "gad")
smoke = bool(os.environ.get("EVAL_LIMIT"))
rows = []
for attack in expected:
    path = output / f"summary_{attack}.json"
    if not path.is_file():
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (len(payload) != 1 or payload[0].get("attack") != attack
            or bool(payload[0].get("smoke")) != smoke):
        raise SystemExit(f"Worker summary protocol mismatch: {path}")
    rows.append(payload[0])
if not rows:
    raise SystemExit(f"No completed attack summaries under {output}")
if len(rows) != len(expected):
    (output / "summary.json").unlink(missing_ok=True)
    (output / "summary.csv").unlink(missing_ok=True)
for suffix in ("partial", "") if len(rows) == len(expected) else ("partial",):
    stem = "summary" if not suffix else f"summary_{suffix}"
    (output / f"{stem}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / f"{stem}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
status = {"complete": len(rows) == len(expected), "smoke": smoke,
          "completed_attacks": [row["attack"] for row in rows],
          "missing_attacks": [attack for attack in expected if attack not in {row["attack"] for row in rows}],
          "summary": "summary.csv" if len(rows) == len(expected) else "summary_partial.csv"}
(output / "partial_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
print(json.dumps(status, indent=2))
PY
