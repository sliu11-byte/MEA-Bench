#!/usr/bin/env bash

set -euo pipefail
cd "${REPO_DIR:-${REPO_BASE_DIR:-$(pwd)}}"
[[ -f attacks/methods/soda_impl/train_dpo.py ]] || { echo 'Submit from repository root.' >&2; exit 2; }
mkdir -p logs
command -v module >/dev/null 2>&1 && module purge || true
command -v module >/dev/null 2>&1 && module load conda/25.7.0 cuda/12.4.1 || true
command -v conda >/dev/null 2>&1 && conda activate "${CONDA_ENV:-research}" || true
export STORAGE_ROOT="${STORAGE_ROOT:-/path/to/storage/${USER}/A-Benchmark-for-Model-distillation-survey}"
export HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

OUT="${SODA_CLEAN_OUTPUT:-${STORAGE_ROOT}/outputs/defenses/soda_b1000_seqkd_warmup/clean}"
SOURCE="${OUT}/soda_clean_sources.json"
PREP=(--storage-root "${STORAGE_ROOT}" --output "${SOURCE}")
[[ -n "${SODA_CLEAN_SEQKD_CHECKPOINT:-}" ]] && PREP+=(--seqkd-checkpoint "${SODA_CLEAN_SEQKD_CHECKPOINT}")
[[ -n "${SODA_CLEAN_PREFERENCES:-}" ]] && PREP+=(--preferences "${SODA_CLEAN_PREFERENCES}")
python3 runs/defense/prepare_soda_clean_reuse.py "${PREP[@]}"

readarray -t INPUTS < <(python3 - "${SOURCE}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(d["warmup_checkpoint"])
print(d["preferences"])
PY
)
CHECKPOINT="${OUT}/checkpoint-final"
REUSE_COMPLETE="$(python3 - "${OUT}/complete.json" "${CHECKPOINT}" "${SOURCE}" <<'PY'
import json, sys
from pathlib import Path
from attacks.core.checkpoint_health import checkpoint_tensor_health
complete, checkpoint, source = map(Path, sys.argv[1:])
if not complete.is_file() or not (checkpoint / "adapter_config.json").is_file() or not (checkpoint / "training_safety.json").is_file():
    print("no")
else:
    old = json.loads(complete.read_text(encoding="utf-8"))
    new = json.loads(source.read_text(encoding="utf-8"))
    if old.get("source") != new:
        raise SystemExit("Existing clean SODA output uses different inputs; set SODA_CLEAN_OUTPUT to a new directory")
    health = checkpoint_tensor_health(checkpoint)
    print("yes" if health["all_finite"] else "no")
PY
)"
if [[ "${REUSE_COMPLETE}" != "yes" ]]; then
  python3 -m attacks.methods.soda_impl.train_dpo \
    --preference-jsonl "${INPUTS[1]}" --warmup-model "${INPUTS[0]}" --output-dir "${CHECKPOINT}" \
    --beta 0.1 --learning-rate 5e-6 --epochs 1.0 --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 32 --max-length 3544 --max-prompt-length 1024 \
    --bf16 --use-lora --gradient-checkpointing --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --max-grad-norm "${SODA_MAX_GRAD_NORM:-1.0}" \
    --nonfinite-gradient-retries "${SODA_NONFINITE_GRADIENT_RETRIES:-3}"
fi
python3 - "${OUT}" "${CHECKPOINT}" "${SOURCE}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from attacks.core.checkpoint_health import checkpoint_tensor_health
out, checkpoint, source = map(Path, sys.argv[1:])
health = checkpoint_tensor_health(checkpoint)
if not health["all_finite"]:
    raise SystemExit(f"Refusing to mark non-finite SODA checkpoint complete: {health['nonfinite_tensors'][:8]}")
training_safety = checkpoint / "training_safety.json"
if not training_safety.is_file():
    raise SystemExit(f"Missing SODA training safety report: {training_safety}")
existing = out / "attack_manifest.json"
created_at = None
if existing.is_file():
    created_at = json.loads(existing.read_text(encoding="utf-8")).get("created_at")
payload = {
    "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    "run_config": {"attack": "soda", "budget": 1000, "student_model": "Qwen/Qwen2.5-7B"},
    "result": {"attack": "soda", "budget": 1000, "run_id": "soda_b1000_clean_seqkd_warmup",
               "status": "completed", "checkpoint_dir": str(checkpoint.resolve()),
               "artifacts": {"sources": str(source.resolve()),
                             "training_safety": str(training_safety.resolve()),
                             "checkpoint_health": health}},
}
(out / "attack_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
sources = json.loads(source.read_text(encoding="utf-8"))
(out / "complete.json").write_text(json.dumps({"complete": True, "checkpoint": str(checkpoint.resolve()), "source": sources, "checkpoint_health": health}, indent=2) + "\n", encoding="utf-8")
PY
echo "Corrected clean SODA complete: ${CHECKPOINT}"
