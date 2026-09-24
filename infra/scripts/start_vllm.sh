#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_KEY="${1:?Usage: scripts/start_vllm.sh <model-key> [extra vLLM args...]}"
shift || true

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
if [[ -z "${CUDA_HOME:-}" ]]; then
  PY_CUDA_HOME="$(python - <<'PY'
from pathlib import Path
import sys

prefix = Path(sys.prefix)
candidates = sorted((prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia").glob("cu*/bin/nvcc"))
if candidates:
    print(candidates[-1].parents[1])
PY
)"
  if [[ -n "${PY_CUDA_HOME}" ]]; then
    export CUDA_HOME="${PY_CUDA_HOME}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
  fi
fi

CONFIG_JSON="$(python - "${VLLM_MODELS_CONFIG:-$ROOT/infra/configs/models.json}" "$MODEL_KEY" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    models = json.load(f)
if key not in models:
    raise SystemExit(f"Unknown model key: {key}")
print(json.dumps(models[key]))
PY
)"

read_json() {
  python - "$CONFIG_JSON" "$1" <<'PY'
import json
import sys

spec = json.loads(sys.argv[1])
print(spec[sys.argv[2]])
PY
}

MODEL_ID="$(read_json model)"
SERVED_NAME="$(read_json served_model_name)"
PORT="$(read_json port)"
TP="$(read_json tensor_parallel_size)"
DTYPE="$(read_json dtype)"
MAX_MODEL_LEN="$(read_json max_model_len)"
GPU_UTIL="$(read_json gpu_memory_utilization)"

echo "[vLLM] key=${MODEL_KEY}"
echo "[vLLM] model=${MODEL_ID}"
echo "[vLLM] served_model_name=${SERVED_NAME}"
echo "[vLLM] port=${PORT} tensor_parallel_size=${TP} dtype=${DTYPE}"
echo "[vLLM] CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}"
echo "[vLLM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "[vLLM] CUDA_HOME=${CUDA_HOME:-<not set>}"

exec python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model "${MODEL_ID}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size "${TP}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --trust-remote-code \
  "$@"

