from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .clients import OpenAICompatibleClient, sanitize_base_url
from .config import ModelSpec, RetrySpec


def _hub_cache_repo(model_id: str) -> Path:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return cache_root / f"models--{model_id.replace('/', '--')}"


def _local_snapshot(model_id: str, requested_revision: str | None) -> tuple[Path | None, str | None]:
    repo = _hub_cache_repo(model_id)
    if not repo.exists():
        return None, requested_revision
    if requested_revision:
        candidate = repo / "snapshots" / requested_revision
        if candidate.exists():
            return candidate, requested_revision
    ref = repo / "refs" / "main"
    revision = ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    if revision and (repo / "snapshots" / revision).exists():
        return repo / "snapshots" / revision, revision
    snapshots = sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").exists() else []
    if len(snapshots) == 1:
        return snapshots[0], snapshots[0].name
    return None, requested_revision


def _read_json(path: Path | None, filename: str) -> dict[str, Any] | None:
    if path is None:
        return None
    target = path / filename
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _gpu_metadata() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {"GPU_model": None, "GPU_count": None, "GPU_memory_if_available": None}
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    models, memory = [], []
    for row in rows:
        name, _, total = row.rpartition(",")
        models.append(name.strip())
        memory.append(total.strip())
    return {
        "GPU_model": models,
        "GPU_count": len(rows),
        "GPU_memory_if_available": memory,
    }


def collect_model_metadata(model: ModelSpec, retry: RetrySpec) -> dict[str, Any]:
    snapshot, inferred_revision = _local_snapshot(model.model_id, model.model_revision)
    model_config = _read_json(snapshot, "config.json")
    tokenizer_config = _read_json(snapshot, "tokenizer_config.json")
    tokenizer_name = model.tokenizer_name or model.model_id
    tokenizer_revision = model.tokenizer_revision or inferred_revision
    chat_template = (tokenizer_config or {}).get("chat_template")
    chat_template_hash = None
    if chat_template:
        chat_template_hash = hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest()

    endpoint_models: list[str] = []
    endpoint_error = None
    if model.base_url:
        try:
            endpoint_models = OpenAICompatibleClient(model, retry).list_models()
        except Exception as exc:
            endpoint_error = f"{type(exc).__name__}: {exc}"

    metadata = {
        "model_key": model.key,
        "model_id": model.model_id,
        "request_model_name": model.request_model_name,
        "role": model.role,
        "family": model.family,
        "declared_parameter_size": model.size,
        "checkpoint_type": model.checkpoint_type,
        "Hugging_Face_revision_or_commit_hash_if_available": inferred_revision,
        "model_config": model_config,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision_or_commit_hash_if_available": tokenizer_revision,
        "tokenizer_config": tokenizer_config,
        "chat_template_or_template_hash_if_used": chat_template_hash,
        "prompt_rendering_mode": model.prompt_rendering_mode,
        "backend": model.backend,
        "backend_model_name": endpoint_models,
        "backend_metadata_error": endpoint_error,
        "dtype": model.dtype,
        "quantization_setting": model.quantization_setting,
        "tensor_parallel_size": model.tensor_parallel_size,
        "endpoint_base_url_without_credentials": sanitize_base_url(model.base_url),
    }
    metadata.update(_gpu_metadata())
    return metadata

