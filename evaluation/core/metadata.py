from __future__ import annotations

import datetime as dt
import hashlib
import os
import platform
import shlex
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .config import M1Config
from .clients import sanitize_base_url
from .writers import write_json, write_yaml


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def new_run_id(mode: str) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"m1_{mode}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _command_output(command: list[str], cwd: Path | None = None) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip()
    return result.stdout.strip(), None


def git_metadata(cwd: Path) -> tuple[str | None, str]:
    commit, error = _command_output(["git", "rev-parse", "HEAD"], cwd)
    if error:
        return None, "not_a_git_repository"
    status, status_error = _command_output(["git", "status", "--porcelain"], cwd)
    return commit, "unknown" if status_error else ("dirty" if status else "clean")


def package_version(name: str) -> str | None:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_environment(run_dir: Path) -> dict[str, Any]:
    output, error = _command_output([sys.executable, "-m", "pip", "freeze"])
    path = run_dir / "environment.txt"
    text = output if output is not None else f"pip freeze failed: {error}\n"
    path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return {
        "path": str(path.relative_to(run_dir)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "capture_error": error,
    }


def initial_manifest(
    config: M1Config,
    run_id: str,
    run_dir: Path,
    mode: str,
    command_line: list[str],
) -> dict[str, Any]:
    repository = Path.cwd().resolve()
    commit, dirty = git_metadata(repository)
    try:
        import torch
        cuda_version = torch.version.cuda
        torch_version = torch.__version__
    except ImportError:
        cuda_version = None
        torch_version = None
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "start_time_utc": utc_now(),
        "end_time_utc": None,
        "repository_git_commit": commit,
        "git_dirty_status": dirty,
        "hostname": socket.gethostname(),
        "working_directory": str(repository),
        "command_line": shlex.join(command_line),
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "CUDA_version": cuda_version,
        "PyTorch_version": torch_version,
        "Transformers_version": package_version("transformers"),
        "vLLM_version_or_backend_version": package_version("vllm") or package_version("openai"),
        "pip_freeze_or_conda_environment_export": write_environment(run_dir),
        "random_seed": config.generation.seed,
        "generation_settings": config.sanitized_dict().get("generation", {}),
        "retry_settings": config.sanitized_dict().get("retry", {}),
        "endpoint_settings_excluding_secrets": {
            key: {
                "base_url": sanitize_base_url(model.base_url),
                "request_model_name": model.request_model_name,
                "prompt_rendering_mode": model.prompt_rendering_mode,
                "backend": model.backend,
            }
            for key, model in config.models.items()
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    write_yaml(run_dir / "resolved_config.yaml", config.sanitized_dict())
    return manifest


def finalize_manifest(run_dir: Path, manifest: dict[str, Any], status: str) -> None:
    manifest = dict(manifest)
    manifest["end_time_utc"] = utc_now()
    manifest["status"] = status
    write_json(run_dir / "manifest.json", manifest)
