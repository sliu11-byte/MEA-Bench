from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from attacks.core.manifest import stable_hash, write_json


SCHEMA_VERSION = "mea_experiment_manifest_v1"


def make_run_id(method: str, budget: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_{method}_b{budget}"


def portable_path(path: Path, *, repo_root: Path, output_root: Path | None = None) -> str:
    resolved = path.expanduser().resolve()
    roots = ((repo_root, Path(".")),)
    if output_root is not None:
        roots = ((output_root, Path("outputs")),) + roots
    for root, prefix in roots:
        try:
            relative = resolved.relative_to(root.resolve())
            return (prefix / relative).as_posix()
        except ValueError:
            continue
    return resolved.name


def git_source(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def software_versions() -> dict[str, str | None]:
    packages = ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "vllm")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def find_resumable_manifest(run_root: Path, config_hash: str) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in run_root.glob("*/attack_manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("config_hash") == config_hash and payload.get("status") in {
            "planned",
            "running",
            "completed",
        }:
            candidates.append((path, payload))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0].stat().st_mtime)


def write_experiment_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema must be {SCHEMA_VERSION}")
    return write_json(path, payload)


__all__ = [
    "SCHEMA_VERSION",
    "find_resumable_manifest",
    "git_source",
    "make_run_id",
    "portable_path",
    "software_versions",
    "stable_hash",
    "write_experiment_manifest",
]
