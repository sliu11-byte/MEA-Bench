from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from defenses.core.io_utils import ensure_dir
from defenses.core.process_logging import run_logged_command, stage_log_path


def bool_flag(name: str, value: Optional[bool]) -> List[str]:
    if value is None:
        return []
    flag = "--" + name.replace("_", "-")
    return [flag if value else "--no-" + name.replace("_", "-")]


def find_latest_attack_manifest(output_dir: Path, attack: str) -> Optional[Path]:
    root = output_dir / attack
    if not root.exists():
        return None
    manifests = sorted(root.glob("*/attack_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0] if manifests else None


def checkpoint_from_attack_manifest(path: Optional[Path]) -> Optional[Path]:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result") or {}
    checkpoint = result.get("checkpoint_dir") or payload.get("checkpoint_dir")
    return None if not checkpoint else Path(checkpoint)


def run_attack_process(
    *,
    repo_root: Path,
    attack: str,
    budget: int,
    teacher_base_url: Optional[str],
    served_model_name: str,
    teacher_model: Optional[str],
    student_model: Optional[str],
    output_dir: Path,
    query_pool: str = "auto",
    query_ordering: str = "auto",
    teacher_mode: str = "chat",
    teacher_temperature: float = 0.0,
    teacher_top_p: float = 1.0,
    teacher_max_tokens: int = 1536,
    stage1_config: Optional[str] = None,
    dry_run: bool = False,
    extra_args: Optional[Iterable[str]] = None,
    countermeasure: Optional[str] = None,
    countermeasure_manifest: Optional[Path] = None,
    teacher_transcript_path: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    log_path = stage_log_path(output_dir / f"{attack}_attack.log")
    command = [
        sys.executable,
        "attacks/scripts/run_attack.py",
        "--attack",
        attack,
        "--budget",
        str(budget),
        "--query-pool",
        query_pool,
        "--query-ordering",
        query_ordering,
        "--output-dir",
        str(output_dir),
    ]
    if teacher_transcript_path is not None:
        command.extend(["--teacher-transcript-path", str(teacher_transcript_path)])
    else:
        if teacher_base_url is None:
            raise ValueError("teacher_base_url is required unless teacher_transcript_path is provided")
        command.extend(
            [
                "--teacher-backend",
                "vllm_openai",
                "--teacher-endpoint-url",
                teacher_base_url,
                "--teacher-request-model",
                served_model_name,
                "--teacher-mode",
                teacher_mode,
                "--teacher-temperature",
                str(teacher_temperature),
                "--teacher-top-p",
                str(teacher_top_p),
                "--teacher-max-tokens",
                str(teacher_max_tokens),
            ]
        )
    if teacher_model:
        command.extend(["--teacher-model", teacher_model])
    if student_model:
        command.extend(["--student-model", student_model])
    if stage1_config:
        command.extend(["--stage1-config", stage1_config])
    if dry_run:
        command.append("--dry-run")
    if countermeasure:
        command.extend(["--countermeasure", countermeasure])
    if countermeasure_manifest:
        command.extend(["--countermeasure-manifest", str(countermeasure_manifest)])
    if extra_args:
        command.extend(list(extra_args))

    proc = run_logged_command(command, cwd=repo_root, log_path=log_path, env=extra_env)
    if proc.returncode != 0:
        raise RuntimeError(f"attack failed with exit code {proc.returncode}; see {log_path}")
    manifest = find_latest_attack_manifest(output_dir, attack)
    return {
        "command": command,
        "log_path": str(log_path.resolve()),
        "manifest_path": None if manifest is None else str(manifest.resolve()),
        "checkpoint_path": None if manifest is None else str(checkpoint_from_attack_manifest(manifest).resolve()) if checkpoint_from_attack_manifest(manifest) else None,
    }
