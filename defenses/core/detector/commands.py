from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from defenses.core.detector.result import DetectorResult
from defenses.core.io_utils import ensure_dir
from defenses.core.process_logging import run_logged_command, stage_log_path


def run_detector_command(
    *,
    repo_root: Path,
    detector: str,
    command: List[str],
    output_dir: Path,
    report_name: str = "detector_report.json",
    manifest_name: str = "detector_manifest.json",
) -> DetectorResult:
    ensure_dir(output_dir)
    log_path = stage_log_path(output_dir / f"{detector}.log")
    proc = run_logged_command(command, cwd=repo_root, log_path=log_path)
    if proc.returncode != 0:
        raise RuntimeError(f"detector {detector} failed with exit code {proc.returncode}; see {log_path}")
    report_path = output_dir / report_name
    manifest_path = output_dir / manifest_name
    return DetectorResult(
        detector=detector,
        status="ok",
        output_dir=output_dir,
        report_path=report_path if report_path.exists() else None,
        manifest_path=manifest_path if manifest_path.exists() else None,
        log_path=log_path,
    )


def watermark_detector_command(
    *,
    module: str,
    artifacts_dir: Path,
    probe_queries: Path,
    attack_manifest: Path,
    output_dir: Path,
    label: str = "positive",
    max_queries: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        module,
        "--watermark_artifacts",
        str(artifacts_dir),
        "--probe_queries",
        str(probe_queries),
        "--attack_manifest",
        str(attack_manifest),
        "--label",
        label,
        "--output_dir",
        str(output_dir),
    ]
    if max_queries is not None:
        command.extend(["--max_queries", str(max_queries)])
    if max_new_tokens is not None:
        command.extend(["--max_new_tokens", str(max_new_tokens)])
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if extra_args:
        command.extend(list(extra_args))
    return command


def adfp_detector_command(
    *,
    artifacts_dir: Path,
    transcript_path: Path,
    attack_manifest: Path,
    output_dir: Path,
    label: str = "positive",
    max_contexts: Optional[int] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "defenses.adfp.detector_run",
        "--fingerprint_artifacts",
        str(artifacts_dir),
        "--transcript",
        str(transcript_path),
        "--attack_manifest",
        str(attack_manifest),
        "--label",
        label,
        "--output_dir",
        str(output_dir),
    ]
    if max_contexts is not None:
        command.extend(["--max_contexts", str(max_contexts)])
    if extra_args:
        command.extend(list(extra_args))
    return command
