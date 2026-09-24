from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from defenses.core.io_utils import ensure_dir
from defenses.core.process_logging import stage_log_path, start_logged_process


@dataclass(frozen=True)
class OracleProcess:
    process: subprocess.Popen
    base_url: str
    output_dir: Path
    manifest_path: Path
    query_log_path: Path
    transcript_path: Path
    artifacts_dir: Path
    log_path: Path

    def terminate(self, timeout: float = 20.0) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)


def wait_for_oracle(base_url: str, process: subprocess.Popen, *, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"defended teacher server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"defended teacher did not become ready at {base_url}: {last_error}")


def start_oracle(
    *,
    repo_root: Path,
    defense: str,
    output_dir: Path,
    host: str,
    port: int,
    served_model_name: str,
    teacher_model: Optional[str],
    defense_config: Optional[str] = None,
    device: Optional[str] = None,
    teacher_base_url: Optional[str] = None,
    teacher_request_model: Optional[str] = None,
    teacher_api_key: str = "EMPTY",
    proxy_model: Optional[str] = None,
    grad_path: Optional[str] = None,
    rewriter_model: Optional[str] = None,
    rewriter_backend: str = "local_hf",
    rewriter_base_url: Optional[str] = None,
    rewriter_request_model: Optional[str] = None,
    rewriter_api_key: str = "EMPTY",
    doge_checkpoint: Optional[str] = None,
    run_id: Optional[str] = None,
    startup_timeout: float = 300.0,
    extra_args: Optional[List[str]] = None,
) -> OracleProcess:
    output_dir = ensure_dir(output_dir)
    artifacts_dir = ensure_dir(output_dir / "artifacts")
    log_path = stage_log_path(output_dir / "oracle_server.log")
    command = [
        sys.executable,
        "-m",
        "defenses.oracle.serve_defended_teacher",
        "--defense",
        defense,
        "--host",
        host,
        "--port",
        str(port),
        "--output-dir",
        str(output_dir),
        "--served-model-name",
        served_model_name,
    ]
    optional_pairs = [
        ("--run-id", run_id),
        ("--defense-config", defense_config),
        ("--device", device),
        ("--teacher-model", teacher_model),
        ("--teacher-base-url", teacher_base_url),
        ("--teacher-request-model", teacher_request_model),
        ("--teacher-api-key", teacher_api_key),
        ("--proxy-model", proxy_model),
        ("--grad-path", grad_path),
        ("--rewriter-model", rewriter_model),
        ("--rewriter-backend", rewriter_backend),
        ("--rewriter-base-url", rewriter_base_url),
        ("--rewriter-request-model", rewriter_request_model),
        ("--rewriter-api-key", rewriter_api_key),
        ("--doge-checkpoint", doge_checkpoint),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            command.extend([flag, str(value)])
    if extra_args:
        command.extend(extra_args)

    process = start_logged_process(command, cwd=repo_root, log_path=log_path)
    base_url = f"http://{host}:{port}/v1"
    try:
        wait_for_oracle(base_url, process, timeout=startup_timeout)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    return OracleProcess(
        process=process,
        base_url=base_url,
        output_dir=output_dir,
        manifest_path=output_dir / "oracle_manifest.json",
        query_log_path=output_dir / "teacher_received_queries.jsonl",
        transcript_path=output_dir / "defended_teacher_transcript.jsonl",
        artifacts_dir=artifacts_dir,
        log_path=log_path,
    )


def load_oracle_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
