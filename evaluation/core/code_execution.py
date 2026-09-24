from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    status: str
    test_result: str | None
    error_trace: str | None


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    one_gibibyte = 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (one_gibibyte, one_gibibyte))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def execute_humaneval(
    canonical_prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    timeout_seconds: float = 10.0,
) -> ExecutionResult:
    source = (
        canonical_prompt.rstrip()
        + "\n"
        + completion.rstrip()
        + "\n\n"
        + test_code.rstrip()
        + f"\n\ncheck({entry_point})\n"
    )
    with tempfile.TemporaryDirectory(prefix="m1_humaneval_") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={"PATH": os.environ.get("PATH", "")},
                preexec_fn=_limits,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult("timeout", "failed", str(exc))
        except Exception as exc:
            return ExecutionResult("execution_error", "failed", f"{type(exc).__name__}: {exc}")
        if result.returncode == 0:
            return ExecutionResult("passed", "passed", None)
        trace = (result.stderr or result.stdout or "candidate exited non-zero").strip()
        return ExecutionResult("failed", "failed", trace[-10000:])

