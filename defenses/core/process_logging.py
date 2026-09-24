"""Mirror stage output into the parent job log and group detailed logs by job."""

import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path


def stage_log_path(path: Path) -> Path:
    directory = os.environ.get("RUN_LOG_DIR")
    if not directory:
        return path
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
    return Path(directory) / f"{path.stem}_{digest}{path.suffix}"


def start_logged_process(command, *, cwd: Path, log_path: Path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    header = f"[stage:{log_path.stem}] $ " + " ".join(command) + "\n"
    log.write(header)
    log.flush()
    print(header, end="", flush=True)
    try:
        process = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"})
    except BaseException:
        log.close()
        raise

    def mirror():
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(f"[{log_path.stem}] {line}")
                sys.stdout.flush()
        finally:
            process.stdout.close()
            log.close()

    thread = threading.Thread(target=mirror, daemon=True)
    process.log_thread = thread
    thread.start()
    return process


def run_logged_command(command, *, cwd: Path, log_path: Path, env=None):
    process = start_logged_process(command, cwd=cwd, log_path=log_path, env=env)
    try:
        code = process.wait()
        process.log_thread.join()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.log_thread.join(timeout=20)
        raise
    return subprocess.CompletedProcess(command, code)
