from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable


PROMPT_FIELDS = ("prompt", "prompt_text", "rendered_prompt", "logical_prompt", "instruction", "question")
RESPONSE_FIELDS = (
    "teacher_response",
    "student_response",
    "response",
    "output",
    "raw_response",
    "generated_completion",
)
ID_FIELDS = ("prompt_id", "example_id", "id", "task_id")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def first_present(record: dict[str, Any], fields: Iterable[str]) -> Any:
    for field in fields:
        value = record.get(field)
        if value is not None and value != "":
            return value
    return None


def record_id(record: dict[str, Any], fallback_index: int | None = None) -> str:
    value = first_present(record, ID_FIELDS)
    if value is not None:
        return str(value)
    if fallback_index is not None:
        return f"row_{fallback_index:08d}"
    raise ValueError("Record has no id field and no fallback index was provided.")


def record_prompt(record: dict[str, Any]) -> str:
    value = first_present(record, PROMPT_FIELDS)
    if value is None:
        raise ValueError(f"Record has no prompt field. Tried: {PROMPT_FIELDS}")
    return str(value)


def record_response(record: dict[str, Any], preferred_fields: Iterable[str] | None = None) -> str:
    fields = tuple(preferred_fields or ()) + RESPONSE_FIELDS
    value = first_present(record, fields)
    if value is None:
        raise ValueError(f"Record has no response field. Tried: {fields}")
    return str(value)


def now_run_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"


