"""Shared helpers for teacher-query traffic detectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from defenses.core.io_utils import ensure_dir, read_jsonl, write_jsonl

QUERY_FIELDS = (
    "teacher_query",
    "rendered_prompt",
    "rendered_query",
    "templated_prompt",
    "template_prompt",
    "prompt",
    "prompt_text",
    "query",
    "logical_prompt",
    "instruction",
    "question",
    "input",
    "text",
)


def pick_query(record: dict[str, Any]) -> str | None:
    for key in QUERY_FIELDS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = record.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"].strip())
        if parts:
            return "\n".join(part for part in parts if part)
    request = record.get("request")
    if isinstance(request, dict):
        return pick_query(request)
    return None


def normalize_teacher_query_log(
    source_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Extract the actual teacher-received query text into {id, query} JSONL."""
    source = Path(source_path)
    records = read_jsonl(source)
    out_records = []
    skipped = 0
    for i, record in enumerate(records):
        text = pick_query(record)
        if text is None:
            skipped += 1
            continue
        qid = (
            record.get("query_id")
            or record.get("id")
            or record.get("idx")
            or record.get("example_id")
            or f"teacher_query_{i:06d}"
        )
        out_records.append(
            {
                "id": str(qid),
                "query_id": str(qid),
                "query": text,
                "source_index": i,
                "source_path": str(source.resolve()),
            }
        )
        if limit is not None and len(out_records) >= limit:
            break
    if not out_records:
        raise ValueError(f"No actual teacher queries could be extracted from {source}")
    out = Path(output_path)
    ensure_dir(out.parent)
    write_jsonl(out_records, out)
    return {
        "path": str(out.resolve()),
        "source_path": str(source.resolve()),
        "num_queries": len(out_records),
        "skipped_records": skipped,
        "query_fields_tried": list(QUERY_FIELDS),
    }


def resolve_teacher_query_log(
    *,
    teacher_query_log: str | None = None,
    attack_manifest: str | None = None,
) -> str:
    if teacher_query_log:
        return teacher_query_log
    if not attack_manifest:
        raise ValueError("Pass --teacher_query_log or --attack_manifest.")

    manifest_path = Path(attack_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: list[Any] = []
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    teacher_transcript = artifacts.get("teacher_transcript")
    if isinstance(teacher_transcript, dict):
        candidates.extend(
            [
                teacher_transcript.get("source_path"),
                teacher_transcript.get("transcript"),
                teacher_transcript.get("jsonl"),
            ]
        )
        bundle = teacher_transcript.get("bundle_dir")
        if bundle:
            candidates.extend(
                [
                    Path(str(bundle)) / "transcript.jsonl",
                    Path(str(bundle)) / "teacher_raw.jsonl",
                ]
            )
    candidates.extend(
        [
            artifacts.get("teacher_raw"),
            artifacts.get("teacher_transcript"),
            result.get("teacher_transcript_path"),
            manifest.get("teacher_transcript_path"),
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            return str(path)
        if not path.is_absolute():
            rel = manifest_path.parent / path
            if rel.exists():
                return str(rel)
    raise FileNotFoundError(
        f"Could not infer teacher query log from attack manifest: {manifest_path}. "
        "Pass --teacher_query_log explicitly."
    )
