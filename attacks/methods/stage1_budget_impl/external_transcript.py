from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .stage1_transcript import (
    DecodeConfig,
    PromptOrdering,
    QueryAttempt,
    QueryLedger,
    TranscriptBudgetError,
    TranscriptRecord,
    TranscriptSchemaError,
    save_transcript_bundle,
    stable_hash,
)

QUERY_ID_KEYS = ("query_id", "id", "qid", "uid", "example_id", "idx")
QUERY_TEXT_KEYS = ("query", "prompt", "rendered_prompt", "logical_prompt", "instruction", "question", "problem", "text", "input")
RESPONSE_KEYS = ("response", "teacher_response", "output", "completion", "answer", "final_answer")


def _pick(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _read_external_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TranscriptSchemaError(f"invalid external JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise TranscriptSchemaError(f"external transcript line {line_number} is not a JSON object")
            records.append(dict(payload))
    return records


def _normalize_external_record(raw: Mapping[str, Any], index: int) -> tuple[str, str, str]:
    query_id = _pick(raw, QUERY_ID_KEYS)
    query_text = _pick(raw, QUERY_TEXT_KEYS)
    response = _pick(raw, RESPONSE_KEYS)

    if query_id is None:
        query_id = f"external_{index:06d}"
    if query_text is None:
        raise TranscriptSchemaError(
            f"external transcript record {index} has no query field; tried {QUERY_TEXT_KEYS}"
        )
    if response is None:
        raise TranscriptSchemaError(
            f"external transcript record {index} has no response field; tried {RESPONSE_KEYS}"
        )
    return str(query_id), str(query_text), str(response)


def import_external_teacher_transcript(
    transcript_path: Path,
    bundle_root: Path,
    *,
    budget: int,
    teacher_model_id: str | None = None,
    seed: int | None = None,
    prompt_pool_id: str = "external_teacher_transcript",
) -> dict[str, Any]:
    """Convert a simple query/response JSONL into the stage1 transcript bundle format."""
    source = transcript_path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"external teacher transcript does not exist: {source}")
    if budget <= 0:
        raise TranscriptBudgetError("budget must be positive")

    raw_records = _read_external_jsonl(source)
    if budget > len(raw_records):
        raise TranscriptBudgetError(
            f"budget={budget} exceeds external transcript records={len(raw_records)} in {source}"
        )

    seed_value = 0 if seed is None else int(seed)
    decode_config = DecodeConfig(
        strategy="external_jsonl",
        settings={"source_path": str(source), "format": "query_response_jsonl"},
    )
    teacher_id = teacher_model_id or "external_teacher_transcript"
    selected = raw_records[:budget]
    normalized = [_normalize_external_record(record, idx) for idx, record in enumerate(selected)]
    prompt_pool_hash = stable_hash([(query_id, query) for query_id, query, _ in normalized])
    ordered_prompt_ids = tuple(query_id for query_id, _, _ in normalized)
    ordered_indices = tuple(range(len(normalized)))
    ordering_hash = stable_hash(
        {
            "prompt_pool_id": prompt_pool_id,
            "seed": seed_value,
            "prompt_pool_hash": prompt_pool_hash,
            "ordered_prompt_ids": ordered_prompt_ids,
        }
    )
    ordering = PromptOrdering(
        prompt_pool_id=prompt_pool_id,
        seed=seed_value,
        prompt_pool_hash=prompt_pool_hash,
        ordering_hash=ordering_hash,
        ordered_prompt_ids=ordered_prompt_ids,
        ordered_indices=ordered_indices,
    )

    records: list[TranscriptRecord] = []
    ledger = QueryLedger()
    created_at = "external"
    for order_index, (query_id, query, response) in enumerate(normalized):
        record = TranscriptRecord(
            prompt_id=query_id,
            prompt_text=query,
            teacher_response=response,
            teacher_model_id=teacher_id,
            decode_config={"strategy": decode_config.strategy, "settings": dict(decode_config.settings)},
            seed=seed_value,
            order_index=order_index,
            query_id=query_id,
            query_status="success",
            created_at=created_at,
        )
        records.append(record)
        ledger.record_success(
            QueryAttempt(
                query_id=query_id,
                prompt_id=query_id,
                order_index=order_index,
                attempt_index=0,
                status="success",
                created_at=created_at,
            )
        )

    bundle_dir = bundle_root / prompt_pool_id / teacher_id / stable_hash(
        {
            "source": str(source),
            "budget": budget,
            "prompt_pool_hash": prompt_pool_hash,
            "teacher_model_id": teacher_id,
        }
    )[:24]
    manifest = save_transcript_bundle(
        bundle_dir=bundle_dir,
        prompt_pool_id=prompt_pool_id,
        ordering=ordering,
        records=records,
        teacher_model_id=teacher_id,
        decode_config={"strategy": decode_config.strategy, "settings": dict(decode_config.settings)},
        seed=seed_value,
        requested_budget=budget,
        ledger=ledger,
        prompt_pool_hash=prompt_pool_hash,
    )
    return {
        "status": "imported_external_jsonl",
        "source_path": str(source),
        "bundle_dir": str(bundle_dir),
        "manifest_path": str(bundle_dir / "manifest.json"),
        "transcript_path": str(bundle_dir / "transcript.jsonl"),
        "ordering_path": str(bundle_dir / "ordering.json"),
        "ledger_path": str(bundle_dir / "ledger.json"),
        "budget": budget,
        "actual_successful_queries": len(records),
        "teacher_model_id": teacher_id,
        "bundle_id": manifest.bundle_id,
    }