from __future__ import annotations

from typing import Any


REQUIRED_FIELDS = {
    "run_id", "timestamp_utc", "model_key", "model_id", "model_role", "model_revision",
    "tokenizer_name", "tokenizer_revision", "backend", "endpoint_base_url", "dataset",
    "dataset_source", "dataset_config", "dataset_revision", "split", "example_id",
    "example_index", "prompt_template_id", "prompt_template_version",
    "prompt_rendering_mode", "logical_prompt", "rendered_prompt", "generation_config",
    "raw_response", "parsed_prediction", "normalized_prediction", "gold_answer",
    "normalized_gold_answer", "parse_status", "parser_name", "correct", "request_status",
    "retry_count", "error_type", "error_message", "latency_seconds", "input_tokens",
    "output_tokens", "total_tokens", "split_fallback_used", "requested_split",
}

REQUIRED_GENERATION_FIELDS = {
    "temperature", "top_p", "top_k", "max_tokens", "seed", "repetition_penalty",
    "stop_sequences",
}

HUMANEVAL_FIELDS = {
    "task_id", "canonical_prompt", "generated_completion", "execution_enabled",
    "execution_status", "test_result", "error_trace_if_any",
}


class SchemaValidationError(ValueError):
    pass


def validate_prediction_record(record: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise SchemaValidationError(f"Prediction record missing fields: {sorted(missing)}")
    generation = record.get("generation_config")
    if not isinstance(generation, dict):
        raise SchemaValidationError("generation_config must be an object")
    generation_missing = REQUIRED_GENERATION_FIELDS - set(generation)
    if generation_missing:
        raise SchemaValidationError(
            f"generation_config missing fields: {sorted(generation_missing)}"
        )
    if record["request_status"] not in {"completed", "failed", "dry_run"}:
        raise SchemaValidationError(f"Invalid request_status: {record['request_status']!r}")
    if record["parse_status"] not in {"success", "failed", "not_applicable", "not_attempted"}:
        raise SchemaValidationError(f"Invalid parse_status: {record['parse_status']!r}")
    if record["parse_status"] == "failed" and record["parsed_prediction"] is not None:
        raise SchemaValidationError("Failed parsing must have parsed_prediction=null")
    if record["dataset"] == "humaneval":
        missing_humaneval = HUMANEVAL_FIELDS - set(record)
        if missing_humaneval:
            raise SchemaValidationError(
                f"HumanEval record missing fields: {sorted(missing_humaneval)}"
            )


def completed_record_key(record: dict[str, Any]) -> tuple[str, str, str, str, str] | None:
    try:
        validate_prediction_record(record)
    except SchemaValidationError:
        return None
    if record.get("request_status") != "completed":
        return None
    return (
        str(record["run_id"]),
        str(record["model_key"]),
        str(record["dataset"]),
        str(record["split"]),
        str(record["example_id"]),
    )

