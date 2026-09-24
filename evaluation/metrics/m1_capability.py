from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from evaluation.core.writers import write_csv


DATASET_ORDER = [
    "arc_challenge", "hellaswag", "mmlu", "truthfulqa", "winogrande", "gsm8k",
    "medqa", "pubmedqa", "chemprot", "fomc", "headline", "fpb", "humaneval",
]


def latest_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("model_key")),
            str(record.get("dataset")),
            str(record.get("split")),
            str(record.get("example_id")),
        )
        previous = latest.get(key)
        if previous is None or str(record.get("timestamp_utc", "")) >= str(previous.get("timestamp_utc", "")):
            latest[key] = record
    return list(latest.values())


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    request_failures = sum(item.get("request_status") == "failed" for item in records)
    parse_failures = sum(item.get("parse_status") == "failed" for item in records)
    completed = sum(item.get("request_status") == "completed" for item in records)
    correct = sum(item.get("correct") is True for item in records)
    latencies = [float(item.get("latency_seconds") or 0.0) for item in records]
    is_humaneval = bool(records and records[0].get("dataset") == "humaneval")
    execution_enabled = any(item.get("execution_enabled") for item in records) if is_humaneval else False
    if is_humaneval:
        extraction_success = sum(item.get("parse_status") == "success" for item in records)
        if execution_enabled:
            metric_name = "pass@1"
            score = correct / total if total else None
        else:
            metric_name = "completion_extraction_rate"
            score = extraction_success / total if total else None
    else:
        metric_name = "M1 rollout accuracy"
        score = correct / total if total else None
        extraction_success = None
    return {
        "metric_name": metric_name,
        "score": score,
        "evaluated_count": total,
        "completed_request_count": completed,
        "correct_count": correct,
        "parse_failure_count": parse_failures,
        "parse_failure_rate": parse_failures / total if total else None,
        "failed_request_count": request_failures,
        "failed_request_rate": request_failures / total if total else None,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "generated_completion_count": completed if is_humaneval else None,
        "completion_extraction_success_count": extraction_success,
        "code_execution_enabled": execution_enabled if is_humaneval else None,
    }


def build_summaries(records: list[dict[str, Any]], model_keys: list[str]) -> dict[str, list[dict[str, Any]]]:
    records = latest_records(records)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model_key"], record["dataset"], record["split"])].append(record)

    per_dataset = []
    pair_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (model, dataset, split), values in sorted(grouped.items()):
        row = {"model_key": model, "dataset": dataset, "split": split}
        row.update(_group_summary(values))
        per_dataset.append(row)
        pair_groups[(model, dataset)].extend(values)

    matrix_long = []
    for model in model_keys:
        for dataset in DATASET_ORDER:
            values = pair_groups.get((model, dataset), [])
            row = {"model_key": model, "dataset": dataset}
            row.update(_group_summary(values) if values else {
                "metric_name": "pass@1" if dataset == "humaneval" else "M1 rollout accuracy",
                "score": None,
                "evaluated_count": 0,
                "completed_request_count": 0,
                "correct_count": 0,
                "parse_failure_count": 0,
                "parse_failure_rate": None,
                "failed_request_count": 0,
                "failed_request_rate": None,
                "mean_latency_seconds": None,
                "median_latency_seconds": None,
                "generated_completion_count": None,
                "completion_extraction_success_count": None,
                "code_execution_enabled": None,
            })
            matrix_long.append(row)

    overall_matrix = []
    for model in model_keys:
        row: dict[str, Any] = {"model_key": model}
        for dataset in DATASET_ORDER:
            detail = next(item for item in matrix_long if item["model_key"] == model and item["dataset"] == dataset)
            row[dataset] = detail["score"]
        overall_matrix.append(row)

    per_model = []
    for model in model_keys:
        values = [item for item in records if item["model_key"] == model]
        latencies = [float(item.get("latency_seconds") or 0.0) for item in values]
        per_model.append({
            "model_key": model,
            "attempted_count": len(values),
            "completed_request_count": sum(item.get("request_status") == "completed" for item in values),
            "request_failure_count": sum(item.get("request_status") == "failed" for item in values),
            "parse_failure_count": sum(item.get("parse_status") == "failed" for item in values),
            "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
            "note": "No aggregate capability score is computed across datasets.",
        })

    failures_parse = [item for item in records if item.get("parse_status") == "failed"]
    failures_request = [item for item in records if item.get("request_status") == "failed"]
    latency = [
        {
            "model_key": item["model_key"],
            "dataset": item["dataset"],
            "split": item["split"],
            "example_id": item["example_id"],
            "latency_seconds": item["latency_seconds"],
            "input_tokens": item["input_tokens"],
            "output_tokens": item["output_tokens"],
            "total_tokens": item["total_tokens"],
        }
        for item in records
    ]
    return {
        "per_dataset": per_dataset,
        "per_model": per_model,
        "matrix_long": matrix_long,
        "overall_matrix": overall_matrix,
        "parse_failures": failures_parse,
        "request_failures": failures_request,
        "latency": latency,
    }


def write_summary_artifacts(summary_dir: Path, records: list[dict[str, Any]], model_keys: list[str], mode: str) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summaries(records, model_keys)
    detail_fields = [
        "model_key", "dataset", "split", "metric_name", "score", "evaluated_count",
        "completed_request_count", "correct_count", "parse_failure_count", "parse_failure_rate",
        "failed_request_count", "failed_request_rate", "mean_latency_seconds",
        "median_latency_seconds", "generated_completion_count",
        "completion_extraction_success_count", "code_execution_enabled",
    ]
    write_csv(summary_dir / "per_dataset.csv", summary["per_dataset"], detail_fields)
    write_csv(summary_dir / "per_model.csv", summary["per_model"])
    write_csv(summary_dir / "overall_matrix.csv", summary["overall_matrix"])
    failure_fields = [
        "run_id", "model_key", "dataset", "split", "example_id", "parse_status",
        "request_status", "error_type", "error_message", "raw_response",
    ]
    write_csv(summary_dir / "parse_failures.csv", summary["parse_failures"], failure_fields)
    write_csv(summary_dir / "request_failures.csv", summary["request_failures"], failure_fields)
    write_csv(
        summary_dir / "latency_summary.csv",
        summary["latency"],
        ["model_key", "dataset", "split", "example_id", "latency_seconds", "input_tokens", "output_tokens", "total_tokens"],
    )
    if mode == "pilot":
        write_csv(summary_dir / "pilot_matrix.csv", summary["matrix_long"])
        write_csv(summary_dir / "pilot_per_dataset.csv", summary["per_dataset"], detail_fields)
        write_csv(summary_dir / "pilot_per_model.csv", summary["per_model"])


def report_markdown(
    run_id: str,
    mode: str,
    records: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    status: str,
) -> str:
    latest = latest_records(records)
    request_failures = sum(item.get("request_status") == "failed" for item in latest)
    parse_failures = sum(item.get("parse_status") == "failed" for item in latest)
    unavailable = [
        row for row in split_rows
        if not row.get("available_splits")
        or row.get("train_status") == "unavailable"
        or row.get("validation_status") == "unavailable"
    ]
    title = {
        "pilot": "M1 Pilot Report",
        "dry-run": "M1 Dry Run Report",
    }.get(mode, "M1 Preflight Smoke Report")
    lines = [
        f"# {title}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Status: `{status}`",
        f"- Prediction records: {len(latest)}",
        f"- Request failures: {request_failures}",
        f"- Parse failures: {parse_failures}",
        f"- Datasets with unavailable requested splits: {len(unavailable)}",
        "",
        "These are pipeline-validation artifacts, not final benchmark results.",
    ]
    return "\n".join(lines) + "\n"
# Matrix M1 report helpers imported from the 2026-07-15 handoff package.
# The rollout summary functions above stay in this module; these helpers cover
# lm-eval OpenLLM6 aggregation, generic accuracy, and domain6 reports.
from evaluation.metrics.m1_capability_reports import (
    DOMAIN6_TASKS,
    MODEL_ROLES,
    OPENLLM6_FEWSHOT,
    OPENLLM6_METRICS,
    accuracy_report,
    domain6_report,
    iter_jsonl,
    label_key,
    openllm6_report,
    read_json_object,
    unit_interval,
)

