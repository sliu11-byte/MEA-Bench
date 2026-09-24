#!/usr/bin/env python3
"""M1 capability metrics.

Commands:
  openllm6  Read an lm-eval results_*.json file and average the six v1 tasks.
  accuracy  Compute exact accuracy from configurable JSONL fields.

The accuracy command expects one JSON object per line. Its default schema is:
  {"task": "math", "prediction": "A", "target": "A"}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


OPENLLM6_METRICS = {
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "mmlu": "acc,none",
    "truthfulqa_mc2": "acc,none",
    "winogrande": "acc,none",
    "gsm8k": "exact_match,strict-match",
}
OPENLLM6_FEWSHOT = {
    "arc_challenge": 25,
    "hellaswag": 10,
    "mmlu": 5,
    "truthfulqa_mc2": 0,
    "winogrande": 5,
    "gsm8k": 5,
}
DOMAIN6_TASKS = ("medqa", "pubmedqa", "chemprot", "fomc", "headline", "fpb")
MODEL_ROLES = ("teacher", "student", "bare_base")


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def read_json_object(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_json_constant,
        )
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def iter_jsonl(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line, parse_constant=reject_json_constant)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSONL: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                yield row


def write_report(report: dict[str, Any], output: str | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def unit_interval(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a numeric score in [0, 1]")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{location} must be a numeric score in [0, 1]")
    return score


def openllm6_report(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("lm-eval payload is missing object field 'results'")

    n_shot = payload.get("n-shot")
    configs = payload.get("configs")
    fewshot = {}
    for task, expected in OPENLLM6_FEWSHOT.items():
        if isinstance(n_shot, dict) and task in n_shot:
            value = n_shot[task]
        elif (
            isinstance(configs, dict)
            and isinstance(configs.get(task), dict)
            and "num_fewshot" in configs[task]
        ):
            value = configs[task]["num_fewshot"]
        else:
            raise ValueError(
                f"lm-eval payload is missing few-shot metadata for task {task!r}"
            )
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"Task {task!r} must use {expected}-shot, got {value!r}"
            )
        fewshot[task] = value

    scores = {}
    for task, metric_key in OPENLLM6_METRICS.items():
        task_results = results.get(task)
        if not isinstance(task_results, dict):
            raise ValueError(f"lm-eval results are missing task object {task!r}")
        if metric_key not in task_results:
            raise ValueError(f"lm-eval task {task!r} is missing metric {metric_key!r}")
        scores[task] = unit_interval(
            task_results[metric_key],
            f"results[{task!r}][{metric_key!r}]",
        )

    average = math.fsum(scores.values()) / len(scores)
    return {
        "metric": "openllm6_average",
        "scores": scores,
        "openllm6_average": average,
        "n_tasks": len(scores),
        "fewshot": fewshot,
        "protocol": {
            "source": "lm-evaluation-harness results_*.json",
            "task_metric_keys": dict(OPENLLM6_METRICS),
            "aggregation": "unweighted arithmetic mean of six task scores",
            "fewshot": dict(OPENLLM6_FEWSHOT),
            "score_range": "0..1",
        },
    }


def label_key(value: Any, field: str, row_number: int) -> tuple[str, Any]:
    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, str):
        return "string", value
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return "number", number
    raise ValueError(
        f"Row {row_number} field {field!r} must be a finite JSON scalar, not {value!r}"
    )


def accuracy_report(
    rows: Iterable[dict[str, Any]],
    *,
    prediction_field: str = "prediction",
    target_field: str = "target",
    task_field: str = "task",
) -> dict[str, Any]:
    fields = (prediction_field, target_field, task_field)
    if any(not field for field in fields) or len(set(fields)) != len(fields):
        raise ValueError("prediction, target, and task field names must be non-empty and distinct")

    correct = 0
    total = 0
    per_task_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {row_number} must be a JSON object")
        missing = [field for field in fields if field not in row]
        if missing:
            names = ", ".join(repr(field) for field in missing)
            raise ValueError(f"Row {row_number} is missing required field(s): {names}")

        task = row[task_field]
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Row {row_number} field {task_field!r} must be non-empty text")
        prediction = label_key(row[prediction_field], prediction_field, row_number)
        target = label_key(row[target_field], target_field, row_number)
        matched = prediction == target

        total += 1
        correct += int(matched)
        per_task_counts[task][1] += 1
        per_task_counts[task][0] += int(matched)

    if total == 0:
        raise ValueError("No accuracy rows found")

    per_task = [
        {"task": task, "accuracy": matched / count, "correct": matched, "n": count}
        for task, (matched, count) in sorted(per_task_counts.items())
    ]
    return {
        "metric": "accuracy",
        "accuracy": correct / total,
        "correct": correct,
        "n": total,
        "per_task": per_task,
        "protocol": {
            "prediction_field": prediction_field,
            "target_field": target_field,
            "task_field": task_field,
            "comparison": "exact JSON scalar equality",
            "aggregation": "micro-average over all rows",
            "score_range": "0..1",
        },
    }


def domain6_report(
    rows: Iterable[dict[str, Any]],
    *,
    model_id: str,
    model_role: str,
    prediction_field: str = "prediction",
    target_field: str = "target",
    task_field: str = "task",
    id_field: str = "id",
) -> dict[str, Any]:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be non-empty text")
    if model_role not in MODEL_ROLES:
        raise ValueError(f"model_role must be one of {MODEL_ROLES}")
    materialized = list(rows)
    seen = set()
    for row_number, row in enumerate(materialized, start=1):
        if id_field not in row:
            raise ValueError(f"Row {row_number} is missing required id field {id_field!r}")
        if task_field not in row:
            raise ValueError(f"Row {row_number} is missing task field {task_field!r}")
        sample_id = label_key(row[id_field], id_field, row_number)
        key = (str(row[task_field]), sample_id)
        if key in seen:
            raise ValueError(
                f"Duplicate Matrix domain6 sample: task={row[task_field]!r}, "
                f"{id_field}={row[id_field]!r}"
            )
        seen.add(key)
    report = accuracy_report(
        materialized,
        prediction_field=prediction_field,
        target_field=target_field,
        task_field=task_field,
    )
    observed = {row["task"] for row in report["per_task"]}
    required = set(DOMAIN6_TASKS)
    if observed != required:
        raise ValueError(
            "Matrix domain6 task coverage differs: "
            f"missing={sorted(required - observed)}, extra={sorted(observed - required)}"
        )
    report["metric"] = "matrix_domain6_accuracy"
    report["domain_micro_accuracy"] = report["accuracy"]
    report["domain_macro_accuracy"] = math.fsum(
        row["accuracy"] for row in report["per_task"]
    ) / len(DOMAIN6_TASKS)
    report["model_id"] = model_id
    report["model_role"] = model_role
    report["protocol"]["required_tasks"] = list(DOMAIN6_TASKS)
    report["protocol"]["coverage"] = "exact six-domain task set required"
    report["protocol"]["id_field"] = id_field
    report["protocol"]["deduplication"] = "(task, id) must be unique"
    report["protocol"]["domain_aggregation"] = {
        "micro": "all examples have equal weight",
        "macro": "the six domain accuracies have equal weight",
    }
    return report


def command_openllm6(args: argparse.Namespace) -> int:
    write_report(openllm6_report(read_json_object(args.input)), args.output)
    return 0


def command_accuracy(args: argparse.Namespace) -> int:
    report = accuracy_report(
        iter_jsonl(args.inputs),
        prediction_field=args.prediction_field,
        target_field=args.target_field,
        task_field=args.task_field,
    )
    write_report(report, args.output)
    return 0


def command_domain6(args: argparse.Namespace) -> int:
    report = domain6_report(
        iter_jsonl(args.inputs),
        model_id=args.model_id,
        model_role=args.model_role,
        prediction_field=args.prediction_field,
        target_field=args.target_field,
        task_field=args.task_field,
        id_field=args.id_field,
    )
    write_report(report, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Matrix M1 capability metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    openllm6 = subparsers.add_parser(
        "openllm6",
        help="Read the six official OpenLLM v1 metrics from lm-eval results JSON.",
    )
    openllm6.add_argument("input", help="lm-eval results_*.json file.")
    openllm6.add_argument("--output", help="Write JSON report to this path.")
    openllm6.set_defaults(func=command_openllm6)

    accuracy = subparsers.add_parser("accuracy", help="Compute grouped exact accuracy from JSONL.")
    accuracy.add_argument("inputs", nargs="+", help="Prediction JSONL files.")
    accuracy.add_argument("--prediction-field", default="prediction")
    accuracy.add_argument("--target-field", default="target")
    accuracy.add_argument("--task-field", default="task")
    accuracy.add_argument("--output", help="Write JSON report to this path.")
    accuracy.set_defaults(func=command_accuracy)

    domain6 = subparsers.add_parser(
        "domain6", help="Compute the exact Matrix six-domain capability report."
    )
    domain6.add_argument("inputs", nargs="+", help="Prediction JSONL files.")
    domain6.add_argument("--model-id", required=True)
    domain6.add_argument("--model-role", choices=MODEL_ROLES, required=True)
    domain6.add_argument("--prediction-field", default="prediction")
    domain6.add_argument("--target-field", default="target")
    domain6.add_argument("--task-field", default="task")
    domain6.add_argument("--id-field", default="id")
    domain6.add_argument("--output", help="Write JSON report to this path.")
    domain6.set_defaults(func=command_domain6)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
