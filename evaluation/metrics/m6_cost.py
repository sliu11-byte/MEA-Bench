#!/usr/bin/env python3
"""Matrix M6 cost metrics.

Commands:
  cost    Validate cost records and compute itemized monetary costs.
  target  Compute the optional, right-censored queries-to-target readout.

The core cost command expects JSONL rows such as:
  {
    "id": "attack-a", "stage": "attack", "currency": "USD",
    "price_as_of": "2026-07-13",
    "api_billing": "per_query", "query_count": 500,
    "price_per_query": 0.01, "gpu_hours": 2.0,
    "gpu_price_per_hour": 3.0, "data_cost": 0.0,
    "human_cost": 0.0, "other_cost": 0.0
  }

Unknown costs must be null or omitted. Zero means a measured cost of zero or a
component that is explicitly not applicable. The script never turns unknown
costs into zero and never collapses M6 into a single efficiency score.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGES = {"attack", "defense_install", "countermeasure"}
BILLING_MODES = {"none", "unknown", "per_query", "tokens"}


def iter_jsonl(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                yield row


def write_report(report: dict[str, Any], output: str | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def optional_nonnegative(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative number or null")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return value


def optional_nonnegative_integer(row: dict[str, Any], field: str) -> int | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer or null")
    return value


def multiply_if_known(left: float | None, right: float | None, scale: float = 1.0) -> float | None:
    if left is None or right is None:
        return None
    return left * right / scale


def cost_record(row: dict[str, Any]) -> dict[str, Any]:
    record_id = row.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Every M6 record needs a non-empty string id")

    stage = row.get("stage")
    if stage not in STAGES:
        raise ValueError(f"Record {record_id}: stage must be one of {sorted(STAGES)}")

    currency = row.get("currency")
    if not isinstance(currency, str) or not currency:
        raise ValueError(f"Record {record_id}: currency must be explicit")

    price_as_of = row.get("price_as_of")
    if not isinstance(price_as_of, str) or not price_as_of.strip():
        raise ValueError(f"Record {record_id}: price_as_of must be explicit")

    billing_mode = row.get("api_billing")
    if billing_mode not in BILLING_MODES:
        raise ValueError(
            f"Record {record_id}: api_billing must be one of {sorted(BILLING_MODES)}"
        )

    count_fields = (
        "query_count",
        "input_tokens",
        "output_tokens",
        "sample_count",
        "budget_cap",
    )
    numeric_fields = (
        "price_per_query",
        "input_price_per_million_tokens",
        "output_price_per_million_tokens",
        "gpu_hours",
        "gpu_price_per_hour",
        "wall_clock_hours",
        "data_cost",
        "human_cost",
        "other_cost",
    )
    numbers = {field: optional_nonnegative(row, field) for field in numeric_fields}
    numbers.update(
        {field: optional_nonnegative_integer(row, field) for field in count_fields}
    )

    price_fields = (
        "price_per_query",
        "input_price_per_million_tokens",
        "output_price_per_million_tokens",
    )

    if billing_mode == "none":
        if any(numbers[field] is not None for field in price_fields):
            raise ValueError(
                f"Record {record_id}: api_billing=none cannot include API price fields"
            )
        api_cost = 0.0
    elif billing_mode == "unknown":
        if any(numbers[field] is not None for field in price_fields):
            raise ValueError(
                f"Record {record_id}: choose per_query or tokens when API prices are known"
            )
        api_cost = None
    elif billing_mode == "per_query":
        if numbers["query_count"] is None or numbers["price_per_query"] is None:
            raise ValueError(
                f"Record {record_id}: per_query billing needs query_count and price_per_query"
            )
        if any(
            numbers[field] is not None
            for field in (
                "input_price_per_million_tokens",
                "output_price_per_million_tokens",
            )
        ):
            raise ValueError(
                f"Record {record_id}: do not mix per-query and token billing in one API charge"
            )
        api_cost = multiply_if_known(numbers["query_count"], numbers["price_per_query"])
    else:
        required = (
            "input_tokens",
            "output_tokens",
            "input_price_per_million_tokens",
            "output_price_per_million_tokens",
        )
        missing = [field for field in required if numbers[field] is None]
        if missing:
            raise ValueError(f"Record {record_id}: token billing is missing {', '.join(missing)}")
        if numbers["price_per_query"] is not None:
            raise ValueError(
                f"Record {record_id}: do not mix token and per-query billing in one API charge"
            )
        input_cost = multiply_if_known(
            numbers["input_tokens"], numbers["input_price_per_million_tokens"], 1_000_000
        )
        output_cost = multiply_if_known(
            numbers["output_tokens"], numbers["output_price_per_million_tokens"], 1_000_000
        )
        assert input_cost is not None and output_cost is not None
        api_cost = input_cost + output_cost

    gpu_hours = numbers["gpu_hours"]
    gpu_price = numbers["gpu_price_per_hour"]
    if (gpu_hours is None) != (gpu_price is None):
        gpu_cost = None
    else:
        gpu_cost = multiply_if_known(gpu_hours, gpu_price)

    components = {
        "api_cost": api_cost,
        "gpu_cost": gpu_cost,
        "data_cost": numbers["data_cost"],
        "human_cost": numbers["human_cost"],
        "other_cost": numbers["other_cost"],
    }
    unknown = [name for name, value in components.items() if value is None]
    known_subtotal = sum(value for value in components.values() if value is not None)

    reported_dimensions = []
    if numbers["query_count"] is not None:
        reported_dimensions.append("queries")
    if numbers["input_tokens"] is not None or numbers["output_tokens"] is not None:
        reported_dimensions.append("tokens")
    if numbers["sample_count"] is not None:
        reported_dimensions.append("samples")
    if gpu_hours is not None:
        reported_dimensions.append("gpu_hours")
    if numbers["wall_clock_hours"] is not None:
        reported_dimensions.append("wall_clock_hours")
    if any(value is not None for value in components.values()):
        reported_dimensions.append("money")
    if len(reported_dimensions) < 2:
        raise ValueError(
            f"Record {record_id}: report at least two cost dimensions; "
            f"got {reported_dimensions}"
        )

    return {
        "id": record_id,
        "method": row.get("method"),
        "stage": stage,
        "currency": currency,
        "price_as_of": price_as_of,
        "reported_cost_dimensions": reported_dimensions,
        "dimensions": {
            "query_count": numbers["query_count"],
            "input_tokens": numbers["input_tokens"],
            "output_tokens": numbers["output_tokens"],
            "sample_count": numbers["sample_count"],
            "gpu_hours": gpu_hours,
            "wall_clock_hours": numbers["wall_clock_hours"],
            "budget_cap": numbers["budget_cap"],
        },
        "monetary_cost": {
            "api_billing": billing_mode,
            **components,
            "known_subtotal": known_subtotal,
            "complete": not unknown,
            "unknown_components": unknown,
        },
        "linked_effect": row.get("linked_effect"),
    }


def cost_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = [cost_record(row) for row in rows]
    if not records:
        raise ValueError("No M6 cost records found")
    seen = set()
    for record in records:
        if record["id"] in seen:
            raise ValueError(f"Duplicate M6 record id: {record['id']}")
        seen.add(record["id"])
    return {
        "metric_family": "M6_cost",
        "records": records,
        "n": len(records),
        "protocol": {
            "scope": "Cost axes only; linked M1/M2/M3/M5 effects are preserved but not merged.",
            "unknown_values": "null is unknown; zero is measured zero or explicitly not applicable",
            "money": "API, GPU, data, human, and other costs remain itemized",
            "minimum_dimensions": "Every record reports at least two cost dimensions.",
            "no_efficiency_score": True,
        },
    }


def queries_to_target(
    points: Iterable[dict[str, Any]],
    target: float,
    budget_cap: int,
    higher_is_better: bool,
) -> dict[str, Any]:
    if not math.isfinite(target):
        raise ValueError("target must be finite")
    if not isinstance(budget_cap, int) or isinstance(budget_cap, bool) or budget_cap < 0:
        raise ValueError("budget_cap must be a non-negative integer")

    by_method: dict[str, list[tuple[int, float]]] = defaultdict(list)
    seen_points: set[tuple[str, int]] = set()
    effect_definitions: set[tuple[str, str, str]] = set()
    for row in points:
        method = row.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("Every target-curve row needs a non-empty method")
        effect_values = []
        for field in ("effect_family", "metric", "unit"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Method {method}: target-curve field {field!r} must be non-empty text"
                )
            effect_values.append(value)
        effect_definitions.add(tuple(effect_values))
        budget = optional_nonnegative_integer(row, "budget")
        value = row.get("value")
        if budget is None:
            raise ValueError(f"Method {method}: budget is required")
        point_key = (method, budget)
        if point_key in seen_points:
            raise ValueError(
                f"Duplicate target-curve point for method={method!r}, budget={budget}"
            )
        seen_points.add(point_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Method {method}: value must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Method {method}: value must be finite")
        by_method[method].append((budget, value))

    if not by_method:
        raise ValueError("No target-curve points found")
    if len(effect_definitions) != 1:
        raise ValueError(
            "All target-curve rows must use one effect_family, metric, and unit"
        )
    effect_family, metric, unit = effect_definitions.pop()

    results = []
    for method, all_method_points in sorted(by_method.items()):
        method_points = [point for point in all_method_points if point[0] <= budget_cap]
        method_points.sort()
        if not method_points:
            results.append(
                {
                    "method": method,
                    "queries_to_target": None,
                    "censored": False,
                    "censor_at": None,
                    "display": "unavailable",
                    "status": "unavailable",
                    "reason": "No observation at or below budget_cap",
                    "points_used": 0,
                    "observed_at_budget_cap": False,
                }
            )
            continue
        if higher_is_better:
            reached = [budget for budget, value in method_points if value >= target]
        else:
            reached = [budget for budget, value in method_points if value <= target]
        first = min(reached) if reached else None
        last_observed_budget = max(budget for budget, _ in method_points)
        results.append(
            {
                "method": method,
                "queries_to_target": first,
                "censored": first is None,
                "censor_at": last_observed_budget if first is None else None,
                "display": f">{last_observed_budget}" if first is None else f"{first}",
                "status": "censored" if first is None else "reached",
                "points_used": len(method_points),
                "observed_at_budget_cap": last_observed_budget == budget_cap,
            }
        )
    return {
        "metric": "queries_to_target",
        "effect": {
            "family": effect_family,
            "metric": metric,
            "unit": unit,
        },
        "target": target,
        "budget_cap": budget_cap,
        "higher_is_better": higher_is_better,
        "results": results,
        "protocol": {
            "scope": "Secondary M6 readout; benefit values belong to their source metric family.",
            "effect_identity": "All rows share one explicit effect family, metric, and unit.",
            "censoring": "An unreached target is censored at the method's largest observed budget at or below budget_cap.",
            "duplicates": "One pre-aggregated value is required per method and budget.",
        },
    }


def command_cost(args: argparse.Namespace) -> int:
    write_report(cost_report(iter_jsonl(args.inputs)), args.output)
    return 0


def command_target(args: argparse.Namespace) -> int:
    report = queries_to_target(
        iter_jsonl(args.inputs),
        target=args.target,
        budget_cap=args.budget_cap,
        higher_is_better=args.direction == "higher",
    )
    write_report(report, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Matrix M6 cost metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cost = subparsers.add_parser("cost", help="Validate and summarize itemized cost records.")
    cost.add_argument("inputs", nargs="+", help="M6 cost JSONL files.")
    cost.add_argument("--output", help="Write JSON report to this path.")
    cost.set_defaults(func=command_cost)

    target = subparsers.add_parser("target", help="Compute optional queries-to-target.")
    target.add_argument("inputs", nargs="+", help="JSONL rows with method, budget, and value.")
    target.add_argument("--target", required=True, type=float)
    target.add_argument("--budget-cap", required=True, type=int)
    target.add_argument("--direction", choices=("higher", "lower"), required=True)
    target.add_argument("--output", help="Write JSON report to this path.")
    target.set_defaults(func=command_target)
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
