#!/usr/bin/env python3
"""Matrix M7 active model-stealing detection metrics.

This module evaluates precomputed detector scores. It does not train a detector.
The threshold is calibrated only from benign calibration entities, then locked
for the evaluation set.

Event JSONL schema:
  {
    "entity_id": "attacker-1", "label": "attacker",
    "query_index": 20, "score": 0.87, "attack_type": "evasive",
    "detector": "active-v1", "monitor_mode": "per-account"
  }

All events belonging to one entity must share the same label and attack_type.
Each entity must record one score per query, starting at query 1 without gaps.
Calibration benign entities and evaluation benign entities must reach the full
budget. Undetected attackers must also reach the full budget. A detected
attacker may stop at detection, but ROC-AUC is then unavailable because the
positive and negative score horizons differ.
An entity is detected at the first query whose score crosses the locked
threshold. Undetected attackers remain in the report as right-censored at the
explicit query budget.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LABELS = {"benign", "attacker"}
DIRECTIONS = {"higher", "lower"}
MONITOR_MODES = {"per-account", "global-context"}


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


def validate_event(row: dict[str, Any]) -> dict[str, Any]:
    entity_id = row.get("entity_id")
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("Every M7 event needs a non-empty entity_id")
    label = row.get("label")
    if label not in LABELS:
        raise ValueError(f"Entity {entity_id}: label must be benign or attacker")
    query_index = row.get("query_index")
    if not isinstance(query_index, int) or isinstance(query_index, bool) or query_index < 1:
        raise ValueError(f"Entity {entity_id}: query_index must be a positive integer")
    score = row.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError(f"Entity {entity_id}: score must be numeric")
    score = float(score)
    if not math.isfinite(score):
        raise ValueError(f"Entity {entity_id}: score must be finite")
    attack_type = row.get("attack_type")
    if label == "attacker" and (not isinstance(attack_type, str) or not attack_type):
        raise ValueError(f"Entity {entity_id}: attacker events need attack_type")
    detector = row.get("detector")
    if not isinstance(detector, str) or not detector.strip():
        raise ValueError(f"Entity {entity_id}: detector must be non-empty text")
    monitor_mode = row.get("monitor_mode")
    if monitor_mode not in MONITOR_MODES:
        raise ValueError(
            f"Entity {entity_id}: monitor_mode must be one of {sorted(MONITOR_MODES)}"
        )
    return {
        "entity_id": entity_id,
        "label": label,
        "query_index": query_index,
        "score": score,
        "attack_type": attack_type if label == "attacker" else None,
        "detector": detector,
        "monitor_mode": monitor_mode,
        "rho": row.get("rho"),
        "sybil_group": row.get("sybil_group"),
    }


def group_events(rows: Iterable[dict[str, Any]], budget: int) -> dict[str, dict[str, Any]]:
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("budget must be a positive integer")
    grouped: dict[str, dict[str, Any]] = {}
    seen_indices: dict[str, set[int]] = defaultdict(set)
    for raw in rows:
        event = validate_event(raw)
        if event["query_index"] > budget:
            continue
        entity_id = event["entity_id"]
        if event["query_index"] in seen_indices[entity_id]:
            raise ValueError(
                f"Entity {entity_id}: duplicate query_index {event['query_index']}"
            )
        seen_indices[entity_id].add(event["query_index"])
        entity = grouped.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "label": event["label"],
                "attack_type": event["attack_type"],
                "detector": event["detector"],
                "monitor_mode": event["monitor_mode"],
                "rho": event["rho"],
                "sybil_group": event["sybil_group"],
                "events": [],
            },
        )
        if entity["label"] != event["label"]:
            raise ValueError(f"Entity {entity_id}: label changes across events")
        if entity["attack_type"] != event["attack_type"]:
            raise ValueError(f"Entity {entity_id}: attack_type changes across events")
        if entity["detector"] != event["detector"]:
            raise ValueError(f"Entity {entity_id}: detector changes across events")
        if entity["monitor_mode"] != event["monitor_mode"]:
            raise ValueError(f"Entity {entity_id}: monitor_mode changes across events")
        entity["events"].append((event["query_index"], event["score"]))
    for entity in grouped.values():
        entity["events"].sort()
        indices = [query_index for query_index, _ in entity["events"]]
        if indices != list(range(1, indices[-1] + 1)):
            raise ValueError(
                f"Entity {entity['entity_id']}: query_index values must be contiguous from 1"
            )
        entity["observed_until"] = indices[-1]
    if not grouped:
        raise ValueError("No M7 events at or below the query budget")
    if len({entity["detector"] for entity in grouped.values()}) != 1:
        raise ValueError("All M7 entities in one input must use the same detector")
    if len({entity["monitor_mode"] for entity in grouped.values()}) != 1:
        raise ValueError("All M7 entities in one input must use the same monitor_mode")
    return grouped


def binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if successes >= trials else 0.0
    log_term = (
        math.lgamma(trials + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(trials - successes + 1)
        + successes * math.log(probability)
        + (trials - successes) * math.log1p(-probability)
    )
    max_log = log_term
    scaled_total = 1.0
    for count in range(successes, 0, -1):
        log_term += (
            math.log(count)
            - math.log(trials - count + 1)
            + math.log1p(-probability)
            - math.log(probability)
        )
        if log_term > max_log:
            scaled_total = scaled_total * math.exp(max_log - log_term) + 1.0
            max_log = log_term
        elif log_term - max_log > -745.0:
            scaled_total += math.exp(log_term - max_log)
    log_cdf = max_log + math.log(scaled_total)
    if log_cdf >= 0.0:
        return 1.0
    if log_cdf < math.log(sys.float_info.min):
        return 0.0
    return math.exp(log_cdf)


def binomial_upper_bound(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> float:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Binomial counts must satisfy 0 <= successes <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must satisfy 0 < confidence < 1")
    if successes == trials:
        return 1.0
    alpha = 1.0 - confidence
    low = successes / trials
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if binomial_cdf(successes, trials, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return high


def crosses(score: float, threshold: float, direction: str) -> bool:
    if direction == "higher":
        return score >= threshold
    if direction == "lower":
        return score <= threshold
    raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}")


def entity_extreme(entity: dict[str, Any], direction: str) -> float:
    scores = [score for _, score in entity["events"]]
    if not scores:
        raise ValueError(f"Entity {entity['entity_id']} has no events")
    return max(scores) if direction == "higher" else min(scores)


def conservative_threshold(benign_scores: list[float], target_fpr: float, direction: str) -> float:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    if not benign_scores:
        raise ValueError("Threshold calibration needs benign entities")

    if direction == "higher":
        candidates = [math.nextafter(max(benign_scores), math.inf), *set(benign_scores)]
        feasible = [
            threshold
            for threshold in candidates
            if sum(score >= threshold for score in benign_scores) / len(benign_scores)
            <= target_fpr
        ]
        return min(feasible)

    candidates = [math.nextafter(min(benign_scores), -math.inf), *set(benign_scores)]
    feasible = [
        threshold
        for threshold in candidates
        if sum(score <= threshold for score in benign_scores) / len(benign_scores)
        <= target_fpr
    ]
    return max(feasible)


def rank_auc(positive_scores: list[float], negative_scores: list[float], direction: str) -> float:
    if not positive_scores or not negative_scores:
        raise ValueError("ROC-AUC needs at least one attacker and one benign entity")
    favorable = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive == negative:
                favorable += 0.5
            elif (positive > negative) == (direction == "higher"):
                favorable += 1.0
    return favorable / (len(positive_scores) * len(negative_scores))


def first_detection(entity: dict[str, Any], threshold: float, direction: str) -> int | None:
    for query_index, score in entity["events"]:
        if crosses(score, threshold, direction):
            return query_index
    return None


def detection_curve(
    attacker_results: list[dict[str, Any]], checkpoints: list[int], budget: int
) -> list[dict[str, Any]]:
    if not attacker_results:
        raise ValueError("Detection curve needs attacker entities")
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoints must not contain duplicates")
    normalized = sorted(checkpoints)
    if not normalized or any(point < 1 or point > budget for point in normalized):
        raise ValueError("checkpoints must be unique positive integers at or below budget")
    return [
        {
            "queries": point,
            "detection_rate": sum(
                result["queries_to_detection"] is not None
                and result["queries_to_detection"] <= point
                for result in attacker_results
            )
            / len(attacker_results),
            "detected": sum(
                result["queries_to_detection"] is not None
                and result["queries_to_detection"] <= point
                for result in attacker_results
            ),
            "n": len(attacker_results),
        }
        for point in normalized
    ]


def type_summary(results: list[dict[str, Any]], checkpoints: list[int], budget: int) -> dict[str, Any]:
    detected_times = [
        result["queries_to_detection"]
        for result in results
        if result["queries_to_detection"] is not None
    ]
    return {
        "detection_rate": len(detected_times) / len(results),
        "detected": len(detected_times),
        "n": len(results),
        "censored": len(results) - len(detected_times),
        "queries_to_detection_detected_only": {
            "mean": statistics.fmean(detected_times) if detected_times else None,
            "median": statistics.median(detected_times) if detected_times else None,
        },
        "curve": detection_curve(results, checkpoints, budget),
    }


def require_observed_through_budget(
    entities: Iterable[dict[str, Any]], budget: int, role: str
) -> None:
    incomplete = [
        entity["entity_id"]
        for entity in entities
        if entity["observed_until"] < budget
    ]
    if incomplete:
        raise ValueError(
            f"{role} entities must be observed through query budget {budget}: {sorted(incomplete)}"
        )


def active_detection_report(
    calibration_rows: Iterable[dict[str, Any]],
    evaluation_rows: Iterable[dict[str, Any]],
    *,
    target_fpr: float,
    budget: int,
    checkpoints: list[int],
    direction: str,
    monitor_mode: str,
    confidence: float = 0.95,
    required_attack_types: Iterable[str] | None = None,
) -> dict[str, Any]:
    if monitor_mode not in MONITOR_MODES:
        raise ValueError(f"monitor_mode must be one of {sorted(MONITOR_MODES)}")
    calibration = group_events(calibration_rows, budget)
    calibration_detector = next(iter(calibration.values()))["detector"]
    calibration_mode = next(iter(calibration.values()))["monitor_mode"]
    if calibration_mode != monitor_mode:
        raise ValueError("Calibration event monitor_mode differs from the requested mode")
    non_benign = [
        entity_id for entity_id, entity in calibration.items() if entity["label"] != "benign"
    ]
    if non_benign:
        raise ValueError("Calibration input must contain benign entities only")
    require_observed_through_budget(calibration.values(), budget, "Calibration benign")
    calibration_scores = [entity_extreme(entity, direction) for entity in calibration.values()]
    threshold = conservative_threshold(calibration_scores, target_fpr, direction)
    calibration_fpr = sum(
        crosses(score, threshold, direction) for score in calibration_scores
    ) / len(calibration_scores)

    evaluation = group_events(evaluation_rows, budget)
    evaluation_detector = next(iter(evaluation.values()))["detector"]
    evaluation_mode = next(iter(evaluation.values()))["monitor_mode"]
    if evaluation_detector != calibration_detector:
        raise ValueError("Calibration and evaluation detectors differ")
    if evaluation_mode != calibration_mode:
        raise ValueError("Calibration and evaluation monitor modes differ")
    overlap = sorted(set(calibration) & set(evaluation))
    if overlap:
        raise ValueError(
            f"Calibration and evaluation entity IDs must be disjoint: {overlap}"
        )
    benign = [entity for entity in evaluation.values() if entity["label"] == "benign"]
    attackers = [entity for entity in evaluation.values() if entity["label"] == "attacker"]
    if not benign or not attackers:
        raise ValueError("Evaluation input needs both benign and attacker entities")
    observed_attack_types = {entity["attack_type"] for entity in attackers}
    required_types = set(required_attack_types or ())
    missing_attack_types = sorted(required_types - observed_attack_types)
    if missing_attack_types:
        raise ValueError(
            f"Evaluation is missing required attack types: {missing_attack_types}"
        )
    require_observed_through_budget(benign, budget, "Evaluation benign")

    benign_scores = [entity_extreme(entity, direction) for entity in benign]
    attacker_scores = [entity_extreme(entity, direction) for entity in attackers]
    actual_fpr = sum(crosses(score, threshold, direction) for score in benign_scores) / len(
        benign_scores
    )
    calibration_false_positives = sum(
        crosses(score, threshold, direction) for score in calibration_scores
    )
    evaluation_false_positives = sum(
        crosses(score, threshold, direction) for score in benign_scores
    )
    calibration_fpr_upper = binomial_upper_bound(
        calibration_false_positives, len(calibration_scores), confidence
    )
    evaluation_fpr_upper = binomial_upper_bound(
        evaluation_false_positives, len(benign_scores), confidence
    )

    per_attacker = []
    for entity in sorted(attackers, key=lambda item: item["entity_id"]):
        detected_at = first_detection(entity, threshold, direction)
        if detected_at is None and entity["observed_until"] < budget:
            raise ValueError(
                f"Undetected attacker {entity['entity_id']} must be observed through query budget {budget}"
            )
        per_attacker.append(
            {
                "entity_id": entity["entity_id"],
                "attack_type": entity["attack_type"],
                "rho": entity["rho"],
                "sybil_group": entity["sybil_group"],
                "queries_to_detection": detected_at,
                "censored": detected_at is None,
                "censor_at": entity["observed_until"] if detected_at is None else None,
                "observed_until": entity["observed_until"],
            }
        )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in per_attacker:
        by_type[result["attack_type"]].append(result)

    locked_tpr = sum(not result["censored"] for result in per_attacker) / len(
        per_attacker
    )
    calibration_resolution = 1.0 / len(calibration_scores)
    evaluation_resolution = 1.0 / len(benign_scores)
    fixed_fpr_reasons = []
    if calibration_fpr_upper > target_fpr:
        fixed_fpr_reasons.append(
            "Calibration FPR upper confidence bound exceeds target_fpr"
        )
    if evaluation_fpr_upper > target_fpr:
        fixed_fpr_reasons.append(
            "Evaluation FPR upper confidence bound exceeds target_fpr"
        )
    if actual_fpr > target_fpr:
        fixed_fpr_reasons.append("Evaluation FPR exceeds target_fpr at the locked threshold")
    fixed_fpr_available = not fixed_fpr_reasons
    auc_incomplete_attackers = sorted(
        entity["entity_id"]
        for entity in attackers
        if entity["observed_until"] < budget
    )
    auc_available = not auc_incomplete_attackers

    return {
        "metric_family": "M7_active_detection",
        "detector": calibration_detector,
        "threshold": threshold,
        "score_direction": direction,
        "target_fpr": target_fpr,
        "confidence": confidence,
        "calibration_fpr": calibration_fpr,
        "actual_fpr": actual_fpr,
        "calibration": {
            "benign_entities": len(calibration_scores),
            "empirical_fpr_resolution": calibration_resolution,
            "target_below_empirical_resolution": target_fpr < calibration_resolution,
            "false_positives": calibration_false_positives,
            "fpr_upper_confidence_bound": calibration_fpr_upper,
        },
        "evaluation": {
            "benign_entities": len(benign),
            "attacker_entities": len(attackers),
            "empirical_fpr_resolution": evaluation_resolution,
            "false_positives": evaluation_false_positives,
            "fpr_upper_confidence_bound": evaluation_fpr_upper,
        },
        "tpr_at_locked_threshold": locked_tpr,
        "tpr_at_fixed_fpr": locked_tpr if fixed_fpr_available else None,
        "fixed_fpr_status": {
            "available": fixed_fpr_available,
            "reasons": fixed_fpr_reasons,
        },
        "roc_auc": (
            rank_auc(attacker_scores, benign_scores, direction)
            if auc_available else None
        ),
        "roc_auc_status": {
            "available": auc_available,
            "reason": (
                None
                if auc_available
                else "Attacker and benign entity scores do not share the full query-budget horizon"
            ),
            "incomplete_attacker_entities": auc_incomplete_attackers,
        },
        "query_budget": budget,
        "monitor_mode": monitor_mode,
        "coverage": {
            "required_attack_types": sorted(required_types),
            "observed_attack_types": sorted(observed_attack_types),
            "missing_attack_types": missing_attack_types,
            "monitor_mode": monitor_mode,
            "full_matrix_requires_separate_modes": sorted(MONITOR_MODES),
        },
        "overall": type_summary(per_attacker, checkpoints, budget),
        "per_attack_type": {
            attack_type: type_summary(results, checkpoints, budget)
            for attack_type, results in sorted(by_type.items())
        },
        "per_attacker": per_attacker,
        "protocol": {
            "threshold": "Calibrated on separate benign entities and locked before evaluation.",
            "fpr_unit": "Fraction of benign monitoring entities detected at least once by budget.",
            "right_censoring": "Undetected attackers are accepted only after observation through the explicit query budget.",
            "fixed_fpr": "The headline TPR is unavailable unless one-sided binomial FPR upper bounds stay at or below target_fpr.",
            "detected_only_warning": "Mean/median queries-to-detection exclude censored attackers; use with detection rate.",
            "monitoring": "Run each monitoring mode on its own detector-score file; this code does not invent a global detector.",
            "event_coverage": "Scores must be contiguous from query 1. Benign and undetected attacker entities must reach budget.",
            "roc_auc": "Available only when attacker and benign entities all share the full budget horizon.",
        },
    }


def command_evaluate(args: argparse.Namespace) -> int:
    report = active_detection_report(
        iter_jsonl(args.calibration),
        iter_jsonl(args.evaluation),
        target_fpr=args.target_fpr,
        budget=args.budget,
        checkpoints=args.checkpoints,
        direction=args.score_direction,
        monitor_mode=args.monitor_mode,
        confidence=args.confidence,
        required_attack_types=args.required_attack_types,
    )
    write_report(report, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Matrix M7 active-detection metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate locked-threshold detection.")
    evaluate.add_argument("--calibration", nargs="+", required=True, help="Benign calibration JSONL.")
    evaluate.add_argument("--evaluation", nargs="+", required=True, help="Benign + attacker JSONL.")
    evaluate.add_argument("--target-fpr", type=float, required=True)
    evaluate.add_argument("--confidence", type=float, required=True)
    evaluate.add_argument("--budget", type=int, required=True)
    evaluate.add_argument("--checkpoints", type=int, nargs="+", required=True)
    evaluate.add_argument("--score-direction", choices=sorted(DIRECTIONS), required=True)
    evaluate.add_argument("--monitor-mode", choices=sorted(MONITOR_MODES), required=True)
    evaluate.add_argument(
        "--required-attack-types",
        nargs="+",
        default=["naive", "evasive"],
        help="Coverage required in this run. Default: naive evasive.",
    )
    evaluate.add_argument("--output", help="Write JSON report to this path.")
    evaluate.set_defaults(func=command_evaluate)
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
