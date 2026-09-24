#!/usr/bin/env python3
"""Matrix M4 passive provenance metrics.

Commands:
  detector     Calibrate a conservative threshold on negative scores and report
               TPR at the requested FPR, actual test FPR, and ROC-AUC.
  kgw          Compute the one-sided KGW z-score and p-value.
  fingerprint  Compute exact-match FSR and negative-model FPR/uniqueness.

Detector-score JSONL rows must state all protocol-critical fields explicitly:
  {"id": "p1", "label": "positive", "score": 2.4,
   "direction": "higher", "detector": "kgw-v1"}

Fingerprint suspect rows:
  {"id": "trigger-1", "response": "blue", "expected": "blue",
   "detector": "fingerprint-v1"}

Fingerprint negative-pool rows add a model identifier:
  {"model_id": "base", "id": "trigger-1", "response": "red", "expected": "blue"}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LABELS = {"positive", "negative"}
DIRECTIONS = {"higher", "lower"}


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
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


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


def load_score_rows(
    paths: Iterable[str | Path],
) -> tuple[list[dict[str, Any]], str, str]:
    rows: list[dict[str, Any]] = []
    seen_ids = set()
    directions = set()
    detectors = set()
    for row in iter_jsonl(paths):
        sample_id = row.get("id")
        if sample_id is None:
            raise ValueError("Detector-score row is missing 'id'")
        sample_id = str(sample_id)
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate detector-score id: {sample_id}")
        seen_ids.add(sample_id)

        label = row.get("label")
        if label not in LABELS:
            raise ValueError(f"Row {sample_id} label must be 'positive' or 'negative'")
        direction = row.get("direction")
        if direction not in DIRECTIONS:
            raise ValueError(f"Row {sample_id} direction must be 'higher' or 'lower'")
        directions.add(direction)
        detector = row.get("detector")
        if not isinstance(detector, str) or not detector.strip():
            raise ValueError(f"Row {sample_id} needs a non-empty detector identifier")
        detectors.add(detector)
        rows.append(
            {
                "id": sample_id,
                "label": label,
                "score": finite_number(row.get("score"), f"Row {sample_id} score"),
                "direction": direction,
                "detector": detector,
            }
        )

    if not rows:
        raise ValueError("No detector-score rows found")
    if len(directions) != 1:
        raise ValueError("All detector-score rows must use the same direction")
    if len(detectors) != 1:
        raise ValueError("All detector-score rows must use the same detector")
    return rows, directions.pop(), detectors.pop()


def oriented_score(score: float, direction: str) -> float:
    return score if direction == "higher" else -score


def is_detected(score: float, threshold: float, direction: str) -> bool:
    return score > threshold if direction == "higher" else score < threshold


def calibrate_threshold(
    negative_scores: list[float], target_fpr: float, direction: str
) -> tuple[float, int, float]:
    if not negative_scores:
        raise ValueError("Threshold calibration needs at least one negative score")
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("target_fpr must satisfy 0 <= target_fpr < 1")
    if direction not in DIRECTIONS:
        raise ValueError("direction must be 'higher' or 'lower'")

    oriented = [oriented_score(score, direction) for score in negative_scores]
    candidates = sorted(set(oriented))
    valid = [
        threshold
        for threshold in candidates
        if sum(score > threshold for score in oriented) / len(oriented) <= target_fpr
    ]
    threshold_oriented = min(valid)
    threshold = threshold_oriented if direction == "higher" else -threshold_oriented
    false_positives = sum(
        is_detected(score, threshold, direction) for score in negative_scores
    )
    return threshold, false_positives, false_positives / len(negative_scores)


def roc_auc(rows: list[dict[str, Any]], direction: str) -> float:
    ranked = sorted(
        (oriented_score(row["score"], direction), row["label"]) for row in rows
    )
    positive_count = sum(label == "positive" for _, label in ranked)
    negative_count = len(ranked) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC needs both positive and negative test rows")

    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == "positive" for _, label in ranked[index:end]
        )
        index = end

    mann_whitney = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return mann_whitney / (positive_count * negative_count)


def detector_counts(
    rows: list[dict[str, Any]], threshold: float, direction: str
) -> dict[str, Any]:
    positives = [row for row in rows if row["label"] == "positive"]
    negatives = [row for row in rows if row["label"] == "negative"]
    if not positives or not negatives:
        raise ValueError("Detector test data needs both positive and negative rows")
    true_positives = sum(
        is_detected(row["score"], threshold, direction) for row in positives
    )
    false_positives = sum(
        is_detected(row["score"], threshold, direction) for row in negatives
    )
    return {
        "n": len(rows),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "tpr": true_positives / len(positives),
        "fpr": false_positives / len(negatives),
    }


def detector_report(
    test_paths: Iterable[str | Path],
    target_fpr: float = 0.01,
    calibration_paths: Iterable[str | Path] | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    test_rows, direction, detector = load_score_rows(test_paths)
    if calibration_paths is None:
        calibration_rows = [row for row in test_rows if row["label"] == "negative"]
        calibration_source = "test_negatives"
    else:
        calibration_rows, calibration_direction, calibration_detector = load_score_rows(
            calibration_paths
        )
        if calibration_direction != direction:
            raise ValueError("Calibration and test score directions differ")
        if calibration_detector != detector:
            raise ValueError("Calibration and test detectors differ")
        if any(row["label"] != "negative" for row in calibration_rows):
            raise ValueError("Separate calibration data must contain only negative rows")
        overlap = sorted(
            {row["id"] for row in test_rows}
            & {row["id"] for row in calibration_rows}
        )
        if overlap:
            raise ValueError(
                f"Calibration and test sample IDs must be disjoint: {overlap}"
            )
        calibration_source = "separate_negative_set"

    negative_scores = [row["score"] for row in calibration_rows]
    threshold, calibration_fp, calibration_fpr = calibrate_threshold(
        negative_scores, target_fpr, direction
    )
    test = detector_counts(test_rows, threshold, direction)
    calibration_resolution = 1.0 / len(negative_scores)
    test_resolution = 1.0 / test["n_negative"]
    calibration_fpr_upper = binomial_upper_bound(
        calibration_fp, len(negative_scores), confidence
    )
    test_fpr_upper = binomial_upper_bound(
        test["false_positives"], test["n_negative"], confidence
    )
    fixed_fpr_reasons = []
    if calibration_source != "separate_negative_set":
        fixed_fpr_reasons.append("A separate negative calibration set is required")
    if calibration_fpr_upper > target_fpr:
        fixed_fpr_reasons.append("Calibration FPR upper confidence bound exceeds target_fpr")
    if test_fpr_upper > target_fpr:
        fixed_fpr_reasons.append("Test FPR upper confidence bound exceeds target_fpr")
    if test["fpr"] > target_fpr:
        fixed_fpr_reasons.append("Test FPR exceeds target_fpr at the locked threshold")
    fixed_fpr_available = not fixed_fpr_reasons
    return {
        "metric": "tpr_at_target_fpr",
        "detector": detector,
        "target_fpr": target_fpr,
        "confidence": confidence,
        "score_direction": direction,
        "threshold": threshold,
        "threshold_rule": "score > threshold" if direction == "higher" else "score < threshold",
        "calibration": {
            "source": calibration_source,
            "n_negative": len(negative_scores),
            "empirical_fpr_resolution": calibration_resolution,
            "target_below_empirical_resolution": target_fpr < calibration_resolution,
            "false_positives": calibration_fp,
            "fpr": calibration_fpr,
            "fpr_upper_confidence_bound": calibration_fpr_upper,
        },
        "test": {
            **test,
            "empirical_fpr_resolution": test_resolution,
            "tpr_at_locked_threshold": test["tpr"],
            "tpr_at_target_fpr": test["tpr"] if fixed_fpr_available else None,
            "fixed_fpr_status": {
                "available": fixed_fpr_available,
                "reasons": fixed_fpr_reasons,
            },
            "actual_fpr": test["fpr"],
            "fpr_upper_confidence_bound": test_fpr_upper,
            "roc_auc": roc_auc(test_rows, direction),
        },
        "protocol": {
            "calibration": "Threshold is chosen from negative scores only.",
            "finite_sample_rule": "Use the least strict observed-score threshold whose empirical calibration FPR does not exceed target_fpr.",
            "comparison": "Strict > for higher-is-positive; strict < for lower-is-positive.",
            "test": "TPR, actual FPR, and ROC-AUC are reported on the test rows.",
            "fixed_fpr": "The headline TPR is unavailable when finite negative samples cannot resolve target_fpr or test FPR exceeds it.",
        },
    }


def kgw_z_score(green_count: int, token_count: int, gamma: float) -> float:
    if isinstance(green_count, bool) or not isinstance(green_count, int):
        raise ValueError("green_count must be an integer")
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise ValueError("token_count must be an integer")
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not 0 <= green_count <= token_count:
        raise ValueError("green_count must satisfy 0 <= green_count <= token_count")
    gamma = finite_number(gamma, "gamma")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must satisfy 0 < gamma < 1")
    return (green_count - gamma * token_count) / math.sqrt(
        token_count * gamma * (1.0 - gamma)
    )


def kgw_p_value(z_score: float) -> float:
    z_score = finite_number(z_score, "z_score")
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def load_fingerprint_rows(
    paths: Iterable[str | Path], require_model_id: bool
) -> tuple[list[dict[str, str]], str]:
    rows: list[dict[str, str]] = []
    seen = set()
    detectors = set()
    for row in iter_jsonl(paths):
        trigger_id = row.get("id")
        if trigger_id is None:
            raise ValueError("Fingerprint row is missing 'id'")
        trigger_id = str(trigger_id)
        model_id = row.get("model_id") if require_model_id else "suspect"
        if require_model_id and (model_id is None or not str(model_id)):
            raise ValueError(f"Fingerprint row {trigger_id} is missing 'model_id'")
        model_id = str(model_id)
        detector = row.get("detector")
        if not isinstance(detector, str) or not detector.strip():
            raise ValueError(
                f"Fingerprint row {model_id}/{trigger_id} needs a detector identifier"
            )
        detectors.add(detector)
        key = (model_id, trigger_id)
        if key in seen:
            raise ValueError(f"Duplicate fingerprint row: {model_id}/{trigger_id}")
        seen.add(key)
        for field in ("response", "expected"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"Fingerprint row {model_id}/{trigger_id} needs text field {field!r}")
        rows.append(
            {
                "model_id": model_id,
                "id": trigger_id,
                "response": row["response"],
                "expected": row["expected"],
                "detector": detector,
            }
        )
    if not rows:
        raise ValueError("No fingerprint rows found")
    if len(detectors) != 1:
        raise ValueError("All fingerprint rows must use the same detector")
    return rows, detectors.pop()


def exact_match_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    matches = sum(row["response"] == row["expected"] for row in rows)
    return {"fsr": matches / len(rows), "matches": matches, "n": len(rows)}


def fingerprint_report(
    suspect_paths: Iterable[str | Path],
    negative_paths: Iterable[str | Path],
    threshold: float,
    confidence: float = 0.95,
) -> dict[str, Any]:
    threshold = finite_number(threshold, "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must satisfy 0 <= threshold <= 1")

    suspect_rows, suspect_detector = load_fingerprint_rows(
        suspect_paths, require_model_id=False
    )
    negative_rows, negative_detector = load_fingerprint_rows(
        negative_paths, require_model_id=True
    )
    if suspect_detector != negative_detector:
        raise ValueError("Suspect and negative fingerprint detectors differ")
    suspect_map = {row["id"]: row["expected"] for row in suspect_rows}
    negative_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in negative_rows:
        negative_by_model[row["model_id"]].append(row)

    negative_models = []
    false_positive_models = 0
    for model_id, rows in sorted(negative_by_model.items()):
        row_map = {row["id"]: row["expected"] for row in rows}
        if row_map != suspect_map:
            raise ValueError(
                f"Negative model {model_id} must use the suspect's exact trigger ids and expected responses"
            )
        summary = exact_match_summary(rows)
        flagged = summary["fsr"] > threshold
        false_positive_models += flagged
        negative_models.append({"model_id": model_id, **summary, "flagged": flagged})

    suspect = exact_match_summary(suspect_rows)
    fpr = false_positive_models / len(negative_models)
    fpr_upper = binomial_upper_bound(
        false_positive_models, len(negative_models), confidence
    )
    return {
        "metric": "fingerprint_exact_match_fsr",
        "detector": suspect_detector,
        "match_rule": "response == expected (exact string equality)",
        "threshold": threshold,
        "threshold_rule": "FSR > threshold",
        "suspect": {**suspect, "flagged": suspect["fsr"] > threshold},
        "negative_models": negative_models,
        "negative_model_count": len(negative_models),
        "false_positive_models": false_positive_models,
        "fpr": fpr,
        "uniqueness": 1.0 - fpr,
        "confidence": confidence,
        "fpr_upper_confidence_bound": fpr_upper,
        "uniqueness_lower_confidence_bound": 1.0 - fpr_upper,
        "empirical_fpr_resolution": 1.0 / len(negative_models),
    }


def command_detector(args: argparse.Namespace) -> int:
    write_report(
        detector_report(
            args.inputs,
            args.target_fpr,
            args.calibration,
            confidence=args.confidence,
        ),
        args.output,
    )
    return 0


def command_kgw(args: argparse.Namespace) -> int:
    z_score = kgw_z_score(args.green_count, args.token_count, args.gamma)
    write_report(
        {
            "metric": "kgw_one_sided_test",
            "green_count": args.green_count,
            "token_count": args.token_count,
            "gamma": args.gamma,
            "z_score": z_score,
            "p_value": kgw_p_value(z_score),
        },
        args.output,
    )
    return 0


def command_fingerprint(args: argparse.Namespace) -> int:
    write_report(
        fingerprint_report(
            args.suspect,
            args.negatives,
            args.threshold,
            confidence=args.confidence,
        ),
        args.output,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Matrix M4 passive provenance metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detector = subparsers.add_parser("detector", help="Compute TPR at a fixed target FPR and ROC-AUC.")
    detector.add_argument("inputs", nargs="+", help="Detector-score test JSONL files.")
    detector.add_argument(
        "--calibration",
        nargs="+",
        help="Optional negative-only calibration JSONL files. Test negatives are used if omitted.",
    )
    detector.add_argument("--target-fpr", type=float, default=0.01)
    detector.add_argument("--confidence", type=float, required=True)
    detector.add_argument("--output", help="Write JSON report to this path.")
    detector.set_defaults(func=command_detector)

    kgw = subparsers.add_parser("kgw", help="Compute a KGW z-score and one-sided p-value.")
    kgw.add_argument("--green-count", type=int, required=True)
    kgw.add_argument("--token-count", type=int, required=True)
    kgw.add_argument("--gamma", type=float, required=True)
    kgw.add_argument("--output", help="Write JSON report to this path.")
    kgw.set_defaults(func=command_kgw)

    fingerprint = subparsers.add_parser("fingerprint", help="Compute exact-match FSR and uniqueness.")
    fingerprint.add_argument("suspect", nargs="+", help="Suspect-model trigger JSONL files.")
    fingerprint.add_argument(
        "--negatives", nargs="+", required=True, help="Negative-model trigger JSONL files."
    )
    fingerprint.add_argument("--threshold", type=float, required=True)
    fingerprint.add_argument("--confidence", type=float, required=True)
    fingerprint.add_argument("--output", help="Write JSON report to this path.")
    fingerprint.set_defaults(func=command_fingerprint)
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
