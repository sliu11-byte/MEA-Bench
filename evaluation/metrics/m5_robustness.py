#!/usr/bin/env python3
"""Matrix M5 before-to-after robustness metrics.

M5 never calibrates a threshold. It reads the threshold and protocol from one
frozen M4 JSON report, and the same detector must produce both score files.

Detector-score rows:
  {"id":"p1","label":"positive","score":2.4,
   "direction":"higher","detector":"kgw-v1"}

Fingerprint rows:
  {"id":"trigger-1","response":"blue","expected":"blue",
   "detector":"fingerprint-v1"}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
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


def load_m4_report(
    source: str | Path | dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if isinstance(source, dict):
        payload = source
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    else:
        path = Path(source)
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path} is not a valid M4 JSON report") from exc
    if not isinstance(payload, dict) or payload.get("metric") != "tpr_at_target_fpr":
        raise ValueError("m4_report must be an M4 tpr_at_target_fpr report")
    detector = payload.get("detector")
    if not isinstance(detector, str) or not detector.strip():
        raise ValueError("m4_report is missing detector")
    direction = payload.get("score_direction")
    if direction not in DIRECTIONS:
        raise ValueError("m4_report has an invalid score_direction")
    threshold = finite_number(payload.get("threshold"), "m4_report threshold")
    target_fpr = finite_number(payload.get("target_fpr"), "m4_report target_fpr")
    confidence = finite_number(payload.get("confidence"), "m4_report confidence")
    if not 0.0 <= target_fpr < 1.0:
        raise ValueError("m4_report target_fpr must satisfy 0 <= value < 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("m4_report confidence must satisfy 0 < value < 1")
    test = payload.get("test")
    status = test.get("fixed_fpr_status") if isinstance(test, dict) else None
    if not isinstance(status, dict) or not isinstance(status.get("available"), bool):
        raise ValueError("m4_report is missing test.fixed_fpr_status.available")
    return {
        "detector": detector,
        "score_direction": direction,
        "threshold": threshold,
        "target_fpr": target_fpr,
        "confidence": confidence,
        "fixed_fpr_status": status,
    }, hashlib.sha256(raw).hexdigest()


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


def binomial_upper_bound(successes: int, trials: int, confidence: float) -> float:
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


def is_detected(score: float, threshold: float, direction: str) -> bool:
    return score > threshold if direction == "higher" else score < threshold


def load_detector_rows(
    paths: Iterable[str | Path], stage: str
) -> tuple[dict[str, dict[str, Any]], str, str]:
    rows: dict[str, dict[str, Any]] = {}
    directions = set()
    detectors = set()
    for row in iter_jsonl(paths):
        sample_id = row.get("id")
        if sample_id is None:
            raise ValueError(f"{stage} detector row is missing 'id'")
        sample_id = str(sample_id)
        if sample_id in rows:
            raise ValueError(f"Duplicate {stage} detector id: {sample_id}")

        label = row.get("label")
        if label not in LABELS:
            raise ValueError(f"{stage} row {sample_id} has an invalid label")
        direction = row.get("direction")
        if direction not in DIRECTIONS:
            raise ValueError(f"{stage} row {sample_id} direction must be 'higher' or 'lower'")
        detector = row.get("detector")
        if detector is None or not str(detector):
            raise ValueError(f"{stage} row {sample_id} is missing 'detector'")
        detector = str(detector)
        directions.add(direction)
        detectors.add(detector)
        rows[sample_id] = {
            "id": sample_id,
            "label": label,
            "score": finite_number(row.get("score"), f"{stage} row {sample_id} score"),
            "direction": direction,
            "detector": detector,
        }

    if not rows:
        raise ValueError(f"No {stage} detector rows found")
    if len(directions) != 1:
        raise ValueError(f"All {stage} rows must use one score direction")
    if len(detectors) != 1:
        raise ValueError(f"All {stage} rows must use one detector")
    return rows, directions.pop(), detectors.pop()


def require_same_ids(before: dict[str, Any], after: dict[str, Any]) -> None:
    before_ids = set(before)
    after_ids = set(after)
    if before_ids != after_ids:
        missing_after = sorted(before_ids - after_ids)
        missing_before = sorted(after_ids - before_ids)
        raise ValueError(
            "Before/after sample sets differ: "
            f"missing_after={missing_after}, missing_before={missing_before}"
        )


def roc_auc(rows: dict[str, dict[str, Any]], direction: str) -> float:
    ranked = sorted(
        ((row["score"] if direction == "higher" else -row["score"]), row["label"])
        for row in rows.values()
    )
    positive_count = sum(label == "positive" for _, label in ranked)
    negative_count = len(ranked) - positive_count
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


def detector_snapshot(
    rows: dict[str, dict[str, Any]],
    threshold: float,
    direction: str,
    confidence: float,
) -> dict[str, Any]:
    positives = [row for row in rows.values() if row["label"] == "positive"]
    negatives = [row for row in rows.values() if row["label"] == "negative"]
    if not positives or not negatives:
        raise ValueError("M5 detector data needs both positive and negative rows")
    true_positives = sum(
        is_detected(row["score"], threshold, direction) for row in positives
    )
    false_positives = sum(
        is_detected(row["score"], threshold, direction) for row in negatives
    )
    return {
        "tpr": true_positives / len(positives),
        "actual_fpr": false_positives / len(negatives),
        "fpr_upper_confidence_bound": binomial_upper_bound(
            false_positives, len(negatives), confidence
        ),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "roc_auc": roc_auc(rows, direction),
    }


def retention_values(before: float, after: float, complement_name: str) -> dict[str, Any]:
    return {
        "before": before,
        "after": after,
        "signal_loss": before - after,
        "retention_ratio": None if before == 0.0 else after / before,
        complement_name: 1.0 - after,
    }


def detector_retention_report(
    before_paths: Iterable[str | Path],
    after_paths: Iterable[str | Path],
    *,
    countermeasure: str,
    m4_report: str | Path | dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(countermeasure, str) or not countermeasure.strip():
        raise ValueError("countermeasure must be non-empty text")
    frozen_m4, m4_report_sha256 = load_m4_report(m4_report)
    threshold = frozen_m4["threshold"]
    target_fpr = frozen_m4["target_fpr"]
    confidence = frozen_m4["confidence"]
    before_rows, before_direction, before_detector = load_detector_rows(
        before_paths, "before"
    )
    after_rows, after_direction, after_detector = load_detector_rows(after_paths, "after")
    require_same_ids(before_rows, after_rows)
    if before_direction != after_direction:
        raise ValueError("Before/after score directions differ")
    if before_detector != after_detector:
        raise ValueError("Before/after detectors differ")
    if before_detector != frozen_m4["detector"]:
        raise ValueError("M5 detector differs from the frozen M4 report")
    if before_direction != frozen_m4["score_direction"]:
        raise ValueError("M5 score direction differs from the frozen M4 report")
    for sample_id in before_rows:
        if before_rows[sample_id]["label"] != after_rows[sample_id]["label"]:
            raise ValueError(f"Before/after label differs for id {sample_id}")

    before_snapshot = detector_snapshot(
        before_rows, threshold, before_direction, confidence
    )
    after_snapshot = detector_snapshot(
        after_rows, threshold, before_direction, confidence
    )
    fixed_fpr_reasons = []
    if not frozen_m4["fixed_fpr_status"]["available"]:
        fixed_fpr_reasons.append(
            "Frozen M4 report did not validate its fixed-FPR headline"
        )
    if before_snapshot["fpr_upper_confidence_bound"] > target_fpr:
        fixed_fpr_reasons.append(
            "Before FPR upper confidence bound exceeds target_fpr"
        )
    if after_snapshot["fpr_upper_confidence_bound"] > target_fpr:
        fixed_fpr_reasons.append(
            "After FPR upper confidence bound exceeds target_fpr"
        )
    if before_snapshot["actual_fpr"] > target_fpr:
        fixed_fpr_reasons.append("Before actual FPR exceeds target_fpr")
    if after_snapshot["actual_fpr"] > target_fpr:
        fixed_fpr_reasons.append("After actual FPR exceeds target_fpr")
    fixed_fpr_available = not fixed_fpr_reasons
    locked_retention = retention_values(
        before_snapshot["tpr"], after_snapshot["tpr"], "evasion_rate"
    )
    if fixed_fpr_available:
        fixed_fpr_retention = {
            "available": True,
            "reasons": [],
            **locked_retention,
        }
    else:
        fixed_fpr_retention = {
            "available": False,
            "reasons": fixed_fpr_reasons,
            "before": None,
            "after": None,
            "signal_loss": None,
            "retention_ratio": None,
            "evasion_rate": None,
        }
    return {
        "metric": "m5_detector_retention",
        "countermeasure": countermeasure,
        "detector": before_detector,
        "score_direction": before_direction,
        "threshold": threshold,
        "target_fpr": target_fpr,
        "confidence": confidence,
        "threshold_rule": (
            "score > threshold" if before_direction == "higher" else "score < threshold"
        ),
        "threshold_protocol": "locked before countermeasure; M5 does not recalibrate",
        "threshold_provenance": "read from the frozen M4 report",
        "m4_report_sha256": m4_report_sha256,
        "m4_fixed_fpr_status": frozen_m4["fixed_fpr_status"],
        "n": len(before_rows),
        "n_positive": sum(row["label"] == "positive" for row in before_rows.values()),
        "n_negative": sum(row["label"] == "negative" for row in before_rows.values()),
        **locked_retention,
        "reported_value_basis": "tpr_at_locked_threshold",
        "fixed_fpr_retention": fixed_fpr_retention,
        "before_guard": before_snapshot,
        "after_guard": after_snapshot,
        "protocol": {
            "pairing": "Same sample IDs, labels, detector, direction, and locked threshold.",
            "provenance": "Detector, direction, threshold, target FPR, and confidence are locked by the hashed M4 report.",
            "evasion_rate": "1 - after TPR; this is distinct from 1 - retention_ratio.",
            "fixed_fpr": "The fixed-FPR headline is null unless both before and after one-sided FPR upper confidence bounds stay at or below target_fpr.",
            "scope": "M5 only; quality/ability guards belong to M1/M3 result interpretation.",
        },
    }


def load_fingerprint_rows(
    paths: Iterable[str | Path], stage: str
) -> tuple[dict[str, dict[str, str]], str]:
    rows: dict[str, dict[str, str]] = {}
    detectors = set()
    for row in iter_jsonl(paths):
        trigger_id = row.get("id")
        if trigger_id is None:
            raise ValueError(f"{stage} fingerprint row is missing 'id'")
        trigger_id = str(trigger_id)
        if trigger_id in rows:
            raise ValueError(f"Duplicate {stage} fingerprint id: {trigger_id}")
        detector = row.get("detector")
        if detector is None or not str(detector):
            raise ValueError(f"{stage} fingerprint row {trigger_id} is missing 'detector'")
        detector = str(detector)
        detectors.add(detector)
        for field in ("response", "expected"):
            if not isinstance(row.get(field), str):
                raise ValueError(
                    f"{stage} fingerprint row {trigger_id} needs text field {field!r}"
                )
        rows[trigger_id] = {
            "id": trigger_id,
            "response": row["response"],
            "expected": row["expected"],
            "detector": detector,
        }
    if not rows:
        raise ValueError(f"No {stage} fingerprint rows found")
    if len(detectors) != 1:
        raise ValueError(f"All {stage} fingerprint rows must use one detector")
    return rows, detectors.pop()


def load_m4_fingerprint_report(
    source: str | Path | dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if isinstance(source, dict):
        payload = source
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    else:
        path = Path(source)
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path} is not a valid M4 fingerprint report") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("metric") != "fingerprint_exact_match_fsr"
    ):
        raise ValueError("m4_report must be an M4 fingerprint_exact_match_fsr report")
    detector = payload.get("detector")
    if not isinstance(detector, str) or not detector.strip():
        raise ValueError("m4_report is missing fingerprint detector")
    threshold = finite_number(payload.get("threshold"), "m4_report threshold")
    confidence = finite_number(payload.get("confidence"), "m4_report confidence")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("m4_report threshold must satisfy 0 <= value <= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("m4_report confidence must satisfy 0 < value < 1")
    return {
        "detector": detector,
        "threshold": threshold,
        "confidence": confidence,
    }, hashlib.sha256(raw).hexdigest()


def load_fingerprint_negative_rows(
    paths: Iterable[str | Path], stage: str
) -> tuple[dict[str, dict[str, dict[str, str]]], str]:
    by_model: dict[str, dict[str, dict[str, str]]] = {}
    detectors = set()
    for row in iter_jsonl(paths):
        model_id = row.get("model_id")
        trigger_id = row.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"{stage} negative row needs a model_id")
        if trigger_id is None:
            raise ValueError(f"{stage} negative row needs an id")
        trigger_id = str(trigger_id)
        detector = row.get("detector")
        if not isinstance(detector, str) or not detector.strip():
            raise ValueError(f"{stage} negative row needs a detector")
        detectors.add(detector)
        for field in ("response", "expected"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"{stage} negative row needs text field {field!r}")
        model_rows = by_model.setdefault(model_id, {})
        if trigger_id in model_rows:
            raise ValueError(f"Duplicate {stage} negative row: {model_id}/{trigger_id}")
        model_rows[trigger_id] = {
            "response": row["response"],
            "expected": row["expected"],
        }
    if not by_model:
        raise ValueError(f"No {stage} negative fingerprint rows found")
    if len(detectors) != 1:
        raise ValueError(f"All {stage} negative rows must use one detector")
    return by_model, detectors.pop()


def validate_negative_pool(
    pool: dict[str, dict[str, dict[str, str]]],
    suspect: dict[str, dict[str, str]],
    stage: str,
) -> None:
    expected = {trigger_id: row["expected"] for trigger_id, row in suspect.items()}
    for model_id, rows in pool.items():
        model_expected = {
            trigger_id: row["expected"] for trigger_id, row in rows.items()
        }
        if model_expected != expected:
            raise ValueError(
                f"{stage} negative model {model_id} must use the exact suspect triggers"
            )


def negative_pool_snapshot(
    pool: dict[str, dict[str, dict[str, str]]],
    threshold: float,
    confidence: float,
) -> dict[str, Any]:
    models = []
    false_positive_models = 0
    for model_id, rows in sorted(pool.items()):
        matches = sum(
            row["response"] == row["expected"] for row in rows.values()
        )
        fsr = matches / len(rows)
        flagged = fsr > threshold
        false_positive_models += flagged
        models.append(
            {
                "model_id": model_id,
                "fsr": fsr,
                "matches": matches,
                "n": len(rows),
                "flagged": flagged,
            }
        )
    count = len(models)
    fpr = false_positive_models / count
    fpr_upper = binomial_upper_bound(false_positive_models, count, confidence)
    return {
        "negative_models": models,
        "negative_model_count": count,
        "false_positive_models": false_positive_models,
        "fpr": fpr,
        "uniqueness": 1.0 - fpr,
        "fpr_upper_confidence_bound": fpr_upper,
        "uniqueness_lower_confidence_bound": 1.0 - fpr_upper,
    }


def fingerprint_retention_report(
    before_paths: Iterable[str | Path],
    after_paths: Iterable[str | Path],
    *,
    countermeasure: str,
    m4_report: str | Path | dict[str, Any],
    before_negative_paths: Iterable[str | Path],
    after_negative_paths: Iterable[str | Path],
) -> dict[str, Any]:
    if not isinstance(countermeasure, str) or not countermeasure.strip():
        raise ValueError("countermeasure must be non-empty text")
    frozen_m4, m4_report_sha256 = load_m4_fingerprint_report(m4_report)
    threshold = frozen_m4["threshold"]
    confidence = frozen_m4["confidence"]
    before_rows, before_detector = load_fingerprint_rows(before_paths, "before")
    after_rows, after_detector = load_fingerprint_rows(after_paths, "after")
    require_same_ids(before_rows, after_rows)
    if before_detector != after_detector:
        raise ValueError("Before/after fingerprint detectors differ")
    if before_detector != frozen_m4["detector"]:
        raise ValueError("M5 fingerprint detector differs from the frozen M4 report")
    for trigger_id in before_rows:
        if before_rows[trigger_id]["expected"] != after_rows[trigger_id]["expected"]:
            raise ValueError(f"Before/after expected response differs for id {trigger_id}")

    before_negative, before_negative_detector = load_fingerprint_negative_rows(
        before_negative_paths, "before"
    )
    after_negative, after_negative_detector = load_fingerprint_negative_rows(
        after_negative_paths, "after"
    )
    if before_negative_detector != before_detector:
        raise ValueError("Before suspect and negative fingerprint detectors differ")
    if after_negative_detector != before_detector:
        raise ValueError("After suspect and negative fingerprint detectors differ")
    if set(before_negative) != set(after_negative):
        raise ValueError("Before/after negative model pools differ")
    validate_negative_pool(before_negative, before_rows, "before")
    validate_negative_pool(after_negative, after_rows, "after")

    before_matches = sum(
        row["response"] == row["expected"] for row in before_rows.values()
    )
    after_matches = sum(
        row["response"] == row["expected"] for row in after_rows.values()
    )
    before_fsr = before_matches / len(before_rows)
    after_fsr = after_matches / len(after_rows)
    before_uniqueness = negative_pool_snapshot(
        before_negative, threshold, confidence
    )
    after_uniqueness = negative_pool_snapshot(
        after_negative, threshold, confidence
    )
    return {
        "metric": "m5_fingerprint_retention",
        "countermeasure": countermeasure,
        "detector": before_detector,
        "match_rule": "response == expected (exact string equality)",
        "threshold": threshold,
        "confidence": confidence,
        "threshold_rule": "FSR > threshold",
        "threshold_protocol": "locked before countermeasure; M5 does not recalibrate",
        "threshold_provenance": "read from the frozen M4 fingerprint report",
        "m4_report_sha256": m4_report_sha256,
        "n": len(before_rows),
        **retention_values(before_fsr, after_fsr, "removal_rate"),
        "before_matches": before_matches,
        "after_matches": after_matches,
        "flagged_before": before_fsr > threshold,
        "flagged_after": after_fsr > threshold,
        "uniqueness_before": before_uniqueness,
        "uniqueness_after": after_uniqueness,
        "uniqueness_change": (
            after_uniqueness["uniqueness"] - before_uniqueness["uniqueness"]
        ),
        "protocol": {
            "pairing": "Same suspect triggers, expected responses, detector, negative model IDs, and locked threshold.",
            "provenance": "Detector, FSR threshold, and confidence are locked by the hashed M4 fingerprint report.",
            "removal_rate": "1 - after FSR; this is distinct from 1 - retention_ratio.",
            "match": "Exact string equality; no normalization or fuzzy matching.",
            "uniqueness": "1 - negative-model FPR is recomputed before and after the countermeasure.",
        },
    }


def command_detector(args: argparse.Namespace) -> int:
    write_report(
        detector_retention_report(
            args.before,
            args.after,
            countermeasure=args.countermeasure,
            m4_report=args.m4_report,
        ),
        args.output,
    )
    return 0


def command_fingerprint(args: argparse.Namespace) -> int:
    write_report(
        fingerprint_retention_report(
            args.before,
            args.after,
            countermeasure=args.countermeasure,
            m4_report=args.m4_report,
            before_negative_paths=args.before_negatives,
            after_negative_paths=args.after_negatives,
        ),
        args.output,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Matrix M5 before-to-after robustness metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detector = subparsers.add_parser(
        "detector", help="Compare detector TPR before and after a countermeasure."
    )
    detector.add_argument("before", nargs="+", help="Before-countermeasure score JSONL files.")
    detector.add_argument("--after", nargs="+", required=True, help="After-countermeasure score JSONL files.")
    detector.add_argument("--m4-report", required=True, help="Frozen M4 detector report JSON.")
    detector.add_argument("--countermeasure", required=True)
    detector.add_argument("--output", help="Write JSON report to this path.")
    detector.set_defaults(func=command_detector)

    fingerprint = subparsers.add_parser(
        "fingerprint", help="Compare exact-match FSR before and after a countermeasure."
    )
    fingerprint.add_argument("before", nargs="+", help="Before-countermeasure trigger JSONL files.")
    fingerprint.add_argument("--after", nargs="+", required=True, help="After-countermeasure trigger JSONL files.")
    fingerprint.add_argument("--m4-report", required=True, help="Frozen M4 fingerprint report JSON.")
    fingerprint.add_argument("--before-negatives", nargs="+", required=True)
    fingerprint.add_argument("--after-negatives", nargs="+", required=True)
    fingerprint.add_argument("--countermeasure", required=True)
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
