import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics.m5_robustness import (
    binomial_upper_bound,
    detector_retention_report,
    fingerprint_retention_report,
)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def detector_row(sample_id, label, score, detector="kgw-v1"):
    return {
        "id": sample_id,
        "label": label,
        "score": score,
        "direction": "higher",
        "detector": detector,
    }


def locked_m4_report(*, detector="kgw-v1", available=True):
    return {
        "metric": "tpr_at_target_fpr",
        "detector": detector,
        "score_direction": "higher",
        "threshold": 0.5,
        "target_fpr": 0.01,
        "confidence": 0.95,
        "test": {
            "fixed_fpr_status": {
                "available": available,
                "reasons": [] if available else ["insufficient negative evidence"],
            }
        },
    }


def locked_m4_fingerprint_report(detector="fingerprint-v1"):
    return {
        "metric": "fingerprint_exact_match_fsr",
        "detector": detector,
        "threshold": 0.5,
        "confidence": 0.95,
    }


class M5MetricsTest(unittest.TestCase):
    def test_large_sample_binomial_upper_bound_is_stable(self):
        self.assertAlmostEqual(
            binomial_upper_bound(950, 100_000, 0.95),
            0.01002024,
            places=7,
        )

    def test_detector_reports_before_after_loss_and_evasion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = root / "before.jsonl"
            after = root / "after.jsonl"
            write_jsonl(
                before,
                [
                    detector_row("p1", "positive", 0.9),
                    detector_row("p2", "positive", 0.8),
                    detector_row("n1", "negative", 0.1),
                    detector_row("n2", "negative", 0.2),
                ],
            )
            write_jsonl(
                after,
                [
                    detector_row("p1", "positive", 0.7),
                    detector_row("p2", "positive", 0.4),
                    detector_row("n1", "negative", 0.1),
                    detector_row("n2", "negative", 0.2),
                ],
            )
            report = detector_retention_report(
                [before], [after], countermeasure="paraphrase",
                m4_report=locked_m4_report(),
            )
            self.assertEqual(report["countermeasure"], "paraphrase")
            self.assertEqual(report["before"], 1.0)
            self.assertEqual(report["after"], 0.5)
            self.assertEqual(report["signal_loss"], 0.5)
            self.assertEqual(report["retention_ratio"], 0.5)
            self.assertEqual(report["evasion_rate"], 0.5)
            self.assertEqual(report["before_guard"]["roc_auc"], 1.0)
            self.assertFalse(report["fixed_fpr_retention"]["available"])
            self.assertIsNone(report["fixed_fpr_retention"]["after"])

    def test_before_zero_makes_retention_ratio_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = root / "before.jsonl"
            after = root / "after.jsonl"
            rows = [
                detector_row("p1", "positive", 0.1),
                detector_row("n1", "negative", 0.1),
            ]
            write_jsonl(before, rows)
            write_jsonl(after, rows)
            report = detector_retention_report(
                [before], [after], countermeasure="paraphrase",
                m4_report=locked_m4_report(),
            )
            self.assertIsNone(report["retention_ratio"])

    def test_mismatched_sample_sets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = root / "before.jsonl"
            after = root / "after.jsonl"
            write_jsonl(
                before,
                [detector_row("p1", "positive", 0.9), detector_row("n1", "negative", 0.1)],
            )
            write_jsonl(
                after,
                [detector_row("p2", "positive", 0.9), detector_row("n1", "negative", 0.1)],
            )
            with self.assertRaisesRegex(ValueError, "sample sets differ"):
                detector_retention_report(
                    [before], [after], countermeasure="paraphrase",
                    m4_report=locked_m4_report(),
                )

    def test_changed_detector_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = root / "before.jsonl"
            after = root / "after.jsonl"
            write_jsonl(
                before,
                [detector_row("p1", "positive", 0.9), detector_row("n1", "negative", 0.1)],
            )
            write_jsonl(
                after,
                [
                    detector_row("p1", "positive", 0.9, "other-detector"),
                    detector_row("n1", "negative", 0.1, "other-detector"),
                ],
            )
            with self.assertRaisesRegex(ValueError, "detectors differ"):
                detector_retention_report(
                    [before], [after], countermeasure="paraphrase",
                    m4_report=locked_m4_report(),
                )

    def test_fixed_fpr_retention_needs_enough_negative_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = root / "before.jsonl"
            after = root / "after.jsonl"
            negatives = [detector_row(f"n{index}", "negative", 0.1) for index in range(299)]
            write_jsonl(before, [detector_row("p1", "positive", 0.9), *negatives])
            write_jsonl(after, [detector_row("p1", "positive", 0.8), *negatives])
            report = detector_retention_report(
                [before], [after], countermeasure="paraphrase",
                m4_report=locked_m4_report(),
            )
            self.assertTrue(report["fixed_fpr_retention"]["available"])
            self.assertEqual(report["fixed_fpr_retention"]["after"], 1.0)

    def test_detector_must_match_frozen_m4_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = root / "before.jsonl"
            after = root / "after.jsonl"
            rows = [
                detector_row("p1", "positive", 0.9),
                detector_row("n1", "negative", 0.1),
            ]
            write_jsonl(before, rows)
            write_jsonl(after, rows)
            with self.assertRaisesRegex(ValueError, "frozen M4 report"):
                detector_retention_report(
                    [before], [after], countermeasure="paraphrase",
                    m4_report=locked_m4_report(detector="other-v1"),
                )

    def test_fingerprint_reports_fsr_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = root / "before.jsonl"
            after = root / "after.jsonl"
            before_negatives = root / "before-negatives.jsonl"
            after_negatives = root / "after-negatives.jsonl"
            common = {"detector": "fingerprint-v1"}
            write_jsonl(
                before,
                [
                    {"id": "t1", "response": "blue", "expected": "blue", **common},
                    {"id": "t2", "response": "circle", "expected": "circle", **common},
                ],
            )
            write_jsonl(
                after,
                [
                    {"id": "t1", "response": "blue", "expected": "blue", **common},
                    {"id": "t2", "response": "square", "expected": "circle", **common},
                ],
            )
            negative_rows = [
                {"model_id": "base", "id": "t1", "response": "red", "expected": "blue", **common},
                {"model_id": "base", "id": "t2", "response": "square", "expected": "circle", **common},
                {"model_id": "other", "id": "t1", "response": "red", "expected": "blue", **common},
                {"model_id": "other", "id": "t2", "response": "square", "expected": "circle", **common},
            ]
            write_jsonl(before_negatives, negative_rows)
            write_jsonl(after_negatives, negative_rows)
            report = fingerprint_retention_report(
                [before],
                [after],
                countermeasure="fine-tuning",
                m4_report=locked_m4_fingerprint_report(),
                before_negative_paths=[before_negatives],
                after_negative_paths=[after_negatives],
            )
            self.assertEqual(report["before"], 1.0)
            self.assertEqual(report["after"], 0.5)
            self.assertEqual(report["signal_loss"], 0.5)
            self.assertEqual(report["retention_ratio"], 0.5)
            self.assertEqual(report["removal_rate"], 0.5)
            self.assertEqual(report["uniqueness_before"]["uniqueness"], 1.0)
            self.assertEqual(report["uniqueness_after"]["uniqueness"], 1.0)


if __name__ == "__main__":
    unittest.main()

