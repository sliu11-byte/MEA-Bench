import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics.m4_provenance import (
    binomial_upper_bound,
    detector_report,
    fingerprint_report,
    kgw_p_value,
    kgw_z_score,
    roc_auc,
)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def score_row(sample_id, label, score, direction="higher", detector="kgw-v1"):
    return {
        "id": sample_id,
        "label": label,
        "score": score,
        "direction": direction,
        "detector": detector,
    }


class M4MetricsTest(unittest.TestCase):
    def test_binomial_upper_bound_supports_low_fpr_audits(self):
        self.assertLess(binomial_upper_bound(0, 299, 0.95), 0.01)
        self.assertGreater(binomial_upper_bound(0, 298, 0.95), 0.01)
        self.assertAlmostEqual(
            binomial_upper_bound(950, 100_000, 0.95),
            0.01002024,
            places=7,
        )

    def test_detector_uses_separate_conservative_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "calibration.jsonl"
            test = root / "test.jsonl"
            write_jsonl(
                calibration,
                [score_row(f"c{i}", "negative", score) for i, score in enumerate([0.1, 0.2, 0.3, 0.4])],
            )
            write_jsonl(
                test,
                [
                    score_row("p1", "positive", 0.5),
                    score_row("p2", "positive", 0.2),
                    score_row("n1", "negative", 0.4),
                    score_row("n2", "negative", 0.1),
                ],
            )

            report = detector_report([test], target_fpr=0.25, calibration_paths=[calibration])
            self.assertAlmostEqual(report["threshold"], 0.3)
            self.assertEqual(report["calibration"]["fpr"], 0.25)
            self.assertEqual(report["calibration"]["empirical_fpr_resolution"], 0.25)
            self.assertEqual(report["test"]["tpr_at_locked_threshold"], 0.5)
            self.assertIsNone(report["test"]["tpr_at_target_fpr"])
            self.assertFalse(report["test"]["fixed_fpr_status"]["available"])
            self.assertGreater(report["test"]["fpr_upper_confidence_bound"], 0.25)
            self.assertEqual(report["test"]["actual_fpr"], 0.5)
            self.assertEqual(report["test"]["roc_auc"], 0.75)

    def test_lower_scores_can_mean_more_positive(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.jsonl"
            write_jsonl(
                path,
                [
                    score_row("p1", "positive", 0.1, "lower"),
                    score_row("p2", "positive", 0.8, "lower"),
                    score_row("n1", "negative", 0.6, "lower"),
                    score_row("n2", "negative", 0.7, "lower"),
                    score_row("n3", "negative", 0.8, "lower"),
                    score_row("n4", "negative", 0.9, "lower"),
                ],
            )
            report = detector_report([path], target_fpr=0.25)
            self.assertEqual(report["threshold"], 0.7)
            self.assertEqual(report["calibration"]["fpr"], 0.25)

    def test_separate_calibration_and_test_ids_must_be_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "calibration.jsonl"
            test = root / "test.jsonl"
            write_jsonl(calibration, [score_row("shared", "negative", 0.1)])
            write_jsonl(
                test,
                [
                    score_row("p1", "positive", 0.9),
                    score_row("shared", "negative", 0.1),
                ],
            )
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                detector_report([test], calibration_paths=[calibration])

    def test_calibration_and_test_detectors_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "calibration.jsonl"
            test = root / "test.jsonl"
            write_jsonl(calibration, [score_row("c1", "negative", 0.1)])
            write_jsonl(
                test,
                [
                    score_row("p1", "positive", 0.9, detector="other-v1"),
                    score_row("n1", "negative", 0.1, detector="other-v1"),
                ],
            )
            with self.assertRaisesRegex(ValueError, "detectors differ"):
                detector_report([test], calibration_paths=[calibration])

    def test_roc_auc_counts_ties_as_half(self):
        rows = [
            {"label": "positive", "score": 1.0},
            {"label": "negative", "score": 1.0},
        ]
        self.assertEqual(roc_auc(rows, "higher"), 0.5)

    def test_kgw_z_and_p_are_one_sided(self):
        z_score = kgw_z_score(green_count=25, token_count=100, gamma=0.25)
        self.assertEqual(z_score, 0.0)
        self.assertAlmostEqual(kgw_p_value(z_score), 0.5)

    def test_fingerprint_exact_match_and_uniqueness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suspect = root / "suspect.jsonl"
            negatives = root / "negatives.jsonl"
            write_jsonl(
                suspect,
                [
                    {"id": "t1", "response": "blue", "expected": "blue", "detector": "fp-v1"},
                    {"id": "t2", "response": "circle", "expected": "circle", "detector": "fp-v1"},
                ],
            )
            write_jsonl(
                negatives,
                [
                    {"model_id": "base", "id": "t1", "response": "red", "expected": "blue", "detector": "fp-v1"},
                    {"model_id": "base", "id": "t2", "response": "square", "expected": "circle", "detector": "fp-v1"},
                    {"model_id": "other", "id": "t1", "response": "blue", "expected": "blue", "detector": "fp-v1"},
                    {"model_id": "other", "id": "t2", "response": "square", "expected": "circle", "detector": "fp-v1"},
                ],
            )
            report = fingerprint_report([suspect], [negatives], threshold=0.5)
            self.assertEqual(report["suspect"]["fsr"], 1.0)
            self.assertEqual(report["fpr"], 0.0)
            self.assertEqual(report["uniqueness"], 1.0)


if __name__ == "__main__":
    unittest.main()

