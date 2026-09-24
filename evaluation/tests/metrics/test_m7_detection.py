import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics.m7_detection import active_detection_report, binomial_upper_bound, conservative_threshold


def events(
    entity_id,
    label,
    scores,
    attack_type=None,
    detector="active-v1",
    monitor_mode="per-account",
):
    return [
        {
            "entity_id": entity_id,
            "label": label,
            "query_index": index,
            "score": score,
            "detector": detector,
            "monitor_mode": monitor_mode,
            **({"attack_type": attack_type} if attack_type else {}),
        }
        for index, score in enumerate(scores, start=1)
    ]


class M7MetricsTest(unittest.TestCase):
    def test_binomial_upper_bound_requires_about_299_zero_fp_examples_for_one_percent(self):
        self.assertLess(binomial_upper_bound(0, 299, 0.95), 0.01)
        self.assertGreater(binomial_upper_bound(0, 298, 0.95), 0.01)
        self.assertAlmostEqual(
            binomial_upper_bound(950, 100_000, 0.95),
            0.01002024,
            places=7,
        )

    def test_conservative_threshold_respects_finite_sample_fpr(self):
        threshold = conservative_threshold([0.1, 0.2, 0.3, 0.4, 0.5], 0.2, "higher")
        self.assertEqual(threshold, 0.5)
        actual = sum(score >= threshold for score in [0.1, 0.2, 0.3, 0.4, 0.5]) / 5
        self.assertLessEqual(actual, 0.2)

    def test_target_below_resolution_uses_zero_fpr_threshold(self):
        threshold = conservative_threshold([0.1, 0.2], 0.01, "higher")
        self.assertGreater(threshold, 0.2)

    def test_active_detection_keeps_censored_attackers(self):
        calibration = []
        for index, maximum in enumerate([0.1, 0.2, 0.3, 0.4, 0.5], start=1):
            calibration += events(f"cal-{index}", "benign", [maximum] * 3)

        evaluation = (
            events("good-1", "benign", [0.1, 0.2, 0.2])
            + events("good-2", "benign", [0.1, 0.6, 0.6])
            + events("attack-fast", "attacker", [0.2, 0.7], "naive")
            + events("attack-hidden", "attacker", [0.1, 0.2, 0.3], "evasive")
        )
        report = active_detection_report(
            calibration,
            evaluation,
            target_fpr=0.2,
            budget=3,
            checkpoints=[1, 2, 3],
            direction="higher",
            monitor_mode="per-account",
            required_attack_types=("naive",),
        )
        self.assertEqual(report["threshold"], 0.5)
        self.assertEqual(report["actual_fpr"], 0.5)
        self.assertEqual(report["tpr_at_locked_threshold"], 0.5)
        self.assertIsNone(report["tpr_at_fixed_fpr"])
        self.assertFalse(report["fixed_fpr_status"]["available"])
        self.assertIsNone(report["roc_auc"])
        self.assertFalse(report["roc_auc_status"]["available"])
        self.assertEqual(report["calibration"]["benign_entities"], 5)
        self.assertEqual(report["evaluation"]["attacker_entities"], 2)
        self.assertEqual(report["overall"]["censored"], 1)
        by_id = {row["entity_id"]: row for row in report["per_attacker"]}
        self.assertEqual(by_id["attack-fast"]["queries_to_detection"], 2)
        self.assertTrue(by_id["attack-hidden"]["censored"])
        self.assertEqual(by_id["attack-hidden"]["censor_at"], 3)

    def test_fixed_fpr_is_available_only_when_supported(self):
        calibration = []
        evaluation = []
        for index in range(20):
            calibration += events(f"cal-{index}", "benign", [0.1] * 3)
            evaluation += events(f"good-{index}", "benign", [0.1] * 3)
        evaluation += events("attack", "attacker", [0.2, 0.7, 0.7], "naive")
        report = active_detection_report(
            calibration,
            evaluation,
            target_fpr=0.2,
            budget=3,
            checkpoints=[1, 2, 3],
            direction="higher",
            monitor_mode="per-account",
        )
        self.assertTrue(report["fixed_fpr_status"]["available"])
        self.assertEqual(report["tpr_at_fixed_fpr"], 1.0)
        self.assertTrue(report["roc_auc_status"]["available"])
        self.assertEqual(report["roc_auc"], 1.0)

    def test_undetected_attacker_must_reach_budget(self):
        calibration = events("cal", "benign", [0.1, 0.1, 0.1])
        evaluation = (
            events("good", "benign", [0.1, 0.1, 0.1])
            + events("attack", "attacker", [0.1, 0.1], "evasive")
        )
        with self.assertRaisesRegex(ValueError, "must be observed through"):
            active_detection_report(
                calibration,
                evaluation,
                target_fpr=0.5,
                budget=3,
                checkpoints=[1, 2, 3],
                direction="higher",
                monitor_mode="per-account",
            )

    def test_calibration_and_evaluation_ids_must_be_disjoint(self):
        calibration = events("same", "benign", [0.1])
        evaluation = events("same", "benign", [0.1]) + events(
            "attack", "attacker", [0.9], "naive"
        )
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            active_detection_report(
                calibration,
                evaluation,
                target_fpr=0.5,
                budget=1,
                checkpoints=[1],
                direction="higher",
                monitor_mode="per-account",
            )

    def test_duplicate_checkpoints_are_rejected(self):
        calibration = events("cal", "benign", [0.1])
        evaluation = events("good", "benign", [0.1]) + events(
            "attack", "attacker", [0.9], "naive"
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            active_detection_report(
                calibration,
                evaluation,
                target_fpr=0.5,
                budget=1,
                checkpoints=[1, 1],
                direction="higher",
                monitor_mode="per-account",
            )

    def test_calibration_rejects_attackers(self):
        with self.assertRaises(ValueError):
            active_detection_report(
                events("attack", "attacker", [0.5], "naive"),
                events("good", "benign", [0.1])
                + events("attack-2", "attacker", [0.6], "naive"),
                target_fpr=0.1,
                budget=1,
                checkpoints=[1],
                direction="higher",
                monitor_mode="per-account",
            )

    def test_calibration_and_evaluation_detectors_must_match(self):
        calibration = events("cal", "benign", [0.1], detector="active-v1")
        evaluation = events(
            "good", "benign", [0.1], detector="active-v2"
        ) + events("attack", "attacker", [0.9], "naive", detector="active-v2")
        with self.assertRaisesRegex(ValueError, "detectors differ"):
            active_detection_report(
                calibration,
                evaluation,
                target_fpr=0.5,
                budget=1,
                checkpoints=[1],
                direction="higher",
                monitor_mode="per-account",
            )

    def test_event_monitor_mode_must_match_requested_mode(self):
        calibration = events(
            "cal", "benign", [0.1], monitor_mode="global-context"
        )
        evaluation = events(
            "good", "benign", [0.1], monitor_mode="global-context"
        ) + events(
            "attack", "attacker", [0.9], "naive",
            monitor_mode="global-context",
        )
        with self.assertRaisesRegex(ValueError, "requested mode"):
            active_detection_report(
                calibration,
                evaluation,
                target_fpr=0.5,
                budget=1,
                checkpoints=[1],
                direction="higher",
                monitor_mode="per-account",
            )


if __name__ == "__main__":
    unittest.main()

