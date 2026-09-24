import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics.m1_capability_reports import accuracy_report, domain6_report, iter_jsonl, openllm6_report


def openllm6_payload():
    return {
        "results": {
            "arc_challenge": {"acc_norm,none": 0.5},
            "hellaswag": {"acc_norm,none": 0.6},
            "mmlu": {"acc,none": 0.7},
            "truthfulqa_mc2": {"acc,none": 0.4},
            "winogrande": {"acc,none": 0.8},
            "gsm8k": {
                "exact_match,strict-match": 0.3,
                "exact_match,flexible-extract": 0.99,
            },
        },
        "n-shot": {
            "arc_challenge": 25,
            "hellaswag": 10,
            "mmlu": 5,
            "truthfulqa_mc2": 0,
            "winogrande": 5,
            "gsm8k": 5,
        },
    }


class M1MetricsTest(unittest.TestCase):
    def test_openllm6_reads_exact_keys_and_simple_average(self):
        report = openllm6_report(openllm6_payload())
        self.assertAlmostEqual(report["openllm6_average"], 0.55)
        self.assertEqual(report["scores"]["gsm8k"], 0.3)
        self.assertEqual(report["n_tasks"], 6)

    def test_openllm6_requires_gsm8k_strict_match(self):
        payload = openllm6_payload()
        del payload["results"]["gsm8k"]["exact_match,strict-match"]
        with self.assertRaisesRegex(ValueError, "strict-match"):
            openllm6_report(payload)

    def test_openllm6_rejects_missing_task(self):
        payload = openllm6_payload()
        del payload["results"]["mmlu"]
        with self.assertRaisesRegex(ValueError, "mmlu"):
            openllm6_report(payload)

    def test_openllm6_rejects_out_of_range_score(self):
        payload = openllm6_payload()
        payload["results"]["mmlu"]["acc,none"] = 72.0
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            openllm6_report(payload)

    def test_openllm6_rejects_wrong_fewshot_protocol(self):
        payload = openllm6_payload()
        payload["n-shot"]["arc_challenge"] = 0
        with self.assertRaisesRegex(ValueError, "25-shot"):
            openllm6_report(payload)

    def test_accuracy_supports_custom_fields_and_task_groups(self):
        rows = [
            {"suite": "math", "pred": "A", "gold": "A"},
            {"suite": "math", "pred": "B", "gold": "A"},
            {"suite": "science", "pred": 1, "gold": 1.0},
        ]
        report = accuracy_report(
            rows,
            prediction_field="pred",
            target_field="gold",
            task_field="suite",
        )
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertEqual(report["per_task"], [
            {"task": "math", "accuracy": 0.5, "correct": 1, "n": 2},
            {"task": "science", "accuracy": 1.0, "correct": 1, "n": 1},
        ])
        self.assertEqual(report["protocol"]["prediction_field"], "pred")

    def test_accuracy_rejects_missing_field(self):
        with self.assertRaisesRegex(ValueError, "target"):
            accuracy_report([{"task": "math", "prediction": "A"}])

    def test_accuracy_rejects_invalid_label(self):
        with self.assertRaisesRegex(ValueError, "finite JSON scalar"):
            accuracy_report([{"task": "math", "prediction": ["A"], "target": "A"}])

    def test_iter_jsonl_reports_invalid_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            path.write_text(json.dumps({"task": "math"}) + "\nnot-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"bad\.jsonl:2"):
                list(iter_jsonl([path]))

    def test_domain6_requires_exact_matrix_coverage_and_role(self):
        rows = [
            {"id": index, "task": task, "prediction": "A", "target": "A"}
            for index, task in enumerate(
                ("medqa", "pubmedqa", "chemprot", "fomc", "headline", "fpb")
            )
        ]
        report = domain6_report(
            rows,
            model_id="student-v1",
            model_role="student",
        )
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["model_role"], "student")
        self.assertEqual(report["domain_micro_accuracy"], 1.0)
        self.assertEqual(report["domain_macro_accuracy"], 1.0)
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            domain6_report(
                rows[:-1],
                model_id="student-v1",
                model_role="student",
            )

        duplicate_rows = [dict(row) for row in rows]
        duplicate_rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate Matrix domain6 sample"):
            domain6_report(
                duplicate_rows,
                model_id="student-v1",
                model_role="student",
            )


if __name__ == "__main__":
    unittest.main()

