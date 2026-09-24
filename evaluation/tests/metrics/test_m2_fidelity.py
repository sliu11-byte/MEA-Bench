import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics.m2_fidelity import (
    agreement_report,
    joint_completion_tokenization,
    prediction_from_sample,
    rouge_l_f1,
    symmetric_cross_ppl,
    text_pairs,
)


def sample(doc_id, scores, continuations):
    return {
        "doc_id": doc_id,
        "filter": "none",
        "filtered_resps": [[score, False] for score in scores],
        "arguments": [["prompt", continuation] for continuation in continuations],
    }


class M2MetricsTest(unittest.TestCase):
    def test_cross_ppl_requires_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.jsonl"
            path.write_text(
                json.dumps({"id": "1", "teacher_text": "a", "student_text": "b"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "prompt"):
                text_pairs([str(path)], require_prompt=True)

    def test_cross_ppl_rejects_token_boundary_split(self):
        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1], "offset_mapping": [(0, len(text))]}

        with self.assertRaisesRegex(ValueError, "splits a tokenizer token"):
            joint_completion_tokenization(FakeTokenizer(), "ab", "c", "1")

    def test_symmetric_cross_ppl_requires_identical_tokenizer(self):
        left = {"perplexity": 4.0, "tokenizer_fingerprint": "a"}
        right = {"perplexity": 9.0, "tokenizer_fingerprint": "b"}
        value, status = symmetric_cross_ppl(left, right)
        self.assertIsNone(value)
        self.assertFalse(status["available"])
        right["tokenizer_fingerprint"] = "a"
        value, status = symmetric_cross_ppl(left, right)
        self.assertEqual(value, 6.0)
        self.assertTrue(status["available"])

    def test_rouge_l_identical(self):
        self.assertEqual(rouge_l_f1("the cat sat", "the cat sat"), 1.0)

    def test_rouge_l_partial(self):
        self.assertAlmostEqual(rouge_l_f1("a c", "a b c"), 0.8)

    def test_arc_uses_byte_length_normalization(self):
        row = sample(1, [-1.0, -1.5], ["A", "a much longer answer"])
        self.assertEqual(prediction_from_sample(row, "arc_challenge"), 1)
        self.assertEqual(prediction_from_sample(row, "mmlu_test"), 0)

    def test_agreement_reads_lm_eval_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher"
            student = root / "student"
            teacher.mkdir()
            student.mkdir()
            filename = "samples_mmlu_test_2026-07-09T10-00-00.jsonl"

            teacher_rows = [
                sample(1, [-1.0, -2.0], [" A", " B"]),
                sample(2, [-2.0, -1.0], [" A", " B"]),
            ]
            student_rows = [
                sample(1, [-1.0, -2.0], [" A", " B"]),
                sample(2, [-1.0, -2.0], [" A", " B"]),
            ]
            for directory, rows in ((teacher, teacher_rows), (student, student_rows)):
                with (directory / filename).open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

            report = agreement_report(str(teacher), str(student))
            self.assertEqual(report["n"], 2)
            self.assertEqual(report["matched"], 1)
            self.assertEqual(report["agreement_rate"], 0.5)

    def test_agreement_requires_exact_sample_coverage_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher"
            student = root / "student"
            teacher.mkdir()
            student.mkdir()
            filename = "samples_mmlu_test_2026-07-09T10-00-00.jsonl"
            with (teacher / filename).open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(sample(1, [-1.0, -2.0], [" A", " B"])) + "\n")
                handle.write(json.dumps(sample(2, [-1.0, -2.0], [" A", " B"])) + "\n")
            with (student / filename).open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(sample(1, [-1.0, -2.0], [" A", " B"])) + "\n")

            with self.assertRaisesRegex(ValueError, "sample sets differ"):
                agreement_report(str(teacher), str(student))
            diagnostic = agreement_report(
                str(teacher), str(student), allow_partial=True
            )
            self.assertEqual(diagnostic["n"], 1)
            self.assertEqual(diagnostic["teacher_only"], 1)

    def test_agreement_rejects_same_doc_id_with_different_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher"
            student = root / "student"
            teacher.mkdir()
            student.mkdir()
            filename = "samples_mmlu_test_2026-07-09T10-00-00.jsonl"
            teacher_row = sample(1, [-1.0, -2.0], [" A", " B"])
            student_row = sample(1, [-1.0, -2.0], [" A", " B"])
            student_row["arguments"][0][0] = "different prompt"
            (teacher / filename).write_text(json.dumps(teacher_row) + "\n", encoding="utf-8")
            (student / filename).write_text(json.dumps(student_row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sample sets differ"):
                agreement_report(str(teacher), str(student))


if __name__ == "__main__":
    unittest.main()

