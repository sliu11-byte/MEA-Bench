import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from defenses.core.io_utils import write_jsonl
from defenses.duffin.reference import get_teacher_reference
from evaluation.defense_eval.evaluate_duffin import summarize


class DuffinEvaluationTests(unittest.TestCase):
    def test_reference_generated_once_and_then_reused(self):
        probes = [{"probe_id": "one", "question": "Q", "options": ["x", "y"]}]
        torch = Mock()
        torch.cuda.is_available.return_value = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.json"
            with patch.dict("sys.modules", {"torch": torch}), \
                    patch("defenses.duffin.reference.load_local_model", return_value=(Mock(), Mock())) as loader, \
                    patch("defenses.duffin.reference.query_local_model_batch", return_value=[("A", "A")]) as query:
                first, count = get_teacher_reference(path, probes, "teacher", 4)
                self.assertEqual(count, 1)
                second, count = get_teacher_reference(path, probes, "teacher", 4)
                self.assertEqual(count, 0)
                self.assertEqual(first, second)
                loader.assert_called_once()
                query.assert_called_once()

    def test_cached_reference_does_not_load_teacher(self):
        probes = [{"probe_id": "one", "question": "Q", "options": ["x", "y"]}]
        artifact = {"protocol": {"version": 1, "teacher_model": "teacher", "max_new_tokens": 4,
                                 "batch_size": 4, "do_sample": False,
                                 "prompt_format": "duffin_official_cot_v2", "probes": probes},
                    "rows": [{"teacher_response": "A", "teacher_choice": "A"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.json"
            path.write_text(json.dumps(artifact))
            with patch("defenses.duffin.reference.load_local_model") as loader:
                rows, queries = get_teacher_reference(path, probes, "teacher", 4)
                self.assertEqual(queries, 0)
                self.assertEqual(rows[0]["teacher_choice"], "A")
                loader.assert_not_called()
            with self.assertRaisesRegex(ValueError, "mismatch"):
                get_teacher_reference(path, probes, "different", 4)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                get_teacher_reference(path, list(reversed(probes)) + probes, "teacher", 4)

    def test_summary_uses_common_valid_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("base_negative", "seqkd", "soda", "qedks"):
                directory = root / name
                directory.mkdir()
                rows = [{"probe_id": "one", "teacher_choice": "A", "student_choice": "B" if name == "base_negative" else "A"},
                        {"probe_id": "two", "teacher_choice": "B", "student_choice": None if name == "qedks" else "B"}]
                write_jsonl(rows, directory / "probe_rows.jsonl")
            results = summarize(root, ("base_negative",))
            self.assertEqual(len(results), 4)
            self.assertEqual(results[0]["score"], 0)
            self.assertEqual(results[1]["score"], 1)
            self.assertEqual(results[1]["difference_from_reference_negative"], 1)
            self.assertTrue(all(row["common_valid_probes"] == 1 for row in results))
            self.assertTrue((root / "summary.csv").exists())
            self.assertTrue((root / "roc_auc.json").exists())
            rows.reverse()
            write_jsonl(rows, root / "qedks" / "probe_rows.jsonl")
            with self.assertRaisesRegex(ValueError, "order mismatch"):
                summarize(root, ("base_negative",))


if __name__ == "__main__":
    unittest.main()
