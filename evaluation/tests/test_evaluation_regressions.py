import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluation.attack_eval.evaluate_attack_four_metrics import make_example_ids_unique
from evaluation.metrics.m2_fidelity import bertscore_f1
from evaluation.tasks.parsing.numeric import normalize_number


class EvaluationRegressionTests(unittest.TestCase):
    def test_huge_integer_does_not_raise_decimal_invalid_operation(self):
        value = "9" * 10000
        self.assertEqual(normalize_number(value), value)
        self.assertEqual(normalize_number("-000.000"), "0")

    def test_only_duplicate_dataset_ids_gain_row_suffix(self):
        rows = [SimpleNamespace(example_id="unique", example_index=0),
                SimpleNamespace(example_id="repeat", example_index=1),
                SimpleNamespace(example_id="repeat", example_index=27)]
        make_example_ids_unique(rows)
        self.assertEqual([row.example_id for row in rows],
                         ["unique", "repeat::row-1", "repeat::row-27"])

    def test_bertscore_handles_empty_candidates_without_sending_them_to_library(self):
        tensor = Mock()
        tensor.sum.return_value.item.return_value = 1.2
        score = Mock(return_value=(None, None, tensor))
        module = SimpleNamespace(score=score)
        with patch.dict(sys.modules, {"bert_score": module}):
            result = bertscore_f1(["", "valid", ""], ["reference", "target", ""],
                                  "en", "roberta-large", 16, "cuda:0")
        self.assertAlmostEqual(result, (0 + 1.2 + 1) / 3)
        self.assertEqual(score.call_args.kwargs["cands"], ["valid"])
        self.assertEqual(score.call_args.kwargs["refs"], ["target"])

    def test_bertscore_rejects_unpaired_or_empty_inputs(self):
        with self.assertRaises(ValueError):
            bertscore_f1([], [], "en", None, 16, None)
        with self.assertRaises(ValueError):
            bertscore_f1(["one"], [], "en", None, 16, None)


if __name__ == "__main__":
    unittest.main()
