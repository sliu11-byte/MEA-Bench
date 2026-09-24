import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import datasets  # noqa: F401
except ImportError:
    sys.modules["datasets"] = SimpleNamespace(
        Dataset=type("Dataset", (), {}), DatasetDict=type("DatasetDict", (), {}),
        load_dataset=Mock(), load_from_disk=Mock(),
    )
try:
    import huggingface_hub  # noqa: F401
except ImportError:
    sys.modules["huggingface_hub"] = SimpleNamespace(hf_hub_download=Mock())

from evaluation.core.choice_scoring import SCORING_METHOD, choice_spec, encode_pair, score_examples
from evaluation.core.config import DatasetSpec
from evaluation.tasks.adapters.tasks import (
    ArcChallengeAdapter, HellaSwagAdapter, MMLUAdapter, TruthfulQAAdapter, WinoGrandeAdapter,
)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens):
        ids = [ord(character) for character in text]
        return ([1] + ids) if add_special_tokens else ids

    def apply_chat_template(self, messages, **kwargs):
        return f"<chat>{messages[0]['content']}<assistant>"


def example(example_id, normalize):
    return SimpleNamespace(
        example_id=example_id,
        normalized_gold_answer="A",
        extra={"choice_scoring": {
            "method": SCORING_METHOD,
            "prompt": "Question:\nAnswer:",
            "labels": ["A", "B"],
            "continuations": [" alpha", " beta"],
            "length_normalize": normalize,
            "metric": "acc_norm" if normalize else "acc",
        }},
    )


class ChoiceScoringTests(unittest.TestCase):
    def adapter(self, adapter_class):
        return adapter_class(DatasetSpec(
            key="test", display_name="test", source="test", identifier="test",
            config_name=None, revision=None, local_path=None, adapter="test", task_type="multiple_choice",
            requested_splits=[], provenance_note="test",
        ))

    def test_encode_pair_preserves_trailing_space_as_continuation(self):
        tokenizer = CharacterTokenizer()
        full, boundary = encode_pair(tokenizer, "Answer: ", "A", True)
        self.assertEqual(full[:boundary], tokenizer.encode("Answer:", add_special_tokens=True))
        self.assertEqual(full[boundary:], tokenizer.encode(" A", add_special_tokens=False))

    def test_spec_rejects_missing_protocol(self):
        with self.assertRaisesRegex(ValueError, SCORING_METHOD):
            choice_spec(SimpleNamespace(example_id="bad", extra={}, normalized_gold_answer="A"))

    def test_normalized_and_raw_scores_can_select_different_choices(self):
        normalized = example("normalized", True)
        raw = example("raw", False)
        values = [
            {"loglikelihood": -4.0, "avg_loglikelihood": -1.0, "token_count": 4},
            {"loglikelihood": -3.0, "avg_loglikelihood": -3.0, "token_count": 1},
            {"loglikelihood": -4.0, "avg_loglikelihood": -1.0, "token_count": 4},
            {"loglikelihood": -3.0, "avg_loglikelihood": -3.0, "token_count": 1},
        ]
        with patch("evaluation.core.choice_scoring._score_encoded", return_value=values):
            rows = dict((item.example_id, row) for item, row in
                        score_examples(None, CharacterTokenizer(), [normalized, raw], False, 2))
        self.assertEqual(rows["normalized"]["prediction"], "A")
        self.assertEqual(rows["raw"]["prediction"], "B")
        self.assertEqual(rows["normalized"]["scoring_method"], SCORING_METHOD)

    def test_five_multiple_choice_adapters_define_v2_candidates(self):
        rows = [
            (ArcChallengeAdapter, {"question": "Q", "choices": {"label": ["A", "B"],
                                                                   "text": ["one", "two"]}, "answerKey": "A"}),
            (HellaSwagAdapter, {"ctx": "Context", "endings": ["one", "two"], "label": "0"}),
            (MMLUAdapter, {"question": "Q", "choices": ["one", "two"], "answer": 0}),
            (TruthfulQAAdapter, {"question": "Q", "mc1_targets": {"choices": ["one", "two"],
                                                                        "labels": [1, 0]}}),
            (WinoGrandeAdapter, {"sentence": "The _ won.", "option1": "first", "option2": "second",
                                  "answer": "1"}),
        ]
        for adapter_class, raw in rows:
            with self.subTest(adapter=adapter_class.__name__):
                adapted = self.adapter(adapter_class).adapt(raw, "test", 0)
                spec = choice_spec(adapted)
                self.assertEqual(spec["method"], SCORING_METHOD)
                self.assertEqual(len(spec["labels"]), len(spec["continuations"]))
                self.assertTrue(spec["prompt"][-1:].isspace() or
                                all(value[:1].isspace() for value in spec["continuations"]))


if __name__ == "__main__":
    unittest.main()
