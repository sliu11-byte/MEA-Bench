import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.core.batch_generation import batch_size, generate_batch, iter_generated


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def to(self, device):
        return self

    def __getitem__(self, key):
        rows, columns = key
        return [row[columns] for row in self.rows[rows]]


class BatchGenerationTests(unittest.TestCase):
    def test_padding_attention_and_output_slice(self):
        tokenizer = Mock()
        tokenizer.padding_side = "right"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 9
        tokenizer.apply_chat_template.side_effect = lambda messages, **kw: "CHAT:" + messages[0]["content"]
        tokenizer.return_value = {"input_ids": Tensor([[0, 1, 2], [3, 4, 5]]),
                                  "attention_mask": Tensor([[0, 1, 1], [1, 1, 1]])}
        tokenizer.batch_decode.side_effect = lambda rows, **kw: [str(row[0]) for row in rows]
        model = Mock(device="cuda:0")

        def generate(**kw):
            self.assertEqual(tokenizer.padding_side, "left")
            self.assertIn("attention_mask", kw)
            self.assertEqual(kw["pad_token_id"], 0)
            self.assertFalse(kw["do_sample"])
            return Tensor([[0, 1, 2, 7], [3, 4, 5, 8]])

        model.generate.side_effect = generate
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            self.assertEqual(generate_batch(model, tokenizer, ["short", "longer"], 32, True), ["7", "8"])
        tokenizer.assert_called_once_with(["CHAT:short", "CHAT:longer"], padding=True,
                                          add_special_tokens=False, return_tensors="pt")
        self.assertEqual(tokenizer.padding_side, "right")

    def test_raw_text_and_failure_restore_padding(self):
        tokenizer = Mock(padding_side="right", pad_token_id=0, eos_token_id=9)
        tokenizer.side_effect = RuntimeError("failed")
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                generate_batch(Mock(), tokenizer, ["raw"], 4, False)
        tokenizer.assert_called_once_with(["raw"], padding=True, return_tensors="pt")
        self.assertEqual(tokenizer.padding_side, "right")

    def test_item_association_partial_batch_and_completed_filter(self):
        items = [{"id": "done", "prompt": "skip"}, {"id": "long", "prompt": "longest"},
                 {"id": "short", "prompt": "x"}, {"id": "middle", "prompt": "mid"}]
        pending = [item for item in items if item["id"] != "done"]
        with patch("evaluation.core.batch_generation.generate_batch", side_effect=lambda m, t, p, n, c: p) as generate:
            rows = list(iter_generated(None, None, pending, lambda p: p["prompt"], 4, True, 2))
        self.assertEqual([item["id"] for item, _ in rows], ["short", "middle", "long"])
        self.assertTrue(all(item["prompt"] == text for item, text in rows))
        self.assertEqual([len(call.args[2]) for call in generate.call_args_list], [2, 1])
        with patch("evaluation.core.batch_generation.generate_batch") as generate:
            self.assertEqual(list(iter_generated(None, None, [], str, 4, True, 2)), [])
            generate.assert_not_called()

    def test_config(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(batch_size("mc"), 8)
            self.assertEqual(batch_size("heldout"), 4)
            self.assertEqual(batch_size("mc", teacher=True), 2)
            self.assertEqual(batch_size("heldout", teacher=True), 1)
        with patch.dict(os.environ, {"EVAL_BATCH_MC": "1"}):
            self.assertEqual(batch_size("mc"), 1)
        with patch.dict(os.environ, {"EVAL_BATCH_MC": "0"}):
            with self.assertRaises(ValueError):
                batch_size("mc")

    def test_shared_evaluator_resumes_and_saves_by_id(self):
        from evaluation.attack_eval.evaluate_attack_four_metrics import append, resume_rows
        from evaluation.defense_eval.evaluate import evaluate_model

        parser = SimpleNamespace(parse=lambda text: SimpleNamespace(status="success", normalized_prediction="A"))
        examples = [SimpleNamespace(example_id=str(i), rendered_prompt="Q" * (3 - i),
                                    normalized_gold_answer="A", parser=parser) for i in range(3)]
        prompts = [{"id": str(i), "prompt": "P" * (3 - i)} for i in range(3)]
        transformers = Mock()
        torch = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            append(directory / "m1_predictions/gsm8k.jsonl", {
                "id": "0", "response": "existing", "correct": True, "parse_status": "success"})
            append(directory / "heldout_outputs.jsonl", {**prompts[0], "text": "existing"})

            def generated(model, tokenizer, items, prompt_fn, tokens, chat, size):
                return iter((item, prompt_fn(item)) for item in reversed(items))

            with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}), \
                    patch("evaluation.defense_eval.evaluate.iter_generated", side_effect=generated) as generate:
                acc, rows = evaluate_model(directory, "local-student", True,
                                           {"gsm8k": examples}, prompts, True)
                self.assertEqual(acc, 1)
                self.assertEqual(rows["0"]["text"], "existing")
                self.assertEqual(rows["1"]["text"], prompts[1]["prompt"])
                self.assertEqual(len(resume_rows(directory / "m1_predictions/gsm8k.jsonl")), 3)
                self.assertTrue(all(len(call.args[2]) == 2 for call in generate.call_args_list))
            transformers.reset_mock()
            with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
                evaluate_model(directory, "local-student", True, {"gsm8k": examples}, prompts, True)
            transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_shared_evaluator_reuses_matching_gsm8k_rows_without_loading_model(self):
        from evaluation.attack_eval.evaluate_attack_four_metrics import append
        from evaluation.defense_eval.evaluate import evaluate_model

        parser = SimpleNamespace(parse=Mock())
        examples = [SimpleNamespace(example_id=str(i), rendered_prompt=f"Q{i}",
                                    normalized_gold_answer=str(i), parser=parser) for i in range(2)]
        transformers = Mock()
        torch = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "old"
            target = root / "new"
            for i in range(2):
                append(source / "m1_predictions/gsm8k.jsonl", {
                    "id": str(i), "response": str(i), "prediction": str(i), "gold": str(i),
                    "correct": True, "parse_status": "success",
                })
            with patch.dict("sys.modules", {"transformers": transformers, "torch": torch}):
                acc, rows = evaluate_model(target, "student", False, {"gsm8k": examples}, [], False,
                                           reuse_gsm8k_directory=source)
            self.assertEqual(acc, 1)
            self.assertEqual(rows, {})
            self.assertTrue((target / "m1_predictions/gsm8k.jsonl").is_file())
            transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
            parser.parse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
