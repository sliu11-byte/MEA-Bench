import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.attack_eval.evaluate_attack_four_metrics import (
    allocation_cost, parse_job_ids, reuse_gsm8k, write_protocol_before_generation,
)
from evaluation.attack_eval.evaluate_attack_outputs import localize_uploaded_path
from evaluation.defense_eval.evaluate_counter_seqkd import reuse_valid_generations


class AttackM1V2Tests(unittest.TestCase):
    def test_protocol_path_can_change_only_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            protocol = directory / "protocol.json"
            protocol.write_text(json.dumps({"checkpoint": "/other-user/run"}), encoding="utf-8")
            replacement = {"checkpoint": "/snapshot/run"}
            write_protocol_before_generation(protocol, replacement, directory)
            self.assertEqual(json.loads(protocol.read_text(encoding="utf-8")), replacement)
            predictions = directory / "m1_choice_scores_v2" / "mmlu.jsonl"
            predictions.parent.mkdir()
            predictions.write_text(json.dumps({"id": "one"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Protocol changed"):
                write_protocol_before_generation(protocol, {"checkpoint": "/third/run"}, directory)

    def test_uploaded_run_path_wins_over_visible_submitter_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = "20260921T080524Z_soda_b1000"
            submitter = root / "other-user" / run_id / "checkpoints" / "dpo"
            snapshot_run = root / "snapshot" / run_id
            localized = snapshot_run / "checkpoints" / "dpo"
            submitter.mkdir(parents=True)
            localized.mkdir(parents=True)
            self.assertEqual(localize_uploaded_path(str(submitter), snapshot_run, run_id), localized.resolve())

    def test_parse_job_ids_accepts_spaces_and_commas(self):
        self.assertEqual(parse_job_ids(["101,102", " 103  104 "]), ["101", "102", "103", "104"])

    def test_reuse_gsm8k_copies_matching_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "old" / "seqkd" / "run-1" / "m1_predictions" / "gsm8k.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({"id": "one", "correct": True}) + "\n", encoding="utf-8")
            destination = root / "new" / "gsm8k.jsonl"
            used = reuse_gsm8k(root / "old", destination,
                                SimpleNamespace(attack="seqkd", run_id="run-1"))
            self.assertEqual(used, source)
            self.assertEqual(destination.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))

    def test_counter_reuse_excludes_old_multiple_choice_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "old"
            destination = root / "new"
            for relative in ("m1_predictions/gsm8k.jsonl", "heldout_outputs.jsonl",
                             "m1_predictions/mmlu.jsonl"):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"id": relative}) + "\n", encoding="utf-8")
            copied = reuse_valid_generations(source, destination)
            self.assertEqual(copied, ["m1_predictions/gsm8k.jsonl", "heldout_outputs.jsonl"])
            self.assertTrue((destination / "m1_predictions/gsm8k.jsonl").exists())
            self.assertTrue((destination / "heldout_outputs.jsonl").exists())
            self.assertFalse((destination / "m1_predictions/mmlu.jsonl").exists())

    @patch("evaluation.attack_eval.evaluate_attack_four_metrics.subprocess.run")
    def test_allocation_cost_sums_allocated_gpu_hours(self, run):
        run.side_effect = [
            SimpleNamespace(stdout="101|endpoint|COMPLETED|120|cpu=8,gres/gpu:b200=1,gres/gpu=1\n"),
            SimpleNamespace(stdout="102|train|COMPLETED|360|cpu=8,gres/gpu:b200=2,gres/gpu=2\n"),
        ]
        report = allocation_cost(["101", "102"], "run-1")
        self.assertAlmostEqual(report["allocated_gpu_hours"], (120 + 720) / 3600)
        self.assertEqual(report["jobs"][1]["gpu_types"], ["b200"])


if __name__ == "__main__":
    unittest.main()
