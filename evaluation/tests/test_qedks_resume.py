import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attacks.core.base import AttackResult, AttackRunConfig
from attacks.core.pipeline import AttackPipeline
from attacks.methods.qedks_impl.collect_teacher import main as collect_main


class FakeQEDKS:
    name = "qedks"

    def __init__(self):
        self.run_dir = None

    def run(self, config, *, run_id, run_dir):
        self.run_dir = run_dir
        return AttackResult("qedks", config.budget, run_id, run_dir, run_dir / "checkpoint", "completed")


class QEDKSResumeTests(unittest.TestCase):
    def config(self, root):
        pool = root / "pool.jsonl"
        stage = root / "stage.yaml"
        pool.write_text('{"query_id":"q0","query":"hello"}\n', encoding="utf-8")
        stage.write_text("{}\n", encoding="utf-8")
        return AttackRunConfig(
            attack="qedks", budget=1, query_pool_path=pool, output_dir=root / "out",
            stage1_config_path=stage, teacher_model="teacher", teacher_request_model="served",
            teacher_endpoint_url="http://localhost/v1", student_model="student",
        )

    def test_pipeline_reuses_latest_incomplete_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "out" / "qedks" / "20260919T000000Z_qedks_b1"
            old.mkdir(parents=True)
            attacker = FakeQEDKS()
            result = AttackPipeline({"qedks": attacker}).run(self.config(root))
            self.assertEqual(attacker.run_dir, old)
            self.assertEqual(result.output_dir, old)
            self.assertTrue((old / "qedks_resume_config.json").exists())

    def test_collection_skips_valid_saved_response(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / "plan.jsonl"
            output = root / "output.jsonl"
            plan.write_text('{"query_id":"q0","query":"hello"}\n', encoding="utf-8")
            output.write_text(json.dumps({
                "query_id": "q0", "query": "hello", "teacher_response": "answer",
                "request_model_name": "served", "request_mode": "chat",
                "generation_config": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": 42},
            }) + "\n", encoding="utf-8")
            argv = ["collect_teacher", "--query-plan-jsonl", str(plan), "--output-jsonl", str(output),
                    "--base-url", "http://localhost/v1", "--request-model-name", "served"]
            with patch("sys.argv", argv), patch(
                    "attacks.methods.qedks_impl.collect_teacher.generate_text") as generate:
                self.assertEqual(collect_main(), 0)
            generate.assert_not_called()
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
