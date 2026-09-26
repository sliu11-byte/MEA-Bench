import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from benchmark.cli import main
from benchmark.profiles import load_profile
from benchmark.registry import (
    ATTACKS,
    DEFENSES,
    RegistryError,
    validate_adaptive_combination,
)


class BenchmarkContractTests(unittest.TestCase):
    def test_paper_profile_and_registry_match_readme(self):
        profile = load_profile("paper")
        self.assertEqual(profile.attack_budgets, (100, 1000, 10000))
        self.assertEqual(profile.section("defense")["teacher_model"], "Qwen/Qwen2.5-72B-Instruct")
        self.assertEqual(profile.section("defense")["student_model"], "Qwen/Qwen2.5-7B")
        self.assertEqual(set(ATTACKS), {"seqkd", "lord", "soda", "qedks", "model_leeching", "gad"})
        self.assertEqual(
            {name for name, spec in DEFENSES.items() if spec.category == "defended_extraction"},
            {"ads", "doge", "trace_rewriting", "adfp", "ginsew", "radioactivity"},
        )
        self.assertEqual(
            {name for name, spec in DEFENSES.items() if spec.category == "result_based"},
            {"duffin", "mmd", "prada", "seat"},
        )

    def test_adaptive_matrix_rejects_unsupported_defense(self):
        validate_adaptive_combination("dipper", "adfp", "seqkd", 1000)
        with self.assertRaisesRegex(RegistryError, "does not support defense"):
            validate_adaptive_combination("translation", "ads", "seqkd", 1000)

    def test_attack_dry_run_writes_portable_planned_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            query_pool = root / "query_pool_100.json"
            query_pool.write_text(
                json.dumps(
                    {
                        "meta": {"tier": 100, "seed": 20260701},
                        "data": [
                            {"id": f"q{i}", "prompt": f"prompt {i}"}
                            for i in range(100)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch.dict(os.environ, {"STORAGE_ROOT": str(root / "storage")}), contextlib.redirect_stdout(output):
                result = main(
                    [
                        "attack",
                        "--attack",
                        "seqkd",
                        "--budget",
                        "100",
                        "--profile",
                        "paper",
                        "--query-pool",
                        str(query_pool),
                        "--output-root",
                        str(root / "outputs"),
                        "--dry-run",
                    ]
                )
            self.assertEqual(result, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["schema_version"], "mea_experiment_manifest_v1")
            self.assertEqual(payload["status"], "planned")
            self.assertEqual(payload["attack"], "seqkd")
            self.assertEqual(payload["inputs"]["query_pool"]["records"], 100)
            self.assertEqual(payload["config"]["teacher"]["max_tokens"], 1536)
            self.assertNotIn(str(root), json.dumps(payload))
            manifest = next((root / "outputs").glob("attacks/seqkd/b100/*/attack_manifest.json"))
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["config_hash"], payload["config_hash"])

            resumed_output = io.StringIO()
            with patch.dict(os.environ, {"STORAGE_ROOT": str(root / "storage")}), contextlib.redirect_stdout(resumed_output):
                result = main(
                    [
                        "attack",
                        "--attack",
                        "seqkd",
                        "--budget",
                        "100",
                        "--query-pool",
                        str(query_pool),
                        "--output-root",
                        str(root / "outputs"),
                        "--dry-run",
                        "--resume",
                    ]
                )
            self.assertEqual(result, 0)
            resumed = json.loads(resumed_output.getvalue())
            self.assertEqual(resumed["run_id"], payload["run_id"])
            self.assertTrue(resumed["resume"]["matched_existing_run"])

    def test_paper_profile_rejects_non_protocol_budget(self):
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
            main(["attack", "--attack", "seqkd", "--budget", "500", "--dry-run"])
        self.assertEqual(raised.exception.code, 2)

    def test_portable_shell_dispatchers_dry_run(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            query_pool = root / "query_pool_100.json"
            query_pool.write_text(
                json.dumps(
                    {
                        "meta": {"tier": 100, "seed": 20260701},
                        "data": [
                            {"id": f"q{i}", "prompt": f"prompt {i}"}
                            for i in range(100)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            attack_plan = subprocess.run(
                [
                    "bash",
                    "runs/run_attack.sh",
                    "--attack",
                    "seqkd",
                    "--budget",
                    "100",
                    "--query-pool",
                    str(query_pool),
                    "--output-root",
                    str(root / "outputs"),
                    "--dry-run",
                ],
                cwd=repo_root,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertEqual(json.loads(attack_plan.stdout)["attack"], "seqkd")

            defense_plan = subprocess.run(
                [
                    "bash",
                    "runs/run_defense.sh",
                    "--defense",
                    "adfp",
                    "--attack",
                    "seqkd",
                    "--budget",
                    "1000",
                    "--dry-run",
                ],
                cwd=repo_root,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertEqual(json.loads(defense_plan.stdout)["dispatcher"], "runs/defense/common.sh")

            adaptive_plan = subprocess.run(
                [
                    "bash",
                    "runs/run_adaptive_attack.sh",
                    "--adaptive-attack",
                    "translation",
                    "--defense",
                    "ginsew",
                    "--attack",
                    "qedks",
                    "--budget",
                    "1000",
                    "--dry-run",
                ],
                cwd=repo_root,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertEqual(json.loads(adaptive_plan.stdout)["dispatcher"], "runs/counter/common.sh")

            attack_manifest = root / "attack_manifest.json"
            checkpoint = root / "checkpoint-final"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "model.safetensors").write_bytes(b"fixture")
            attack_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "attack_manifest_v1",
                        "result": {
                            "attack": "seqkd",
                            "budget": 1000,
                            "run_id": "test-run",
                            "status": "completed",
                            "checkpoint_dir": str(checkpoint),
                        },
                    }
                ),
                encoding="utf-8",
            )
            detector_plan = subprocess.run(
                [
                    "bash",
                    "runs/run_defense.sh",
                    "--defense",
                    "mmd",
                    "--attack-run",
                    str(attack_manifest),
                    "--dry-run",
                ],
                cwd=repo_root,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertEqual(json.loads(detector_plan.stdout)["dispatcher"], "runs/defense/detector_common.sh")

            evaluation_plan = subprocess.run(
                ["bash", "runs/run_evaluation.sh", "--manifest", str(attack_manifest), "--dry-run"],
                cwd=repo_root,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                json.loads(evaluation_plan.stdout)["driver"],
                "runs/evaluation/evaluate_attack_b1000.sh",
            )


if __name__ == "__main__":
    unittest.main()
