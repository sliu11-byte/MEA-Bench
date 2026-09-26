import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from defenses.core.runner.offline import _run_soda_with_prerequisites


class SODADefenseAutomationTests(unittest.TestCase):
    def test_runner_builds_seqkd_and_soda_in_one_dependency_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attack_dir = root / "attack"
            checkpoint = root / "seqkd-checkpoint"
            checkpoint.mkdir()
            transcript = root / "defended_teacher.jsonl"
            transcript.write_text('{"query_id":"q1","query":"p","response":"r"}\n', encoding="utf-8")
            prepared = attack_dir / "soda" / "prepared_soda_b1000"

            calls = []

            def fake_run_attack_process(**kwargs):
                calls.append(kwargs)
                if kwargs["attack"] == "seqkd":
                    return {"manifest_path": "seqkd.json", "checkpoint_path": str(checkpoint)}
                if "prepare" in kwargs["extra_args"]:
                    prepared.mkdir(parents=True)
                    (prepared / "prepare_manifest.json").write_text("{}\n", encoding="utf-8")
                    return {"manifest_path": None, "checkpoint_path": None}
                return {"manifest_path": "soda.json", "checkpoint_path": "soda-checkpoint"}

            args = SimpleNamespace(
                budget=1000,
                teacher_model="teacher",
                student_model="student",
                query_pool="auto",
                query_ordering="auto",
                teacher_mode="chat",
                teacher_temperature=0.0,
                teacher_top_p=1.0,
                teacher_max_tokens=1536,
                stage1_config="stage.yaml",
                startup_timeout=600.0,
            )
            process = object()
            with patch(
                "defenses.core.runner.offline.run_attack_process",
                side_effect=fake_run_attack_process,
            ), patch(
                "defenses.core.runner.offline._start_soda_student",
                return_value=(process, "http://127.0.0.1:9999/v1", "soda-warmup"),
            ), patch("defenses.core.runner.offline._stop_process") as stop:
                result = _run_soda_with_prerequisites(
                    repo_root=root,
                    args=args,
                    attack_dir=attack_dir,
                    transcript_path=transcript,
                    attack_extra_args=[],
                    defense="adfp",
                )

            self.assertEqual([call["attack"] for call in calls], ["seqkd", "soda", "soda"])
            self.assertIn("prepare", calls[1]["extra_args"])
            self.assertIn("train", calls[2]["extra_args"])
            self.assertIn(str(prepared), calls[2]["extra_args"])
            self.assertEqual(result["prerequisites"]["seqkd"]["checkpoint_path"], str(checkpoint))
            stop.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
