import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from evaluation.scripts.resolve_local_run import resolve


class LocalRunResolverTests(unittest.TestCase):
    def save(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def adapter(self, path):
        self.save(path / "adapter_config.json", {"base_model_name_or_path": "Qwen/Qwen2.5-7B"})
        (path / "adapter_model.safetensors").write_bytes(b"fixture")
        return path

    def evaluation_plan(self, manifest):
        repo = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["bash", "runs/run_evaluation.sh", "--manifest", str(manifest), "--dry-run"],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )
        return json.loads(completed.stdout)

    def test_attack_uses_manifest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self.adapter(root / "outputs/attacks/qedks/b100/run/checkpoints/lora_sft")
            manifest = self.save(checkpoint.parents[1] / "attack_manifest.json", {
                "schema_version": "attack_manifest_v1",
                "result": {"attack": "qedks", "budget": 100, "run_id": "run", "checkpoint_dir": str(checkpoint)},
            })
            resolved = resolve(manifest, root)
            self.assertEqual(resolved["kind"], "attack")
            self.assertEqual(Path(resolved["checkpoint"]), checkpoint.resolve())

    def test_old_absolute_output_path_is_remapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self.adapter(root / "outputs/defenses/adfp/seqkd/b1000/run/generator/checkpoint-final")
            manifest = self.save(root / "outputs/defenses/adfp/seqkd/b1000/run/defense_run_manifest.json", {
                "defense": "adfp",
                "generator_result": {
                    "student_checkpoint_path": "/old/host/outputs/defenses/adfp/seqkd/b1000/run/generator/checkpoint-final",
                    "metadata": {"attack": {"attack": "seqkd", "budget": 1000}},
                },
            })
            resolved = resolve(manifest, root)
            self.assertEqual(resolved["kind"], "defense")
            self.assertEqual(Path(resolved["checkpoint"]), checkpoint.resolve())

    def test_detector_identity_is_inferred_from_canonical_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "outputs/defenses/mmd/lord/b10000"
            report = self.save(directory / "detector_report.json", {"score": 0.5})
            manifest = self.save(directory / "detector_manifest.json", {
                "detector": "mmd",
                "attack_run_id": "run-1",
                "output_report": "/retired/machine/outputs/defenses/mmd/lord/b10000/detector_report.json",
            })
            resolved = resolve(manifest, root)
            self.assertEqual((resolved["kind"], resolved["defense"]), ("detector", "mmd"))
            self.assertEqual((resolved["attack"], resolved["budget"]), ("lord", 10000))
            self.assertEqual(resolved["run_id"], "mmd_lord_b10000_run-1")
            self.assertEqual(Path(resolved["report"]), report.resolve())

    def test_top_level_evaluator_dispatches_all_manifest_kinds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attack_checkpoint = self.adapter(root / "outputs/attacks/qedks/b100/run/checkpoint-final")
            attack = self.save(attack_checkpoint.parent / "attack_manifest.json", {
                "schema_version": "attack_manifest_v1",
                "result": {"attack": "qedks", "budget": 100, "run_id": "attack-run",
                           "checkpoint_dir": str(attack_checkpoint)},
            })
            self.assertEqual(self.evaluation_plan(attack)["driver"], "runs/evaluation/evaluate_attack_b100.sh")

            defense_checkpoint = self.adapter(root / "outputs/defenses/adfp/seqkd/b1000/run/checkpoint-final")
            defense = self.save(defense_checkpoint.parent / "defense_run_manifest.json", {
                "defense": "adfp",
                "generator_result": {
                    "student_checkpoint_path": str(defense_checkpoint),
                    "metadata": {"attack": {"attack": "seqkd", "budget": 1000}},
                },
            })
            self.assertEqual(self.evaluation_plan(defense)["driver"], "evaluation.defense_eval.evaluate")

            adaptive_dir = root / "outputs/adaptive/dipper/adfp/seqkd/b1000/run"
            comparison = self.save(adaptive_dir / "comparison_report.json", {
                "attack": "seqkd", "budget": 1000, "defense": "adfp",
            })
            adaptive = self.save(adaptive_dir / "defense_run_manifest.json", {
                "defense": "adfp",
                "generator_result": {"metadata": {
                    "attack": {"attack": "seqkd", "budget": 1000},
                    "countermeasure": "waterpark_dipper", "comparison": str(comparison),
                }},
            })
            self.assertEqual(
                self.evaluation_plan(adaptive)["driver"],
                "evaluation.defense_eval.evaluate_counter_seqkd",
            )

            detector_dir = root / "outputs/defenses/mmd/lord/b10000"
            report = self.save(detector_dir / "detector_report.json", {"score": 0.1})
            detector = self.save(detector_dir / "detector_manifest.json", {
                "detector": "mmd", "output_report": str(report),
            })
            self.assertEqual(self.evaluation_plan(detector)["driver"], "local_detector_report")


if __name__ == "__main__":
    unittest.main()
