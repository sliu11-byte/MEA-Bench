import json
import tempfile
import unittest
from pathlib import Path

from evaluation.defense_eval.evaluate_counter_seqkd import (
    BASE, TEACHER, checkpoint_from_manifest, detector_stats, discover_local, local_path,
    validate_teacher_manifest,
)


class CounterLocalEvaluationTests(unittest.TestCase):
    def test_teacher_manifest_accepts_existing_greedy_seed(self):
        manifest = {"teacher_model": TEACHER, "request_model": TEACHER, "mode": "chat",
                    "temperature": 0.0, "top_p": 1.0, "max_tokens": 1536,
                    "seed": 0, "prompt_count": 3000, "completed_count": 3000}
        self.assertEqual(validate_teacher_manifest(manifest)["seed"], 0)
        manifest["seed"] = 2026
        self.assertEqual(validate_teacher_manifest(manifest)["seed"], 2026)
        manifest["completed_count"] = 2999
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_teacher_manifest(manifest)
        manifest["completed_count"] = 3000
        manifest["temperature"] = 0.7
        with self.assertRaisesRegex(ValueError, "temperature"):
            validate_teacher_manifest(manifest)

    def test_path_remapping_and_detector_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "outputs/example.json"
            target.parent.mkdir()
            target.write_text("{}")
            self.assertEqual(local_path("/old/storage/outputs/example.json", root), target)
        self.assertEqual(detector_stats({"gtp": .49, "score": 0, "students": [{"score": 0}]})["gtp"], .49)
        self.assertEqual(detector_stats({"students": [{"green_rate": .25, "p_value": .5}]})["p_value"], .5)

    def test_discovery_reuses_four_baselines_and_finds_six_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def save(path, data):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
                return path

            def attack(path):
                checkpoint = path / "checkpoint-final"
                save(checkpoint / "config.json", {"model_type": "llama"})
                save(checkpoint / "tokenizer_config.json", {})
                (checkpoint / "model.safetensors").write_bytes(b"fixture")
                manifest = save(path / "attack_manifest.json", {
                    "run_config": {"teacher_model": TEACHER, "student_model": BASE},
                    "result": {"attack": "seqkd", "budget": 1000, "status": "completed", "checkpoint_dir": str(checkpoint)},
                })
                return manifest, checkpoint

            baselines = {}
            for name in ("clean", "adfp", "ginsew", "radioactivity"):
                manifest, checkpoint = attack(root / "outputs/counter_baselines/seqkd_b1000" / name / "run")
                baseline = save(manifest.parent.parent / "baseline_manifest.json", {
                    "status": "ok", "attack_manifest_path": str(manifest), "student_checkpoint_path": str(checkpoint),
                })
                baselines[name] = str(baseline)
            for counter in ("dipper", "translation"):
                for defense in ("adfp", "ginsew", "radioactivity"):
                    directory = root / "outputs/countermeasures/seqkd_b1000" / counter / defense
                    manifest, checkpoint = attack(directory / "attack/seqkd/run")
                    report = {"student_checkpoint": str(checkpoint)} if defense == "adfp" else {
                        "students": [{"student_checkpoint": str(manifest)}]}
                    save(directory / "comparison_report.json", {
                        "attack": "seqkd", "budget": 1000, "defense": defense,
                        "baseline_manifests": {"clean": baselines["clean"], "defense_only": baselines[defense]},
                        "reports": {"counter": report},
                    })
            models, reports = discover_local(root)
            self.assertEqual(len(models), 10)
            self.assertEqual(len(reports), 6)
            model_manifest = models["clean"].parent / "attack_manifest.json"
            data = json.loads(model_manifest.read_text())
            data["run_config"]["teacher_model"] = "Qwen/Qwen2.5-72B-Instruct"
            save(model_manifest, data)
            with self.assertRaisesRegex(ValueError, "Wrong teacher"):
                checkpoint_from_manifest(model_manifest, root)


if __name__ == "__main__":
    unittest.main()
