import json
import tempfile
import unittest
from pathlib import Path

from evaluation.defense_eval.evaluate_counter_seqkd import (
    BASE, TEACHER, checkpoint_from_manifest, detector_stats, discover_from_manifest, discover_local, local_path,
    validate_teacher_manifest,
)
from evaluation.defense_eval.rebuild_counter_detector_baselines import shared_baselines_ready


class CounterLocalEvaluationTests(unittest.TestCase):
    def test_shared_baseline_readiness_requires_matching_protocol_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = {"shared_across_countermeasures": True, "seed": 42, "temperature": 0.7,
                        "max_queries": 1000, "max_new_tokens": 124}
            for counter in ("dipper", "translation"):
                for defense in ("adfp", "ginsew", "radioactivity"):
                    path = (root / "outputs/countermeasures/qedks_b1000" / counter / defense /
                            "comparison_report.json")
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({
                        "baseline_detection_protocol": protocol,
                        "reports": {"clean": {"score": 0}, "defense_only": {"score": 1}},
                    }))
            self.assertTrue(shared_baselines_ready(root, "qedks", 1000, 42, 0.7, 1000, 124))
            changed = (root / "outputs/countermeasures/qedks_b1000/translation/ginsew/comparison_report.json")
            payload = json.loads(changed.read_text())
            payload["reports"]["clean"]["score"] = 0.5
            changed.write_text(json.dumps(payload))
            self.assertFalse(shared_baselines_ready(root, "qedks", 1000, 42, 0.7, 1000, 124))

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

    def test_qedks_adapter_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "outputs/qedks/checkpoints/lora_sft"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": BASE}))
            (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
            manifest = root / "outputs/qedks/attack_manifest.json"
            manifest.write_text(json.dumps({
                "run_config": {"teacher_model": TEACHER, "student_model": BASE},
                "result": {"attack": "qedks", "budget": 1000, "status": "completed",
                           "checkpoint_dir": str(checkpoint)},
            }))
            self.assertEqual(checkpoint_from_manifest(manifest, root, "qedks"), checkpoint.resolve())

    def test_qedks_discovery_finds_shared_baselines_and_six_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def save(path, data):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
                return path

            def attack(path):
                checkpoint = path / "checkpoints/lora_sft"
                save(checkpoint / "adapter_config.json", {"base_model_name_or_path": BASE})
                (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
                manifest = save(path / "attack_manifest.json", {
                    "run_config": {"teacher_model": TEACHER, "student_model": BASE},
                    "result": {"attack": "qedks", "budget": 1000, "status": "completed",
                               "checkpoint_dir": str(checkpoint)},
                })
                return manifest, checkpoint

            baselines = {}
            for name in ("clean", "adfp", "ginsew", "radioactivity"):
                manifest, checkpoint = attack(root / "outputs/counter_baselines/qedks_b1000" / name / "run")
                baseline = save(manifest.parent.parent / "baseline_manifest.json", {
                    "status": "ok", "attack_manifest_path": str(manifest),
                    "student_checkpoint_path": str(checkpoint),
                })
                baselines[name] = str(baseline)
            for counter in ("dipper", "translation"):
                for defense in ("adfp", "ginsew", "radioactivity"):
                    directory = root / "outputs/countermeasures/qedks_b1000" / counter / defense
                    manifest, checkpoint = attack(directory / "attack/qedks/run")
                    save(directory / "comparison_report.json", {
                        "attack": "qedks", "budget": 1000, "defense": defense,
                        "baseline_manifests": {"clean": baselines["clean"],
                                               "defense_only": baselines[defense]},
                        "reports": {
                            "clean": {"score": 0.0},
                            "defense_only": {"score": 1.0},
                            "counter": {"student_checkpoint": str(checkpoint)},
                        },
                    })
            models, reports = discover_local(root, "qedks")
            self.assertEqual(len(models), 10)
            self.assertEqual(len(set(models.values())), 10)
            self.assertEqual(len(reports), 6)

    def test_manifest_discovery_follows_only_selected_adaptive_run(self):
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
                    "result": {"attack": "seqkd", "budget": 1000, "status": "completed",
                               "checkpoint_dir": str(checkpoint)},
                })
                return manifest

            clean = attack(root / "outputs/baselines/clean/run")
            defended = attack(root / "outputs/baselines/adfp/run")
            counter = attack(root / "outputs/adaptive/dipper/adfp/seqkd/b1000/run/attack/seqkd/counter")
            clean_baseline = save(root / "outputs/baselines/clean/baseline_manifest.json", {
                "status": "ok", "attack_manifest_path": str(clean),
            })
            defense_baseline = save(root / "outputs/baselines/adfp/baseline_manifest.json", {
                "status": "ok", "attack_manifest_path": str(defended),
            })
            report = save(root / "outputs/adaptive/dipper/adfp/seqkd/b1000/run/comparison_report.json", {
                "attack": "seqkd", "budget": 1000, "defense": "adfp",
                "baseline_manifests": {"clean": str(clean_baseline), "defense_only": str(defense_baseline)},
                "reports": {"counter": {"student_checkpoint": str(counter)}},
            })
            manifest = save(report.parent / "defense_run_manifest.json", {
                "defense": "adfp",
                "generator_result": {
                    "attack_manifest_path": str(counter),
                    "metadata": {"attack": {"attack": "seqkd", "budget": 1000},
                                 "countermeasure": "waterpark_dipper", "comparison": str(report)},
                },
            })

            attack_name, models, reports = discover_from_manifest(manifest, root)
            self.assertEqual(attack_name, "seqkd")
            self.assertEqual(set(models), {"clean", "adfp_defense_only", "adfp_dipper"})
            self.assertEqual(set(reports), {("dipper", "adfp")})

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
