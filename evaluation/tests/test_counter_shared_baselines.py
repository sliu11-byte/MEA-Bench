import argparse
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from defenses.core.detector.result import DetectorResult
from defenses.core.generator.result import GeneratorResult
from defenses.core.runner import online
from defenses.core.runner.result import DefenseRunResult
from countermeasures.waterpark_response import CountermeasureConfig, rewrite_transcript


class SharedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(
            attack="seqkd", budget=1000, teacher_model="teacher", student_model="student",
            query_pool="auto", query_ordering="auto", stage1_config=None,
            teacher_mode="chat", teacher_temperature=0.0, teacher_top_p=1.0,
            teacher_max_tokens=1536, device="cuda", proxy_model="student",
            no_detector=False, defense_config="{}", countermeasure="dipper",
        )
        self.reference = GeneratorResult(
            "ginsew", "test", self.root / "counter", self.root / "oracle", "url",
            self.root / "oracle.json", self.root / "queries", self.root / "transcript",
            self.root / "artifacts", self.root / "attack", self.root / "manifest",
            self.root / "checkpoint", "ok",
        )
        self.training = []
        self.commands = []

    def train(self, *, defense, args, attack_extra_args):
        self.training.append(defense)
        self.assertEqual(args.countermeasure, "none")
        self.assertFalse(args.counter_baselines)
        if defense == "clean":
            self.assertIsNone(args.teacher_base_url)
        out = Path(args.output_dir)
        checkpoint = out / "checkpoint"
        checkpoint.mkdir(exist_ok=True)
        manifest = out / "attack.json"
        manifest.write_text("{}")
        (out / "artifacts").mkdir(exist_ok=True)
        (out / "queries").write_text("{}\n")
        (out / "transcript").write_text('{"query_id":"q1","query":"prompt","response":"defended"}\n')
        result = replace(self.reference, defense=defense, output_dir=out,
                         oracle_dir=out / "oracle", defense_artifacts_dir=out / "artifacts",
                         teacher_query_log_path=out / "queries",
                         defended_transcript_path=out / "transcript",
                         attack_manifest_path=manifest, student_checkpoint_path=checkpoint)
        return DefenseRunResult(defense, "baseline", out, result)

    def factory(self, args, result):
        return ["python", "--label", "positive", "--artifacts", str(result.defense_artifacts_dir),
                "--probes", str(result.teacher_query_log_path), "--output", str(result.output_dir)]

    def detect(self, **kwargs):
        self.commands.append(kwargs["command"])
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        report = out / "detector_report.json"
        report.write_text('{"students": []}')
        return DetectorResult("test", "ok", out, report)

    def run_group(self, defense, counter="dipper", extra=None):
        self.args.countermeasure = counter
        out = self.root / counter / defense
        out.mkdir(parents=True, exist_ok=True)
        return online.run_counter_baselines(
            defense=defense, args=self.args, attack_extra_args=extra or [],
            reference=replace(self.reference, output_dir=out), counter_detector=None,
            factory=self.factory,
        )

    def test_six_combinations_train_four_shared_baselines(self):
        with patch.dict("os.environ", {"COUNTER_BASELINE_ROOT": str(self.root / "shared")}), \
             patch.object(online, "train_shared_baseline", side_effect=self.train), \
             patch.object(online, "run_detector_command", side_effect=self.detect):
            for defense in ("ginsew", "radioactivity", "adfp"):
                dipper, detector = self.run_group(defense, "dipper")
                dipper_payload = json.loads(dipper.read_text())
                self.assertEqual(set(dipper_payload["reports"]), {"clean", "defense_only", "counter"})
                self.assertIsNotNone(detector)
                self.assertEqual([cmd[2] for cmd in self.commands[-3:]],
                                 ["negative", "positive", "positive"])
                before = len(self.commands)
                translation, detector = self.run_group(defense, "translation")
                translation_payload = json.loads(translation.read_text())
                self.assertEqual(len(self.commands), before + 1)
                self.assertEqual(dipper_payload["reports"]["clean"], translation_payload["reports"]["clean"])
                self.assertEqual(dipper_payload["reports"]["defense_only"],
                                 translation_payload["reports"]["defense_only"])
                self.assertEqual(dipper_payload["report_paths"]["clean"],
                                 translation_payload["report_paths"]["clean"])
                self.assertEqual(dipper_payload["report_paths"]["defense_only"],
                                 translation_payload["report_paths"]["defense_only"])
            self.assertEqual(self.training, ["clean", "ginsew", "radioactivity", "adfp"])
            # A restart also reuses completed baselines.
            self.run_group("ginsew")
            self.assertEqual(len(self.training), 4)
            self.args.budget = 100
            self.run_group("ginsew")
            self.assertEqual(self.training[-2:], ["clean", "ginsew"])
            self.assertEqual(len(self.training), 6)

    def test_failed_training_does_not_publish_cache(self):
        with patch.dict("os.environ", {"COUNTER_BASELINE_ROOT": str(self.root / "shared")}), \
             patch.object(online, "train_shared_baseline", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                self.run_group("ginsew")
        self.assertEqual(list((self.root / "shared").rglob("baseline_manifest.json")), [])

    def test_offline_counter_rewrites_shared_source_without_oracle(self):
        self.args.counter_baselines = True
        self.args.dry_run = False
        self.args.run_id = "counter"
        self.args.countermeasure_lex = 40
        self.args.countermeasure_order = 0
        self.args.countermeasure_sent_interval = 3
        self.args.countermeasure_with_context = False
        self.args.countermeasure_model_name = None
        self.args.countermeasure_tokenizer_name = None
        self.args.countermeasure_device = None
        baseline_dir = self.root / "source"
        baseline_dir.mkdir()
        baseline_args = argparse.Namespace(**vars(self.args))
        baseline_args.countermeasure = "none"
        baseline_args.counter_baselines = False
        baseline_args.output_dir = str(baseline_dir)
        defended = self.train(defense="ginsew", args=baseline_args, attack_extra_args=[]).generator_result
        sources = []
        completed = []

        def rewrite(command, **kwargs):
            def value(flag):
                return command[command.index(flag) + 1]
            sources.append(value("--input"))
            cfg = CountermeasureConfig(method=value("--method"), input_path=value("--input"),
                                       output_path=value("--output"), manifest_path=value("--manifest"))
            rewrite_transcript(config=cfg, rewriter=lambda text, **kw: "rewritten " + text)
            completed.append(cfg.output_path)
            return argparse.Namespace(returncode=0)

        def attack(**kwargs):
            transcript = kwargs["teacher_transcript_path"]
            self.assertIn(str(transcript), completed)
            self.assertIsNone(kwargs["teacher_base_url"])
            row = json.loads(transcript.read_text())
            self.assertEqual(row["query_id"], "q1")
            self.assertEqual(row["source_response"], "defended")
            self.assertEqual(row["response"], "rewritten defended")
            return {"manifest_path": str(defended.attack_manifest_path),
                    "checkpoint_path": str(defended.student_checkpoint_path)}

        with patch.object(online, "ensure_shared_baselines", return_value={"clean": defended, "defense_only": defended}), \
             patch.object(online, "start_oracle", side_effect=AssertionError("must not start teacher")), \
             patch.object(online, "run_logged_command", side_effect=rewrite), \
             patch.object(online, "run_attack_process", side_effect=attack), \
             patch.object(online, "run_counter_baselines", return_value=(self.root / "comparison.json", None)):
            for attack_name in ("seqkd", "lord", "soda", "gad"):
                self.args.attack = attack_name
                for counter in ("dipper", "translation"):
                    self.args.countermeasure = counter
                    self.args.output_dir = str(self.root / attack_name / counter)
                    result = online.run_online_defense(defense="ginsew", args=self.args, attack_extra_args=[])
                    self.assertEqual(result.metadata["execution_mode"], "shared_transcript_counter")
        self.assertEqual(len(sources), 8)
        self.assertEqual(set(sources), {str(defended.defended_transcript_path)})

    def test_online_attack_does_not_use_offline_counter(self):
        self.args.attack = "qedks"
        self.args.counter_baselines = True
        self.args.dry_run = False
        self.args.run_id = "online"
        self.args.output_dir = str(self.root / "online")
        self.args.port = 0
        self.args.host = "127.0.0.1"
        self.args.served_model_name = None
        for key in ("teacher_base_url", "teacher_request_model", "grad_path", "rewriter_model",
                    "rewriter_backend", "rewriter_base_url", "rewriter_request_model",
                    "rewriter_api_key", "doge_checkpoint", "teacher_api_key", "startup_timeout"):
            setattr(self.args, key, None)
        with patch.object(online, "start_oracle", side_effect=RuntimeError("online path reached")), \
             patch.object(online, "run_shared_transcript_counter", side_effect=AssertionError("wrong path")):
            with self.assertRaisesRegex(RuntimeError, "online path reached"):
                online.run_online_defense(defense="ginsew", args=self.args)


if __name__ == "__main__":
    unittest.main()
