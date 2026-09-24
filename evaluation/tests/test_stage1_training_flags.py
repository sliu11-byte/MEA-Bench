import os
import unittest
from pathlib import Path
from unittest.mock import patch

from attacks.core.base import AttackRunConfig
from attacks.methods.stage1_budget import patched_stage1_env
from attacks.methods.stage1_budget_impl.run_stage1_budget import _as_bool, _load_yaml


ROOT = Path(__file__).resolve().parents[2]


class Stage1TrainingFlagsTests(unittest.TestCase):
    def config(self, **kwargs):
        return AttackRunConfig(
            attack="seqkd", budget=1000, query_pool_path=Path("pool.json"),
            output_dir=Path("output"),
            stage1_config_path=ROOT / "attacks/configs/formal_stage1_budget.yaml",
            **kwargs,
        )

    def test_runtime_flags_reach_formal_yaml_and_environment_is_restored(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), patch.dict(os.environ, {
                "STAGE1_SEQKD_USE_LORA": str(not enabled).lower(),
                "LORD_USE_LORA": str(not enabled).lower(),
            }, clear=True):
                original = dict(os.environ)
                config = self.config(use_lora=enabled, bf16=enabled,
                                     gradient_checkpointing=enabled,
                                     lora_r=8, lora_alpha=24, lora_dropout=0.1)
                with patched_stage1_env(config, transcript_root=Path("transcripts")):
                    yaml = _load_yaml(config.stage1_config_path)
                    for section in ("seqkd_training", "lord_training"):
                        training = yaml[section]
                        for flag in ("use_lora", "bf16", "gradient_checkpointing"):
                            self.assertEqual(_as_bool(training[flag]), enabled)
                        self.assertEqual(int(training["lora_r"]), 8)
                        self.assertEqual(int(training["lora_alpha"]), 24)
                        self.assertEqual(float(training["lora_dropout"]), 0.1)
                self.assertEqual(dict(os.environ), original)

    def test_defaults_enable_lora(self):
        with patch.dict(os.environ, {}, clear=True):
            config = self.config()
            with patched_stage1_env(config, transcript_root=Path("transcripts")):
                yaml = _load_yaml(config.stage1_config_path)
                self.assertTrue(_as_bool(yaml["seqkd_training"]["use_lora"]))
                self.assertTrue(_as_bool(yaml["lord_training"]["use_lora"]))

    def test_environment_is_restored_on_training_failure(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with patched_stage1_env(self.config(), transcript_root=Path("transcripts")):
                    raise RuntimeError("training failed")
            self.assertEqual(dict(os.environ), {})


if __name__ == "__main__":
    unittest.main()
