import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os

from attacks.methods.soda_impl.build_preferences import build_preference_records
from defenses.core.runner.offline import run_offline_batch_defense
from runs.defense.prepare_soda_seqkd_reuse import (
    align_negative_rows,
    aligned_query,
    read_jsonl,
    resolve_artifacts,
)
from runs.defense.prepare_soda_clean_reuse import (
    first_local_checkpoint,
    first_local_preferences,
    valid_preferences,
)


class SODADefenseReplayTests(unittest.TestCase):
    def test_clean_soda_inputs_are_resolved_locally(self):
        with tempfile.TemporaryDirectory() as folder:
            storage = Path(folder)
            checkpoint = (
                storage / "outputs" / "attacks_full" / "seqkd" / "clean-run"
                / "training" / "seqkd" / "budget_1000" / "seqkd" / "checkpoint-final"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B"}), encoding="utf-8"
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            preferences = (
                storage / "outputs" / "attacks_full" / "soda" / "clean-run"
                / "train_data" / "preferences.jsonl"
            )
            preferences.parent.mkdir(parents=True)
            preferences.write_text(
                "".join(json.dumps({"prompt": f"p{i}", "chosen": "yes", "rejected": "no"}) + "\n"
                        for i in range(999)),
                encoding="utf-8",
            )

            self.assertEqual(first_local_checkpoint(storage, None), checkpoint.resolve())
            self.assertEqual(first_local_preferences(storage, None), preferences.resolve())
            self.assertEqual(valid_preferences(preferences), 999)

    def test_evaluation_resolves_corrected_local_soda(self):
        from evaluation.defense_eval.evaluate import local_corrected_soda_checkpoint

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "clean" / "checkpoint-final"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B"}), encoding="utf-8"
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            manifest = root / "clean" / "attack_manifest.json"
            manifest.write_text(json.dumps({"result": {"checkpoint_dir": str(checkpoint)}}), encoding="utf-8")

            with patch.dict(os.environ, {"SODA_FULL_OUTPUT_ROOT": str(root)}):
                resolved, source = local_corrected_soda_checkpoint("clean")

            self.assertEqual(resolved, checkpoint.resolve())
            self.assertEqual(source["source"], "local_corrected_soda")
            self.assertEqual(source["manifest"], str(manifest.resolve()))

    def test_query_alignment_accepts_both_artifact_schemas(self):
        self.assertEqual(
            aligned_query({"query_id": "q1", "query": "prompt"}),
            aligned_query({"prompt_id": "q1", "prompt": "prompt"}),
        )

    def test_jsonl_reader_rejects_non_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text('[]\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected an object"):
                read_jsonl(path)

    def test_negative_ids_are_remapped_to_replayed_transcript(self):
        transcript = [{"query_id": "run-specific-1", "query": "same prompt"}]
        negatives = [{"prompt_id": "stable-1", "prompt": "same prompt", "student_response": "no"}]
        aligned = align_negative_rows(transcript, negatives)
        self.assertEqual(aligned[0]["prompt_id"], "run-specific-1")
        self.assertEqual(aligned[0]["source_prompt_id"], "stable-1")
        preferences = build_preference_records(
            [{"prompt_id": "run-specific-1", "prompt": "same prompt", "teacher_response": "yes"}],
            aligned,
        )
        self.assertEqual(preferences[0]["chosen"], "yes")
        self.assertEqual(preferences[0]["rejected"], "no")

    def test_negative_alignment_rejects_different_prompts(self):
        with self.assertRaisesRegex(ValueError, "not aligned"):
            align_negative_rows(
                [{"query_id": "one", "query": "first"}],
                [{"prompt_id": "one", "prompt": "different"}],
            )

    def test_resolver_prefers_complete_local_artifacts_without_downloading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            storage = root / "storage"
            method_root = storage / "outputs" / "defenses" / "seqkd_b1000" / "ginsew"
            checkpoint = method_root / "checkpoint-final"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B"}), encoding="utf-8"
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            oracle = method_root / "full" / "oracle"
            (oracle / "artifacts").mkdir(parents=True)
            (oracle / "teacher_received_queries.jsonl").write_text("{}\n", encoding="utf-8")
            (method_root / "full" / "defense_run_manifest.json").write_text("{}\n", encoding="utf-8")
            transcript = oracle / "defended_teacher_transcript.jsonl"
            transcript.write_text(
                "".join(
                    json.dumps({"query_id": f"run-{i}", "query": f"prompt-{i}", "response": "yes"}) + "\n"
                    for i in range(1000)
                ),
                encoding="utf-8",
            )
            negatives = (
                storage / "outputs" / "defenses" / "soda_b1000" / "ginsew"
                / "attack" / "soda" / "run-1" / "train_data" / "student_negatives.jsonl"
            )
            negatives.parent.mkdir(parents=True)
            negatives.write_text(
                "".join(
                    json.dumps({"prompt_id": f"old-{i}", "prompt": f"prompt-{i}", "student_response": "no"}) + "\n"
                    for i in range(1000)
                ),
                encoding="utf-8",
            )
            aligned = root / "aligned.jsonl"
            result = resolve_artifacts("ginsew", aligned, storage)

            self.assertEqual(result["seqkd_source"], "local")
            self.assertEqual(result["student_negatives_source"], "local")
            self.assertEqual(result["warmup_checkpoint"], str(checkpoint.resolve()))
            self.assertEqual(result["source_student_negatives"], str(negatives.resolve()))
            self.assertEqual(read_jsonl(aligned)[0]["prompt_id"], "run-0")

    def test_local_manifest_resolves_nested_generator_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            storage = root / "storage"
            method_root = storage / "outputs" / "defenses" / "seqkd_b1000" / "adfp"
            checkpoint = root / "artifacts" / "checkpoint-final"
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B"}), encoding="utf-8"
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
            oracle = root / "artifacts" / "oracle"
            artifacts = oracle / "artifacts"
            artifacts.mkdir(parents=True)
            transcript = oracle / "defended_teacher_transcript.jsonl"
            transcript.write_text("{}\n", encoding="utf-8")
            query_log = oracle / "teacher_received_queries.jsonl"
            query_log.write_text("{}\n", encoding="utf-8")
            oracle_manifest = oracle / "defense_manifest.json"
            oracle_manifest.write_text("{}\n", encoding="utf-8")
            method_root.mkdir(parents=True)
            (method_root / "defense_run_manifest.json").write_text(json.dumps({
                "generator_result": {
                    "student_checkpoint_path": str(checkpoint),
                    "defended_transcript_path": str(transcript),
                    "defense_artifacts_dir": str(artifacts),
                    "teacher_query_log_path": str(query_log),
                    "oracle_manifest_path": str(oracle_manifest),
                }
            }), encoding="utf-8")

            from runs.defense.prepare_soda_seqkd_reuse import find_local_seqkd_bundle

            bundle = find_local_seqkd_bundle("adfp", storage)
            self.assertIsNotNone(bundle)
            self.assertEqual(bundle["checkpoint"], checkpoint.resolve())
            self.assertEqual(bundle["oracle_manifest"], oracle_manifest.resolve())

    def test_replay_skips_oracle_generation_and_records_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transcript = root / "source" / "transcripts" / "defended_teacher.jsonl"
            transcript.parent.mkdir(parents=True)
            transcript.write_text('{"query_id":"q1","query":"prompt","response":"answer"}\n', encoding="utf-8")
            artifacts = root / "source" / "artifacts"
            artifacts.mkdir()
            query_log = root / "source" / "teacher_received_queries.jsonl"
            query_log.write_text('{"query_id":"q1","query":"prompt"}\n', encoding="utf-8")
            oracle_manifest = root / "source" / "defense_manifest.json"
            oracle_manifest.write_text("{}\n", encoding="utf-8")
            attack_manifest = root / "attack_manifest.json"
            attack_manifest.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()

            args = SimpleNamespace(
                attack="soda", budget=1000, countermeasure="none", run_id="replay",
                output_dir=str(root / "result"), reuse_defense_transcript=str(transcript),
                reuse_defense_artifacts=str(artifacts), reuse_teacher_query_log=str(query_log),
                reuse_oracle_manifest=str(oracle_manifest), no_detector=True,
                teacher_model="teacher", student_model="student", query_pool="pool",
                query_ordering="ordering", teacher_mode="chat", teacher_temperature=0.0,
                teacher_top_p=1.0, teacher_max_tokens=1536, stage1_config="stage.yaml",
                dry_run=False,
            )
            payload = {"manifest_path": str(attack_manifest), "checkpoint_path": str(checkpoint)}
            with patch("defenses.core.runner.offline.run_attack_process", return_value=payload) as attack, \
                    patch("defenses.core.runner.offline._generation_command", side_effect=AssertionError("generated")):
                result = run_offline_batch_defense(defense="adfp", args=args, attack_extra_args=[])

            self.assertEqual(result.metadata["execution_mode"], "offline_replay")
            self.assertEqual(result.generator_result.defended_transcript_path, transcript.resolve())
            self.assertEqual(result.generator_result.defense_artifacts_dir, artifacts.resolve())
            self.assertEqual(
                attack.call_args.kwargs["teacher_transcript_path"], transcript.resolve()
            )
            saved = json.loads((root / "result" / "defense_run_manifest.json").read_text())
            self.assertEqual(saved["metadata"]["execution_mode"], "offline_replay")


if __name__ == "__main__":
    unittest.main()
