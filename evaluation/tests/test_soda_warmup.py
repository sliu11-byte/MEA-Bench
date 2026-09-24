import json
import tempfile
import unittest
from pathlib import Path

from attacks.core.base import AttackRunConfig
from attacks.core.manifest import file_sha256
from attacks.methods.soda_warmup import SODAWarmupError, resolve_soda_warmup


class SODAWarmupTests(unittest.TestCase):
    def config(self, root: Path, **kwargs) -> AttackRunConfig:
        pool = root / "pool.json"
        ordering = root / "ordering.json"
        stage = root / "stage.yaml"
        pool.write_text('{"data": []}\n', encoding="utf-8")
        ordering.write_text('{"ids": []}\n', encoding="utf-8")
        stage.write_text("{}\n", encoding="utf-8")
        values = {
            "attack": "soda",
            "budget": 100,
            "query_pool_path": pool,
            "query_ordering_path": ordering,
            "output_dir": root / "outputs",
            "stage1_config_path": stage,
            "student_model": "example/base-student",
        }
        values.update(kwargs)
        return AttackRunConfig(**values)

    def write_seqkd_run(self, config: AttackRunConfig, transcript: Path) -> tuple[Path, Path]:
        transcript.mkdir(parents=True, exist_ok=True)
        transcript_jsonl = transcript / "transcript.jsonl"
        if not transcript_jsonl.exists():
            transcript_jsonl.write_text(
                '{"query_id":"q0","teacher_response":"answer"}\n', encoding="utf-8"
            )
        run_dir = config.output_dir / "seqkd" / "20260919T000000Z_seqkd_b100"
        checkpoint = run_dir / "training" / "seqkd" / "budget_100" / "seqkd" / "checkpoint-final"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": config.student_model}), encoding="utf-8"
        )
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        manifest_path = run_dir / "attack_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "created_at": "2026-09-19T00:00:00+00:00",
                    "query_pool_sha256": file_sha256(config.query_pool_path),
                    "query_ordering_sha256": file_sha256(config.query_ordering_path),
                    "run_config": {
                        "budget": config.budget,
                        "student_model": config.student_model,
                        "transcript_dir": str(transcript),
                    },
                    "result": {
                        "status": "completed",
                        "run_id": run_dir.name,
                        "checkpoint_dir": str(checkpoint),
                        "artifacts": {"teacher_transcript": {"bundle_dir": str(transcript)}},
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest_path, checkpoint

    def test_auto_discovers_compatible_seqkd_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transcript = root / "teacher"
            transcript.mkdir()
            config = self.config(root)
            manifest, checkpoint = self.write_seqkd_run(config, transcript)

            result = resolve_soda_warmup(config, transcript_dir=transcript)

            self.assertEqual(result.checkpoint, str(checkpoint.resolve()))
            self.assertEqual(result.source, "auto_seqkd")
            self.assertEqual(result.manifest_path, str(manifest.resolve()))

    def test_auto_discovers_seqkd_transcript_without_its_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transcript = root / "teacher"
            transcript.mkdir()
            (transcript / "transcript.jsonl").write_text(
                '{"query_id":"q0","teacher_response":"answer"}\n', encoding="utf-8"
            )
            config = self.config(root)
            _, checkpoint = self.write_seqkd_run(config, transcript)

            result = resolve_soda_warmup(config, transcript_dir=None)

            self.assertEqual(result.checkpoint, str(checkpoint.resolve()))
            self.assertEqual(result.transcript_dir, str(transcript.resolve()))

    def test_compares_ordering_identity_instead_of_path_dependent_file_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transcript = root / "teacher"
            transcript.mkdir()
            config = self.config(root)
            config.query_ordering_path.write_text(
                json.dumps({"ordering_hash": "same-order", "source_query_pool": "/blue/cache/pool.json"}),
                encoding="utf-8",
            )
            manifest_path, checkpoint = self.write_seqkd_run(config, transcript)
            (transcript / "ordering.json").write_text(
                json.dumps({"ordering_hash": "same-order", "source_query_pool": "/home/cache/pool.json"}),
                encoding="utf-8",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["query_ordering_sha256"] = "path-dependent-hash-from-old-run"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = resolve_soda_warmup(config, transcript_dir=transcript)

            self.assertEqual(result.checkpoint, str(checkpoint.resolve()))

    def test_rejects_incompatible_teacher_transcript(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seqkd_transcript = root / "teacher-a"
            soda_transcript = root / "teacher-b"
            seqkd_transcript.mkdir()
            soda_transcript.mkdir()
            config = self.config(root)
            self.write_seqkd_run(config, seqkd_transcript)

            with self.assertRaisesRegex(SODAWarmupError, "teacher transcript differs"):
                resolve_soda_warmup(config, transcript_dir=soda_transcript)

    def test_accepts_identical_transcript_copied_to_another_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seqkd_transcript = root / "teacher-a"
            soda_transcript = root / "teacher-b"
            seqkd_transcript.mkdir()
            soda_transcript.mkdir()
            content = '{"query_id":"q0","teacher_response":"answer"}\n'
            (seqkd_transcript / "transcript.jsonl").write_text(content, encoding="utf-8")
            (soda_transcript / "transcript.jsonl").write_text(content, encoding="utf-8")
            config = self.config(root)
            _, checkpoint = self.write_seqkd_run(config, seqkd_transcript)

            result = resolve_soda_warmup(config, transcript_dir=soda_transcript)

            self.assertEqual(result.checkpoint, str(checkpoint.resolve()))

    def test_rejects_base_student_as_explicit_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            transcript = root / "teacher"
            transcript.mkdir()
            config = self.config(root, warmup_model="example/base-student")

            with self.assertRaisesRegex(SODAWarmupError, "untrained base student"):
                resolve_soda_warmup(config, transcript_dir=transcript)


if __name__ == "__main__":
    unittest.main()
