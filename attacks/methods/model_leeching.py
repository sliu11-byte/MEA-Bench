from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from attacks.core.base import AttackResult, AttackRunConfig


class ModelLeechingError(RuntimeError):
    pass


class ModelLeechingAttacker:
    name = "model_leeching"

    def run(self, config: AttackRunConfig, *, run_id: str, run_dir: Path) -> AttackResult:
        query_dir = run_dir / "queries"
        transcript_dir = run_dir / "teacher_transcripts"
        train_data_dir = run_dir / "train_data"
        checkpoint_dir = run_dir / "checkpoints" / "lora_sft"
        log_dir = run_dir / "logs"

        templated_queries = query_dir / "templated_queries.jsonl"
        teacher_jsonl = transcript_dir / "teacher_raw.jsonl"
        clean_jsonl = train_data_dir / "clean_records.jsonl"
        rejected_jsonl = train_data_dir / "rejected_records.jsonl"
        cleaning_stats_json = train_data_dir / "cleaning_stats.json"
        sft_jsonl = train_data_dir / "model_leeching_sft.jsonl"

        teacher_request_model = config.teacher_request_model or config.teacher_model
        artifacts: dict[str, Any] = {
            "planned_query_dir": str(query_dir),
            "planned_transcript_dir": str(transcript_dir),
            "planned_train_data_dir": str(train_data_dir),
            "planned_checkpoint_dir": str(checkpoint_dir),
            "teacher_request_model": teacher_request_model,
            "execution_stage": config.execution_stage,
        }

        if config.dry_run:
            artifacts["model_leeching_steps"] = [
                "build_queries",
                "collect_teacher",
                "parse_clean_validate",
                "to_sft",
                "train_lora_sft",
            ]
            return AttackResult(
                attack=self.name,
                budget=config.budget,
                run_id=run_id,
                output_dir=run_dir,
                checkpoint_dir=None,
                status="dry_run",
                artifacts=artifacts,
            )

        if config.execution_stage != "train" and config.teacher_endpoint_url is None:
            raise ModelLeechingError("Model Leeching requires --teacher-endpoint-url.")
        if config.execution_stage != "train" and teacher_request_model is None:
            raise ModelLeechingError("Model Leeching requires --teacher-request-model or --teacher-model.")
        if config.execution_stage != "prepare" and config.student_model is None:
            raise ModelLeechingError("Model Leeching requires --student-model for final LoRA SFT training.")

        for path in [query_dir, transcript_dir, train_data_dir, checkpoint_dir, log_dir]:
            path.mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

        build_queries_cmd = [
            sys.executable,
            "-m",
            "attacks.methods.model_leeching_impl.build_queries",
            "--query-pool",
            str(config.query_pool_path),
            "--output-jsonl",
            str(templated_queries),
            "--budget",
            str(config.budget),
        ]
        if config.query_ordering_path is not None:
            build_queries_cmd.extend(["--query-ordering", str(config.query_ordering_path)])

        prepare_commands = [
            ("build_queries", build_queries_cmd),
            (
                "collect_teacher",
                [
                    sys.executable,
                    "-m",
                    "attacks.methods.model_leeching_impl.collect_teacher",
                    "--query-jsonl",
                    str(templated_queries),
                    "--output-jsonl",
                    str(teacher_jsonl),
                    "--base-url",
                    config.teacher_endpoint_url,
                    "--api-key",
                    str(config.teacher_api_key or "EMPTY"),
                    "--request-model-name",
                    teacher_request_model,
                    "--mode",
                    config.teacher_mode,
                    "--temperature",
                    str(config.teacher_temperature),
                    "--top-p",
                    str(config.teacher_top_p),
                    "--max-tokens",
                    str(config.teacher_max_tokens),
                    "--seed",
                    str(config.seed if config.seed is not None else 42),
                ],
            ),
            (
                "parse_clean_validate",
                [
                    sys.executable,
                    "-m",
                    "attacks.methods.model_leeching_impl.parse_clean_validate",
                    "--teacher-jsonl",
                    str(teacher_jsonl),
                    "--output-jsonl",
                    str(clean_jsonl),
                    "--stats-json",
                    str(cleaning_stats_json),
                    "--rejected-jsonl",
                    str(rejected_jsonl),
                ],
            ),
            (
                "to_sft",
                [
                    sys.executable,
                    "-m",
                    "attacks.methods.model_leeching_impl.to_sft",
                    "--clean-jsonl",
                    str(clean_jsonl),
                    "--output-jsonl",
                    str(sft_jsonl),
                ],
            ),
        ]
        train_command = (
            "train_lora_sft",
            [
                    sys.executable,
                    "-m",
                    "attacks.methods.model_leeching_impl.train_lora_sft",
                    "--sft-jsonl",
                    str(sft_jsonl),
                    "--base-model",
                    config.student_model,
                    "--output-dir",
                    str(checkpoint_dir),
                    "--learning-rate",
                    str(config.model_leeching_learning_rate),
                    "--epochs",
                    str(config.model_leeching_epochs),
                    "--lora-r",
                    str(config.model_leeching_lora_r),
                    "--lora-alpha",
                    str(config.model_leeching_lora_alpha),
                    "--lora-dropout",
                    str(config.model_leeching_lora_dropout),
                    "--per-device-train-batch-size",
                    str(config.model_leeching_per_device_train_batch_size),
                    "--gradient-accumulation-steps",
                    str(config.model_leeching_gradient_accumulation_steps),
            ],
        )

        commands = []
        if config.execution_stage != "train":
            commands.extend(prepare_commands)
        if config.execution_stage != "prepare":
            if not sft_jsonl.exists() and config.execution_stage == "train":
                raise ModelLeechingError(f"prepared Model Leeching SFT data does not exist: {sft_jsonl}")
            commands.append(train_command)
        for name, command in commands:
            self._run_command(name, command, repo_root=repo_root, env=env, log_dir=log_dir, artifacts=artifacts)

        cleaning_stats = json.loads(cleaning_stats_json.read_text(encoding="utf-8"))
        artifacts.update(
            {
                "templated_queries_jsonl": str(templated_queries),
                "teacher_jsonl": str(teacher_jsonl),
                "clean_jsonl": str(clean_jsonl),
                "rejected_jsonl": str(rejected_jsonl),
                "cleaning_stats_json": str(cleaning_stats_json),
                "sft_jsonl": str(sft_jsonl),
                "checkpoint_dir": str(checkpoint_dir) if config.execution_stage != "prepare" else None,
                "cleaning_stats": cleaning_stats,
            }
        )
        return AttackResult(
            attack=self.name,
            budget=config.budget,
            run_id=run_id,
            output_dir=run_dir,
            checkpoint_dir=None if config.execution_stage == "prepare" else checkpoint_dir,
            status="prepared" if config.execution_stage == "prepare" else "completed",
            artifacts=artifacts,
            metrics={
                "effective_sample_rate": float(cleaning_stats.get("keep_rate", 0.0)),
            },
        )

    @staticmethod
    def _run_command(
        name: str,
        command: list[str],
        *,
        repo_root: Path,
        env: dict[str, str],
        log_dir: Path,
        artifacts: dict[str, Any],
    ) -> None:
        log_path = log_dir / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                command,
                cwd=str(repo_root),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        artifacts[f"{name}_log"] = str(log_path)
        if proc.returncode != 0:
            raise ModelLeechingError(f"Model Leeching step {name} failed with exit code {proc.returncode}; see {log_path}")
