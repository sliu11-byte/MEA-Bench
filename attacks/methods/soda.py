from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from attacks.core.base import AttackResult, AttackRunConfig
from attacks.methods.stage1_budget import patched_stage1_env
from attacks.methods.soda_warmup import SODAWarmupError, resolve_soda_warmup


class SODAAttackError(RuntimeError):
    pass


class SODAAttacker:
    name = "soda"

    def run(self, config: AttackRunConfig, *, run_id: str, run_dir: Path) -> AttackResult:
        transcript_root = run_dir / "transcripts"
        train_data_dir = run_dir / "train_data"
        checkpoint_dir = run_dir / "checkpoints" / "dpo"
        log_dir = run_dir / "logs"
        student_jsonl = config.soda_student_negatives_jsonl or (train_data_dir / "student_negatives.jsonl")
        preference_jsonl = config.soda_preferences_jsonl or (train_data_dir / "preferences.jsonl")
        warmup_model = config.warmup_model
        student_request_model = config.student_request_model or config.student_model

        artifacts: dict[str, Any] = {
            "planned_transcript_root": str(transcript_root),
            "planned_train_data_dir": str(train_data_dir),
            "planned_checkpoint_dir": str(checkpoint_dir),
            "stage1_config_path": str(config.stage1_config_path),
            "warmup_model": warmup_model,
            "warmup_source": "explicit" if warmup_model else "auto_seqkd",
            "student_request_model": student_request_model,
            "student_negatives_jsonl": str(student_jsonl),
            "preferences_jsonl": str(preference_jsonl),
            "reuse_student_negatives": config.soda_student_negatives_jsonl is not None,
            "reuse_preferences": config.soda_preferences_jsonl is not None,
            "execution_stage": config.execution_stage,
        }

        if config.dry_run:
            artifacts["teacher_transcript"] = {
                "status": "provided" if config.transcript_dir is not None else "planned",
                "bundle_dir": None if config.transcript_dir is None else str(config.transcript_dir),
                "budget": config.budget,
            }
            artifacts["soda_steps"] = [
                "collect_student_negatives",
                "build_preferences",
                "train_dpo",
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

        if config.execution_stage == "train":
            if not preference_jsonl.exists():
                raise SODAAttackError(f"prepared SODA preferences JSONL does not exist: {preference_jsonl}")
        elif config.soda_preferences_jsonl is not None:
            if not preference_jsonl.exists():
                raise SODAAttackError(f"SODA preferences JSONL does not exist: {preference_jsonl}")
        elif config.soda_student_negatives_jsonl is not None:
            if not student_jsonl.exists():
                raise SODAAttackError(f"SODA student negatives JSONL does not exist: {student_jsonl}")
        else:
            if config.student_endpoint_url is None:
                raise SODAAttackError("SODA requires --student-endpoint-url unless --soda-student-negatives-jsonl or --soda-preferences-jsonl is provided.")
            if student_request_model is None:
                raise SODAAttackError("SODA requires --student-request-model or --student-model unless precomputed SODA data is provided.")

        warmup = None
        if config.transcript_dir is None and config.warmup_model is None:
            try:
                warmup = resolve_soda_warmup(config, transcript_dir=None)
            except SODAWarmupError as exc:
                raise SODAAttackError(str(exc)) from exc
            if warmup.transcript_dir is None:
                raise SODAAttackError(f"Auto-discovered SeqKD run has no teacher transcript: {warmup.manifest_path}")
            transcript_dir = Path(warmup.transcript_dir)
            transcript_result = {
                "status": "reused_from_seqkd",
                "bundle_dir": str(transcript_dir),
                "budget": config.budget,
                "seqkd_manifest_path": warmup.manifest_path,
            }

        with patched_stage1_env(config, transcript_root=transcript_root):
            if warmup is not None:
                pass
            elif config.transcript_dir is None:
                from attacks.methods.stage1_budget_impl.build_teacher_transcript import build_teacher_transcript

                transcript_result = build_teacher_transcript(
                    config_path=config.stage1_config_path,
                    budget=config.budget,
                    backend_override=config.teacher_backend,
                    output_dir_override=transcript_root,
                    dry_run=False,
                    validate_only=False,
                )
                transcript_dir = Path(str(transcript_result["bundle_dir"]))
            else:
                transcript_dir = config.transcript_dir
                transcript_result = {
                    "status": "provided",
                    "bundle_dir": str(transcript_dir),
                    "budget": config.budget,
                }

        teacher_jsonl = transcript_dir / "transcript.jsonl"
        if not teacher_jsonl.exists():
            raise SODAAttackError(f"teacher transcript JSONL does not exist: {teacher_jsonl}")
        artifacts["teacher_transcript"] = transcript_result
        artifacts["teacher_jsonl"] = str(teacher_jsonl)

        if warmup is None:
            try:
                warmup = resolve_soda_warmup(config, transcript_dir=transcript_dir)
            except SODAWarmupError as exc:
                raise SODAAttackError(str(exc)) from exc
        warmup_model = warmup.checkpoint
        print(f"[SODA] warmup source: {warmup.source}")
        print(f"[SODA] warmup checkpoint: {warmup_model}")
        print(f"[SODA] teacher transcript: {transcript_dir}")
        artifacts.update(
            {
                "warmup_model": warmup_model,
                "warmup_source": warmup.source,
                "warmup_manifest_path": warmup.manifest_path,
                "warmup_run_id": warmup.run_id,
            }
        )

        train_data_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

        collect_cmd = [
            sys.executable,
            "-m",
            "attacks.methods.soda_impl.collect_student_negatives",
            "--teacher-jsonl",
            str(teacher_jsonl),
            "--output-jsonl",
            str(student_jsonl),
            "--base-url",
            config.student_endpoint_url,
            "--api-key",
            config.student_api_key,
            "--request-model-name",
            student_request_model,
            "--mode",
            config.student_mode,
            "--temperature",
            str(config.student_temperature),
            "--top-p",
            str(config.student_top_p),
            "--max-tokens",
            str(config.student_max_tokens),
            "--seed",
            str(config.seed if config.seed is not None else 42),
            "--limit",
            str(config.budget),
        ]
        build_cmd = [
            sys.executable,
            "-m",
            "attacks.methods.soda_impl.build_preferences",
            "--teacher-jsonl",
            str(teacher_jsonl),
            "--student-jsonl",
            str(student_jsonl),
            "--output-jsonl",
            str(preference_jsonl),
        ]
        train_cmd = [
            sys.executable,
            "-m",
            "attacks.methods.soda_impl.train_dpo",
            "--preference-jsonl",
            str(preference_jsonl),
            "--warmup-model",
            warmup_model,
            "--output-dir",
            str(checkpoint_dir),
            "--beta",
            str(config.soda_beta),
            "--learning-rate",
            str(config.soda_learning_rate),
            "--epochs",
            str(config.soda_epochs),
            "--per-device-train-batch-size",
            str(config.soda_per_device_train_batch_size),
            "--gradient-accumulation-steps",
            str(config.soda_gradient_accumulation_steps),
            "--max-length",
            str(config.soda_max_length),
            "--max-prompt-length",
            str(config.soda_max_prompt_length),
            "--max-grad-norm",
            str(config.soda_max_grad_norm),
            "--nonfinite-gradient-retries",
            str(config.soda_nonfinite_gradient_retries),
            "--bf16" if config.bf16 else "--no-bf16",
            "--use-lora" if config.use_lora else "--no-use-lora",
            "--gradient-checkpointing" if config.gradient_checkpointing else "--no-gradient-checkpointing",
            "--lora-r",
            str(config.lora_r),
            "--lora-alpha",
            str(config.lora_alpha),
            "--lora-dropout",
            str(config.lora_dropout),
        ]

        commands = {}
        if config.execution_stage != "train" and config.soda_preferences_jsonl is None:
            if config.soda_student_negatives_jsonl is None:
                commands["collect_student_negatives"] = collect_cmd
            commands["build_preferences"] = build_cmd
        if config.execution_stage != "prepare":
            commands["train_dpo"] = train_cmd
        for name, command in commands.items():
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
                raise SODAAttackError(f"SODA step {name} failed with exit code {proc.returncode}; see {log_path}")

        artifacts.update(
            {
                "student_negatives_jsonl": str(student_jsonl),
                "preference_jsonl": str(preference_jsonl),
                "checkpoint_dir": str(checkpoint_dir) if config.execution_stage != "prepare" else None,
                "training_safety_json": str(checkpoint_dir / "training_safety.json") if config.execution_stage != "prepare" else None,
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
        )
