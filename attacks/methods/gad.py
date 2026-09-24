from __future__ import annotations

from pathlib import Path
from typing import Any

from attacks.core.base import AttackResult, AttackRunConfig
from attacks.methods.stage1_budget import patched_stage1_env


class GADError(RuntimeError):
    pass


class GADAttacker:
    name = "gad"

    def run(self, config: AttackRunConfig, *, run_id: str, run_dir: Path) -> AttackResult:
        transcript_root = run_dir / "transcripts"
        warmup_root = run_dir / "warmup"
        training_root = run_dir / "training"
        artifacts: dict[str, Any] = {
            "planned_transcript_root": str(transcript_root),
            "planned_warmup_root": str(warmup_root),
            "planned_training_root": str(training_root),
            "algorithm": "GAD: SeqKD warmup + online Bradley-Terry discriminator + GRPO-style student update.",
            "planned_steps": [
                "collect_or_load_offline_teacher_transcript",
                "warmup_generator_with_seqkd_sft",
                "warmup_discriminator_with_bradley_terry_pairs",
                "alternate_discriminator_reward_grpo_update_and_bradley_terry_update",
                "export_student_checkpoint_and_discriminator_artifact",
            ],
        }
        if config.dry_run:
            return AttackResult(
                attack=self.name,
                budget=config.budget,
                run_id=run_id,
                output_dir=run_dir,
                checkpoint_dir=None,
                status="dry_run",
                artifacts=artifacts,
            )
        if config.student_model is None:
            raise GADError("GAD requires --student-model as the generator/base student model.")

        from attacks.methods.gad_impl.train_gad import GADTrainingConfig, train_gad, _resolve_device
        from attacks.methods.stage1_budget_impl.build_teacher_transcript import build_teacher_transcript
        from attacks.methods.stage1_budget_impl.seqkd_train import SeqKDTrainingConfig, train_seqkd

        _resolve_device(config.gad_generator_device)
        _resolve_device(config.gad_discriminator_device)

        with patched_stage1_env(config, transcript_root=transcript_root):
            if config.transcript_dir is None:
                transcript_result = build_teacher_transcript(
                    config_path=config.stage1_config_path,
                    budget=config.budget,
                    backend_override=config.teacher_backend,
                    output_dir_override=transcript_root,
                    dry_run=False,
                    validate_only=False,
                )
                artifacts["teacher_transcript"] = transcript_result
                transcript_dir = Path(str(transcript_result["bundle_dir"]))
            else:
                transcript_dir = config.transcript_dir
                artifacts["teacher_transcript"] = {
                    "status": "provided",
                    "bundle_dir": str(transcript_dir),
                    "budget": config.budget,
                }

        warmup_result = train_seqkd(
            SeqKDTrainingConfig(
                transcript_dir=str(transcript_dir),
                output_dir=str(warmup_root / "seqkd"),
                student_model_id=config.student_model,
                requested_budget=config.budget,
                seed=int(config.seed or 0),
                max_seq_length=config.gad_max_seq_length,
                learning_rate=config.gad_warmup_learning_rate,
                num_train_epochs=config.gad_warmup_epochs,
                max_steps=-1,
                per_device_train_batch_size=config.gad_per_device_train_batch_size,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=config.gad_gradient_accumulation_steps,
                optim="adamw_torch",
                lr_scheduler_type="linear",
                warmup_ratio=0.0,
                fp16=False,
                bf16=config.bf16,
                weight_decay=0.0,
                max_grad_norm=1.0,
                gradient_checkpointing=config.gradient_checkpointing,
                use_lora=config.use_lora,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                allow_sampling_transcript=False,
                include_eos_in_loss=True,
                save_total_limit=1,
                logging_steps=1,
                save_strategy="no",
                report_to=(),
            )
        )
        artifacts["generator_warmup"] = {
            "checkpoint": str(warmup_result.checkpoint_dir),
            "manifest_path": str(warmup_result.manifest_path),
            "dataset_stats": warmup_result.dataset_stats.to_dict(),
        }

        gad_result = train_gad(
            GADTrainingConfig(
                transcript_dir=str(transcript_dir),
                output_dir=str(training_root / "gad"),
                generator_model_id=str(warmup_result.checkpoint_dir),
                discriminator_model_id=config.warmup_model or config.student_model,
                requested_budget=config.budget,
                seed=int(config.seed or 0),
                group_size=config.gad_group_size,
                kl_beta=config.gad_kl_beta,
                learning_rate=config.gad_learning_rate,
                discriminator_learning_rate=config.gad_discriminator_learning_rate,
                gad_epochs=config.gad_epochs,
                discriminator_warmup_steps=config.gad_discriminator_warmup_steps,
                max_steps=config.gad_max_steps,
                per_device_train_batch_size=config.gad_per_device_train_batch_size,
                gradient_accumulation_steps=config.gad_gradient_accumulation_steps,
                max_seq_length=config.gad_max_seq_length,
                max_prompt_length=config.gad_max_prompt_length,
                max_response_length=config.gad_max_response_length,
                temperature=config.gad_temperature,
                top_p=config.gad_top_p,
                bf16=config.bf16,
                use_lora=config.use_lora,
                gradient_checkpointing=config.gradient_checkpointing,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                offload_inactive_models=config.gad_offload_inactive_models,
                memory_log_interval=config.gad_memory_log_interval,
                generator_device=config.gad_generator_device,
                discriminator_device=config.gad_discriminator_device,
            )
        )
        artifacts["training"] = {
            "checkpoint": str(gad_result.checkpoint_dir),
            "discriminator": str(gad_result.discriminator_dir),
            "manifest_path": str(gad_result.manifest_path),
            "metrics": gad_result.metrics,
        }
        return AttackResult(
            attack=self.name,
            budget=config.budget,
            run_id=run_id,
            output_dir=run_dir,
            checkpoint_dir=gad_result.checkpoint_dir,
            status="completed",
            artifacts=artifacts,
            metrics=gad_result.metrics,
        )
