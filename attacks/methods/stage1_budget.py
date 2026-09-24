from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator

from attacks.core.base import AttackResult, AttackRunConfig


@contextmanager
def patched_stage1_env(config: AttackRunConfig, *, transcript_root: Path) -> Iterator[None]:
    values: dict[str, str | None] = {
        "STAGE1_QUERY_POOL_PATH": str(config.query_pool_path),
        "STAGE1_OUTPUT_ROOT": str(transcript_root),
    }
    if config.query_ordering_path is not None:
        values["STAGE1_QUERY_ORDERING_PATH"] = str(config.query_ordering_path)
    if config.teacher_backend is not None:
        values["STAGE1_TEACHER_BACKEND"] = config.teacher_backend
    if config.teacher_model is not None:
        values["STAGE1_TEACHER_MODEL_PATH"] = config.teacher_model
    if config.teacher_request_model is not None:
        values["STAGE1_TEACHER_MODEL_NAME"] = config.teacher_request_model
    if config.teacher_endpoint_url is not None:
        values["STAGE1_TEACHER_BASE_URL"] = config.teacher_endpoint_url
    if config.teacher_api_key is not None:
        values["STAGE1_TEACHER_API_KEY"] = config.teacher_api_key
    values["STAGE1_TEACHER_MODE"] = config.teacher_mode
    if config.student_model is not None:
        values["STAGE1_STUDENT_MODEL_PATH"] = config.student_model
    if config.seed is not None:
        values["STAGE1_SEED"] = str(config.seed)

    # The stage-1 YAML reads environment placeholders rather than AttackRunConfig.
    for prefix in ("STAGE1_SEQKD", "LORD"):
        values.update({
            f"{prefix}_USE_LORA": str(config.use_lora).lower(),
            f"{prefix}_LORA_R": str(config.lora_r),
            f"{prefix}_LORA_ALPHA": str(config.lora_alpha),
            f"{prefix}_LORA_DROPOUT": str(config.lora_dropout),
            f"{prefix}_BF16": str(config.bf16).lower(),
            f"{prefix}_GRADIENT_CHECKPOINTING": str(config.gradient_checkpointing).lower(),
        })

    old_values: dict[str, str | None] = {}
    for key, value in values.items():
        old_values[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def dry_run_result(method_name: str, config: AttackRunConfig, *, run_id: str, run_dir: Path) -> AttackResult:
    transcript_root = run_dir / "transcripts"
    training_root = run_dir / "training"
    artifacts: dict[str, Any] = {
        "planned_transcript_root": str(transcript_root),
        "planned_training_root": str(training_root),
        "stage1_config_path": str(config.stage1_config_path),
        "teacher_transcript": {
            "status": (
                "provided_bundle"
                if config.transcript_dir is not None
                else "provided_external_jsonl"
                if config.teacher_transcript_path is not None
                else "planned"
            ),
            "bundle_dir": None if config.transcript_dir is None else str(config.transcript_dir),
            "source_path": None if config.teacher_transcript_path is None else str(config.teacher_transcript_path),
            "budget": config.budget,
        },
    }
    return AttackResult(
        attack=method_name,
        budget=config.budget,
        run_id=run_id,
        output_dir=run_dir,
        checkpoint_dir=None,
        status="dry_run",
        artifacts=artifacts,
    )


def run_stage1_budget_attacker(method_name: str, config: AttackRunConfig, *, run_id: str, run_dir: Path) -> AttackResult:
    if config.dry_run:
        return dry_run_result(method_name, config, run_id=run_id, run_dir=run_dir)

    from attacks.methods.stage1_budget_impl.build_teacher_transcript import build_teacher_transcript
    from attacks.methods.stage1_budget_impl.external_transcript import import_external_teacher_transcript
    from attacks.methods.stage1_budget_impl.run_stage1_budget import run_stage1_budget

    transcript_root = run_dir / "transcripts"
    training_root = run_dir / "training"
    artifacts: dict[str, Any] = {
        "planned_transcript_root": str(transcript_root),
        "planned_training_root": str(training_root),
        "stage1_config_path": str(config.stage1_config_path),
    }

    with patched_stage1_env(config, transcript_root=transcript_root):
        if config.transcript_dir is not None:
            transcript_dir = config.transcript_dir
            artifacts["teacher_transcript"] = {
                "status": "provided_bundle",
                "bundle_dir": str(transcript_dir),
                "budget": config.budget,
            }
        elif config.teacher_transcript_path is not None:
            transcript_result = import_external_teacher_transcript(
                config.teacher_transcript_path,
                transcript_root,
                budget=config.budget,
                teacher_model_id=config.teacher_model,
                seed=config.seed,
            )
            artifacts["teacher_transcript"] = transcript_result
            transcript_dir = Path(str(transcript_result["bundle_dir"]))
        else:
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

        training_result = run_stage1_budget(
            attack=method_name,
            budget=config.budget,
            config_path=config.stage1_config_path,
            transcript_dir=transcript_dir,
            output_dir=training_root,
        )
        artifacts["training"] = training_result
        return AttackResult(
            attack=method_name,
            budget=config.budget,
            run_id=run_id,
            output_dir=run_dir,
            checkpoint_dir=Path(str(training_result["checkpoint"])),
            status="completed",
            artifacts=artifacts,
            metrics={
                "wall_clock_seconds": float(training_result.get("wall_clock_seconds", 0.0)),
            },
        )
