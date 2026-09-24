from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from attacks.core.base import AttackResult, AttackRunConfig
from attacks.methods.qedks_impl.common import read_jsonl, write_jsonl


class QEDKSError(RuntimeError):
    pass


def wait_for_teacher_release() -> None:
    request_value = os.environ.get("QEDKS_RELEASE_TEACHER_REQUEST")
    ack_value = os.environ.get("QEDKS_RELEASE_TEACHER_ACK")
    if not request_value and not ack_value:
        return
    if not request_value or not ack_value:
        raise QEDKSError("both QEDKS teacher-release marker paths must be configured")
    request_path = Path(request_value)
    ack_path = Path(ack_value)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text("release teacher before student training\n", encoding="utf-8")
    timeout = float(os.environ.get("QEDKS_RELEASE_TEACHER_TIMEOUT_SECONDS", "300"))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ack_path.exists():
            return
        time.sleep(0.2)
    raise QEDKSError(f"timed out waiting for teacher release acknowledgement: {ack_path}")


class QEDKSAttacker:
    name = "qedks"

    def run(self, config: AttackRunConfig, *, run_id: str, run_dir: Path) -> AttackResult:
        plan_dir = run_dir / "query_plans"
        transcript_dir = run_dir / "teacher_transcripts"
        train_data_dir = run_dir / "train_data"
        checkpoint_dir = run_dir / "checkpoints" / "lora_sft"
        log_dir = run_dir / "logs"

        seed_plan = plan_dir / "seed_queries.jsonl"
        seed_teacher = transcript_dir / "seed_teacher.jsonl"
        template_plan = plan_dir / "template_queries.jsonl"
        template_ppl = plan_dir / "template_ppl.jsonl"
        scheduled_template_plan = plan_dir / "template_queries_scheduled.jsonl"
        template_teacher = transcript_dir / "template_teacher.jsonl"
        followup_plan = plan_dir / "followup_queries.jsonl"
        followup_teacher = transcript_dir / "followup_teacher.jsonl"
        combined_teacher = transcript_dir / "combined_teacher.jsonl"
        sft_jsonl = train_data_dir / "qedks_sft.jsonl"

        teacher_request_model = config.teacher_request_model or config.teacher_model
        seed_budget, template_budget, followup_budget = self._split_budget(config.budget)
        artifacts: dict[str, Any] = {
            "planned_query_plan_dir": str(plan_dir),
            "planned_transcript_dir": str(transcript_dir),
            "planned_train_data_dir": str(train_data_dir),
            "planned_checkpoint_dir": str(checkpoint_dir),
            "teacher_request_model": teacher_request_model,
            "budget_split": {
                "seed": seed_budget,
                "template": template_budget,
                "followup": followup_budget,
            },
            "execution_stage": config.execution_stage,
        }

        if config.dry_run:
            artifacts["qedks_steps"] = [
                "planner_seed",
                "collect_teacher_seed",
                "planner_template",
                "collect_teacher_template",
                "followup",
                "collect_teacher_followup",
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
            raise QEDKSError("QEDKS requires --teacher-endpoint-url.")
        if config.execution_stage != "train" and teacher_request_model is None:
            raise QEDKSError("QEDKS requires --teacher-request-model or --teacher-model.")
        if config.execution_stage != "prepare" and config.student_model is None:
            raise QEDKSError("QEDKS requires --student-model for final LoRA SFT training.")

        plan_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        train_data_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

        commands: list[tuple[str, list[str]]] = []
        if config.execution_stage != "train":
            commands.append((
            "planner_seed",
            [
                sys.executable,
                "-m",
                "attacks.methods.qedks_impl.planner",
                "--output-jsonl",
                str(seed_plan),
                "--budget",
                str(seed_budget),
            ],
            ))
            commands.append(("collect_teacher_seed", self._collect_teacher_cmd(config, seed_plan, seed_teacher, teacher_request_model)))

        if config.execution_stage != "train" and template_budget > 0:
            commands.append((
                "planner_template",
                [
                    sys.executable,
                    "-m",
                    "attacks.methods.qedks_impl.planner",
                    "--seed-transcript",
                    str(seed_teacher),
                    "--output-jsonl",
                    str(template_plan),
                    "--budget",
                    str(template_budget),
                ],
            ))
            if config.qedks_use_ppl_schedule:
                commands.append((
                    "score_student_ppl",
                    [
                        sys.executable,
                        "-m",
                        "attacks.methods.qedks_impl.score_student_ppl",
                        "--query-plan-jsonl",
                        str(template_plan),
                        "--student-model",
                        config.student_model,
                        "--output-jsonl",
                        str(template_ppl),
                        "--device",
                        config.qedks_ppl_device,
                    ],
                ))
                commands.append((
                    "planner_template_scheduled",
                    [
                        sys.executable,
                        "-m",
                        "attacks.methods.qedks_impl.planner",
                        "--seed-transcript",
                        str(seed_teacher),
                        "--student-ppl-jsonl",
                        str(template_ppl),
                        "--output-jsonl",
                        str(scheduled_template_plan),
                        "--budget",
                        str(template_budget),
                    ],
                ))
                template_for_teacher = scheduled_template_plan
            else:
                template_for_teacher = template_plan
            commands.append(("collect_teacher_template", self._collect_teacher_cmd(config, template_for_teacher, template_teacher, teacher_request_model)))

        if config.execution_stage != "train" and followup_budget > 0 and template_budget > 0:
            commands.append((
                "followup",
                [
                    sys.executable,
                    "-m",
                    "attacks.methods.qedks_impl.followup",
                    "--teacher-jsonl",
                    str(template_teacher),
                    "--output-jsonl",
                    str(followup_plan),
                    "--max-followups-per-answer",
                    str(config.qedks_max_followups_per_answer),
                    "--budget",
                    str(followup_budget),
                ],
            ))
            commands.append(("collect_teacher_followup", self._collect_teacher_cmd(config, followup_plan, followup_teacher, teacher_request_model)))

        for name, command in commands:
            self._run_command(name, command, repo_root=repo_root, env=env, log_dir=log_dir, artifacts=artifacts)

        if config.execution_stage != "train":
            teacher_parts = [seed_teacher]
            if template_teacher.exists():
                teacher_parts.append(template_teacher)
            if followup_teacher.exists():
                teacher_parts.append(followup_teacher)
            self._combine_jsonl(teacher_parts, combined_teacher, limit=config.budget)

        to_sft_cmd = [
            sys.executable,
            "-m",
            "attacks.methods.qedks_impl.to_sft",
            "--teacher-jsonl",
            str(combined_teacher),
            "--output-jsonl",
            str(sft_jsonl),
        ]
        train_cmd = [
            sys.executable,
            "-m",
            "attacks.methods.qedks_impl.train_lora_sft",
            "--sft-jsonl",
            str(sft_jsonl),
            "--base-model",
            config.student_model,
            "--output-dir",
            str(checkpoint_dir),
            "--learning-rate",
            str(config.qedks_learning_rate),
            "--epochs",
            str(config.qedks_epochs),
            "--lora-r",
            str(config.qedks_lora_r),
            "--lora-alpha",
            str(config.qedks_lora_alpha),
            "--lora-dropout",
            str(config.qedks_lora_dropout),
            "--per-device-train-batch-size",
            str(config.qedks_per_device_train_batch_size),
            "--gradient-accumulation-steps",
            str(config.qedks_gradient_accumulation_steps),
        ]
        if config.execution_stage != "train":
            self._run_command("to_sft", to_sft_cmd, repo_root=repo_root, env=env, log_dir=log_dir, artifacts=artifacts)
        if config.execution_stage == "prepare":
            artifacts.update({"combined_teacher_jsonl": str(combined_teacher), "sft_jsonl": str(sft_jsonl)})
            return AttackResult(
                attack=self.name,
                budget=config.budget,
                run_id=run_id,
                output_dir=run_dir,
                checkpoint_dir=None,
                status="prepared",
                artifacts=artifacts,
            )
        if not sft_jsonl.exists():
            raise QEDKSError(f"prepared QEDKS SFT data does not exist: {sft_jsonl}")
        if config.execution_stage == "all":
            wait_for_teacher_release()
        train_env = dict(env)
        requested_train_devices = train_env.get("QEDKS_TRAIN_CUDA_VISIBLE_DEVICES")
        if requested_train_devices:
            train_env["CUDA_VISIBLE_DEVICES"] = requested_train_devices
        else:
            visible_devices = train_env.get("CUDA_VISIBLE_DEVICES", "")
            if "," in visible_devices:
                train_env["CUDA_VISIBLE_DEVICES"] = visible_devices.split(",", 1)[0]
        self._run_command("train_lora_sft", train_cmd, repo_root=repo_root, env=train_env, log_dir=log_dir, artifacts=artifacts)
        artifacts["train_cuda_visible_devices"] = train_env.get("CUDA_VISIBLE_DEVICES")

        artifacts.update(
            {
                "combined_teacher_jsonl": str(combined_teacher),
                "sft_jsonl": str(sft_jsonl),
                "checkpoint_dir": str(checkpoint_dir),
            }
        )
        return AttackResult(
            attack=self.name,
            budget=config.budget,
            run_id=run_id,
            output_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            status="completed",
            artifacts=artifacts,
        )

    @staticmethod
    def _split_budget(budget: int) -> tuple[int, int, int]:
        seed_budget = min(3, budget)
        remaining = max(0, budget - seed_budget)
        template_budget = remaining // 2
        followup_budget = remaining - template_budget
        if remaining == 1:
            template_budget = 1
            followup_budget = 0
        return seed_budget, template_budget, followup_budget

    @staticmethod
    def _collect_teacher_cmd(config: AttackRunConfig, query_plan: Path, output_jsonl: Path, request_model: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "attacks.methods.qedks_impl.collect_teacher",
            "--query-plan-jsonl",
            str(query_plan),
            "--output-jsonl",
            str(output_jsonl),
            "--base-url",
            str(config.teacher_endpoint_url),
            "--api-key",
            str(config.teacher_api_key or "EMPTY"),
            "--request-model-name",
            request_model,
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
        ]

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
            raise QEDKSError(f"QEDKS step {name} failed with exit code {proc.returncode}; see {log_path}")

    @staticmethod
    def _combine_jsonl(paths: list[Path], output_path: Path, *, limit: int) -> None:
        records: list[dict[str, Any]] = []
        for path in paths:
            if path.exists():
                records.extend(read_jsonl(path))
        write_jsonl(output_path, records[:limit])
