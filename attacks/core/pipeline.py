from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Mapping

from .base import AttackResult, AttackRunConfig, BaseAttacker
from .manifest import file_sha256, stable_hash, utc_now_iso, write_json
from attacks.methods.lord import LoRDAttacker
from attacks.methods.seqkd import SeqKDAttacker
from attacks.methods.soda import SODAAttacker
from attacks.methods.qedks import QEDKSAttacker
from attacks.methods.model_leeching import ModelLeechingAttacker
from attacks.methods.gad import GADAttacker


class AttackPipelineError(RuntimeError):
    pass


ATTACKERS: Mapping[str, BaseAttacker] = {
    SeqKDAttacker.name: SeqKDAttacker(),
    LoRDAttacker.name: LoRDAttacker(),
    SODAAttacker.name: SODAAttacker(),
    QEDKSAttacker.name: QEDKSAttacker(),
    ModelLeechingAttacker.name: ModelLeechingAttacker(),
    GADAttacker.name: GADAttacker(),
}


def make_run_id(attack: str, budget: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{attack}_b{budget}"


class AttackPipeline:
    def __init__(self, attackers: Mapping[str, BaseAttacker] | None = None) -> None:
        self._attackers = dict(attackers or ATTACKERS)

    def run(self, config: AttackRunConfig) -> AttackResult:
        attacker = self._attackers.get(config.attack)
        if attacker is None:
            available = ", ".join(sorted(self._attackers))
            raise AttackPipelineError(f"unknown attack {config.attack!r}; available: {available}")
        if config.budget <= 0:
            raise AttackPipelineError("budget must be positive")
        if not config.query_pool_path.exists():
            raise AttackPipelineError(f"query pool does not exist: {config.query_pool_path}")
        if config.query_ordering_path is not None and not config.query_ordering_path.exists():
            raise AttackPipelineError(f"query ordering does not exist: {config.query_ordering_path}")
        if config.transcript_dir is not None and not config.transcript_dir.exists():
            raise AttackPipelineError(f"transcript dir does not exist: {config.transcript_dir}")
        if config.teacher_transcript_path is not None and not config.teacher_transcript_path.exists():
            raise AttackPipelineError(f"teacher transcript file does not exist: {config.teacher_transcript_path}")
        if config.transcript_dir is not None and config.teacher_transcript_path is not None:
            raise AttackPipelineError("use either --transcript-dir or --teacher-transcript-path, not both")
        if not config.stage1_config_path.exists():
            raise AttackPipelineError(f"stage1 config does not exist: {config.stage1_config_path}")

        run_id = make_run_id(config.attack, config.budget)
        run_dir = config.output_dir / config.attack / run_id
        if config.execution_stage == "train":
            if config.prepared_run_dir is None:
                raise AttackPipelineError("--execution-stage train requires --prepared-run-dir")
            run_dir = config.prepared_run_dir
            run_id = run_dir.name
            if not run_dir.is_dir():
                raise AttackPipelineError(f"prepared run directory does not exist: {run_dir}")
        elif config.prepared_run_dir is not None:
            raise AttackPipelineError("--prepared-run-dir is only valid with --execution-stage train")
        if config.attack == "qedks" and not config.dry_run and config.execution_stage != "train":
            candidates = sorted(
                (path for path in (config.output_dir / config.attack).glob(f"*_qedks_b{config.budget}")
                 if path.is_dir() and not (path / "attack_manifest.json").exists()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            signature = self._qedks_resume_signature(config)
            for candidate in candidates:
                marker = candidate / "qedks_resume_config.json"
                if not marker.exists() or json.loads(marker.read_text(encoding="utf-8")) == signature:
                    run_dir = candidate
                    run_id = candidate.name
                    break
        run_dir.mkdir(
            parents=True,
            exist_ok=(config.attack == "qedks" and not config.dry_run) or config.execution_stage == "train",
        )
        if config.attack == "qedks" and not config.dry_run and config.execution_stage != "train":
            write_json(run_dir / "qedks_resume_config.json", self._qedks_resume_signature(config))

        result = attacker.run(config, run_id=run_id, run_dir=run_dir)
        manifest_name = "prepare_manifest.json" if config.execution_stage == "prepare" else "attack_manifest.json"
        manifest_path = run_dir / manifest_name
        manifest_payload = {
            "schema_version": "attack_manifest_v1",
            "created_at": utc_now_iso(),
            "run_config": config.to_manifest(),
            "query_pool_sha256": file_sha256(config.query_pool_path),
            "query_ordering_sha256": None if config.query_ordering_path is None else file_sha256(config.query_ordering_path),
            "config_hash": stable_hash(config.to_manifest()),
            "result": result.to_manifest(),
        }
        write_json(manifest_path, manifest_payload)
        return AttackResult(
            attack=result.attack,
            budget=result.budget,
            run_id=result.run_id,
            output_dir=result.output_dir,
            checkpoint_dir=result.checkpoint_dir,
            status=result.status,
            manifest_path=manifest_path,
            artifacts={**dict(result.artifacts), "attack_manifest_path": str(manifest_path)},
            metrics=result.metrics,
        )

    @staticmethod
    def _qedks_resume_signature(config: AttackRunConfig) -> dict:
        return {
            "schema_version": "qedks_resume_v1",
            "budget": config.budget,
            "query_pool_sha256": file_sha256(config.query_pool_path),
            "query_ordering_sha256": None if config.query_ordering_path is None else file_sha256(config.query_ordering_path),
            "teacher_model": config.teacher_model,
            "teacher_request_model": config.teacher_request_model,
            "teacher_mode": config.teacher_mode,
            "teacher_temperature": config.teacher_temperature,
            "teacher_top_p": config.teacher_top_p,
            "teacher_max_tokens": config.teacher_max_tokens,
            "seed": config.seed,
            "qedks_max_followups_per_answer": config.qedks_max_followups_per_answer,
            "qedks_use_ppl_schedule": config.qedks_use_ppl_schedule,
        }
