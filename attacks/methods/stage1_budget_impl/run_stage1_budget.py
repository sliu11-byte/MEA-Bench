from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import torch
import yaml

from .lord_train_stage1 import LoRDTrainingConfig, train_lord
from .seqkd_train import SeqKDTrainingConfig, train_seqkd
from .stage1_evaluation import build_unified_result, evaluate_m6_cost, save_unified_result, unavailable
from .stage1_transcript import load_transcript_bundle, stable_hash


ALLOWED_BUDGETS = {100, 500, 1000, 10000, 50000, 200000}
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class Stage1BudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stage1BudgetConfig:
    attack: str
    budget: int
    config_path: str
    transcript_dir: str
    output_dir: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage1BudgetError(f"config file must contain a mapping: {path}")
    return dict(_expand_env(payload))


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2)
            resolved = os.environ.get(name)
            if resolved is not None:
                return resolved
            if default is not None:
                return default
            raise Stage1BudgetError(f"missing environment variable: {name}")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    raise Stage1BudgetError(f"cannot parse boolean value: {value!r}")


def _file_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _budget_run_dir(output_root: Path, attack: str, budget: int) -> Path:
    return output_root / attack / f"budget_{budget}"


def _merge_training_config(config: Mapping[str, Any], budget: int, attack: str, transcript_dir: Path, output_dir: Path) -> dict[str, Any]:
    merged = dict(config)
    merged["attack"] = attack
    merged["budget"] = budget
    merged["transcript_dir"] = str(transcript_dir)
    merged["output_dir"] = str(output_dir)
    return merged


def _training_manifest_path(run_dir: Path, attack: str) -> Path:
    if attack == "seqkd":
        return run_dir / "seqkd" / "seqkd_training_manifest.json"
    return run_dir / "lord" / "lord_manifest.json"


def _seqkd_attack_root() -> Path:
    storage_root = os.environ.get("STORAGE_ROOT")
    if storage_root:
        return Path(storage_root) / "outputs" / "attacks_full" / "seqkd"

    output_root = os.environ.get("OUTPUT_DIR")
    if output_root:
        return Path(output_root) / "seqkd"

    return Path("outputs") / "attacks_full" / "seqkd"


def _manifest_checkpoint(manifest_path: Path, *, budget: int) -> Path | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("schema_version") != "attack_manifest_v1":
        return None
    result = manifest.get("result", {})
    if result.get("status") != "completed":
        return None
    if result.get("attack") != "seqkd" and manifest.get("run_config", {}).get("attack") != "seqkd":
        return None
    if int(result.get("budget", manifest.get("run_config", {}).get("budget", -1))) != budget:
        return None
    checkpoint = result.get("checkpoint_dir")
    if not checkpoint:
        return None
    checkpoint_path = Path(str(checkpoint))
    return checkpoint_path if checkpoint_path.exists() else None


def _resolve_latest_seqkd_checkpoint(*, budget: int) -> Path | None:
    seqkd_root = _seqkd_attack_root()
    if not seqkd_root.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for manifest_path in seqkd_root.glob("*/attack_manifest.json"):
        checkpoint = _manifest_checkpoint(manifest_path, budget=budget)
        if checkpoint is not None:
            candidates.append((manifest_path.stat().st_mtime, checkpoint))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _load_transcript_identity(transcript_dir: Path) -> dict[str, Any]:
    bundle = load_transcript_bundle(transcript_dir)
    ledger_path = transcript_dir / "ledger.json"
    ordering_path = transcript_dir / "ordering.json"
    return {
        "prompt_pool_id": bundle.manifest.prompt_pool_id,
        "prompt_pool_hash": bundle.manifest.prompt_pool_hash,
        "ordering_hash": bundle.manifest.prompt_ordering_hash,
        "transcript_hash": bundle.manifest.transcript_hash,
        "ledger_hash": _file_hash(ledger_path),
        "successful_queries": bundle.ledger.successful_queries,
        "api_attempts": bundle.ledger.api_attempts,
        "failed_attempts": bundle.ledger.failed_attempts,
        "ordering_path": str(ordering_path),
        "manifest_path": str(transcript_dir / "manifest.json"),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(dict(payload)), encoding="utf-8")
    return path


def _run_seqkd(
    *,
    config: Mapping[str, Any],
    budget: int,
    transcript_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    seqkd_cfg = dict(config.get("seqkd_training", {}))
    seqkd_output = output_dir / "seqkd"
    seqkd_result = train_seqkd(
        SeqKDTrainingConfig(
            transcript_dir=str(transcript_dir),
            output_dir=str(seqkd_output),
            student_model_id=str(config["student_model_path"]),
            requested_budget=budget,
            seed=int(config["seed"]),
            max_seq_length=int(seqkd_cfg.get("max_seq_length", 2048)),
            learning_rate=float(seqkd_cfg.get("learning_rate", 2e-5)),
            num_train_epochs=float(seqkd_cfg.get("num_train_epochs", 1.0)),
            max_steps=int(seqkd_cfg.get("max_steps", -1)),
            per_device_train_batch_size=int(seqkd_cfg.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(seqkd_cfg.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(seqkd_cfg.get("gradient_accumulation_steps", 1)),
            optim=str(seqkd_cfg.get("optim", "adamw_torch")),
            lr_scheduler_type=str(seqkd_cfg.get("lr_scheduler_type", "linear")),
            warmup_ratio=float(seqkd_cfg.get("warmup_ratio", 0.0)),
            fp16=_as_bool(seqkd_cfg.get("fp16"), False),
            bf16=_as_bool(seqkd_cfg.get("bf16"), False),
            weight_decay=float(seqkd_cfg.get("weight_decay", 0.0)),
            max_grad_norm=float(seqkd_cfg.get("max_grad_norm", 1.0)),
            gradient_checkpointing=_as_bool(seqkd_cfg.get("gradient_checkpointing"), False),
            use_lora=_as_bool(seqkd_cfg.get("use_lora"), False),
            lora_r=int(seqkd_cfg.get("lora_r", 8)),
            lora_alpha=int(seqkd_cfg.get("lora_alpha", 16)),
            lora_dropout=float(seqkd_cfg.get("lora_dropout", 0.0)),
            allow_sampling_transcript=_as_bool(seqkd_cfg.get("allow_sampling_transcript"), False),
            include_eos_in_loss=_as_bool(seqkd_cfg.get("include_eos_in_loss"), True),
            save_total_limit=int(seqkd_cfg.get("save_total_limit", 1)),
            logging_steps=int(seqkd_cfg.get("logging_steps", 1)),
            save_strategy=str(seqkd_cfg.get("save_strategy", "no")),
            report_to=tuple(seqkd_cfg.get("report_to", ())),
        )
    )
    checkpoint_dir = seqkd_result.checkpoint_dir
    return {
        "checkpoint": str(checkpoint_dir),
        "manifest_path": str(seqkd_result.manifest_path),
        "training_manifest": json.loads(seqkd_result.manifest_path.read_text(encoding="utf-8")),
        "dataset_stats": seqkd_result.dataset_stats.to_dict(),
        "output_dir": str(seqkd_output),
        "training_config": _merge_training_config(seqkd_cfg, budget, "seqkd", transcript_dir, seqkd_output),
    }


def _run_lord(
    *,
    config: Mapping[str, Any],
    budget: int,
    transcript_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    lord_cfg = dict(config.get("lord_training", {}))
    seqkd_checkpoint = lord_cfg.get("seqkd_checkpoint") or None
    if _as_bool(lord_cfg.get("seqkd_warm_start"), False) and seqkd_checkpoint is None:
        resolved_checkpoint = _resolve_latest_seqkd_checkpoint(budget=budget)
        if resolved_checkpoint is None:
            raise Stage1BudgetError(
                "LoRD seqkd_warm_start=True but no seqkd_checkpoint was supplied, "
                f"and no completed SeqKD checkpoint for budget={budget} was found under {_seqkd_attack_root()}. "
                "Run SeqKD for this budget first, set LORD_SEQKD_CHECKPOINT, or unset LORD_SEQKD_WARM_START."
            )
        seqkd_checkpoint = str(resolved_checkpoint)
        lord_cfg["seqkd_checkpoint"] = seqkd_checkpoint
        print(f"LoRD SeqKD warm start checkpoint auto-selected: {seqkd_checkpoint}", flush=True)
    elif _as_bool(lord_cfg.get("seqkd_warm_start"), False):
        print(f"LoRD SeqKD warm start checkpoint provided: {seqkd_checkpoint}", flush=True)
    lord_output = output_dir / "lord"
    lord_result = train_lord(
        LoRDTrainingConfig(
            transcript_dir=str(transcript_dir),
            output_dir=str(lord_output),
            base_student_model_id=str(config["student_model_path"]),
            requested_budget=budget,
            seed=int(config["seed"]),
            periods=int(lord_cfg.get("periods", 1)),
            learning_rate=float(lord_cfg.get("learning_rate", 3e-5)),
            temperature=float(lord_cfg.get("temperature", 1.0)),
            top_p=float(lord_cfg.get("top_p", 0.95)),
            max_new_tokens=int(lord_cfg.get("max_new_tokens", 64)),
            num_return_sequences=int(lord_cfg.get("num_return_sequences", 2)),
            tau1=float(lord_cfg.get("tau1", 0.8)),
            tau2=float(lord_cfg.get("tau2", 0.1)),
            tau_delta=float(lord_cfg.get("tau_delta", 0.05)),
            lambda1=float(lord_cfg.get("lambda1", 0.5)),
            clip_epsilon=float(lord_cfg.get("clip_epsilon", 0.2)),
            max_grad_norm=float(lord_cfg.get("max_grad_norm", 1.0)),
            seqkd_warm_start=_as_bool(lord_cfg.get("seqkd_warm_start"), False),
            seqkd_checkpoint=seqkd_checkpoint,
            resume_from_checkpoint=lord_cfg.get("resume_from_checkpoint"),
            generation_backend=str(lord_cfg.get("generation_backend", "transformers")),
            bf16=_as_bool(lord_cfg.get("bf16"), True),
            use_lora=_as_bool(lord_cfg.get("use_lora"), True),
            gradient_checkpointing=_as_bool(lord_cfg.get("gradient_checkpointing"), True),
            lora_r=int(lord_cfg.get("lora_r", 16)),
            lora_alpha=int(lord_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lord_cfg.get("lora_dropout", 0.05)),
        )
    )
    return {
        "checkpoint": str(lord_result.checkpoint_dir),
        "manifest_path": str(lord_result.manifest_path),
        "training_manifest": json.loads(lord_result.manifest_path.read_text(encoding="utf-8")),
        "output_dir": str(lord_output),
        "training_config": _merge_training_config(lord_cfg, budget, "lord", transcript_dir, lord_output),
    }


def run_stage1_budget(
    *,
    attack: str,
    budget: int,
    config_path: Path,
    transcript_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if attack not in {"seqkd", "lord"}:
        raise Stage1BudgetError("attack must be seqkd or lord.")
    if budget not in ALLOWED_BUDGETS:
        raise Stage1BudgetError(f"budget must be one of {sorted(ALLOWED_BUDGETS)}.")

    config = _load_yaml(config_path)
    if int(config.get("seed", 0)) < 0:
        raise Stage1BudgetError("seed must be non-negative.")
    output_dir = _budget_run_dir(output_dir, attack, budget)
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_identity = _load_transcript_identity(transcript_dir)

    started = datetime.now(timezone.utc)
    if attack == "seqkd":
        training = _run_seqkd(config=config, budget=budget, transcript_dir=transcript_dir, output_dir=output_dir)
    else:
        training = _run_lord(config=config, budget=budget, transcript_dir=transcript_dir, output_dir=output_dir)
    wall_clock_seconds = max((datetime.now(timezone.utc) - started).total_seconds(), 0.0)

    training_manifest = training["training_manifest"]
    checkpoint_dir = Path(training["checkpoint"])
    if attack == "seqkd":
        optimizer_steps = int(training_manifest.get("max_steps", 0))
        if optimizer_steps <= 0:
            optimizer_steps = int(training_manifest.get("valid_train_samples", 0))
    else:
        optimizer_steps = int(training_manifest.get("optimizer_steps", 0))
    m6 = evaluate_m6_cost(
        teacher_successful_queries=int(transcript_identity["successful_queries"]),
        teacher_query_attempts=int(transcript_identity["api_attempts"]),
        student_generation_tokens=0,
        wall_clock_seconds=wall_clock_seconds,
        checkpoint_path=checkpoint_dir,
        optimizer_steps=optimizer_steps,
        lord_candidate_generation_count=int(training_manifest.get("candidate_generation_count", 0)),
        lord_period_count=int(training_manifest.get("period_count", 0)),
        lord_sub_stage_count=int(training_manifest.get("sub_stage_count", 0)),
    )
    result_payload = build_unified_result(
        attack=attack,
        transcript_hash=str(transcript_identity["transcript_hash"]),
        ordering_hash=str(transcript_identity["ordering_hash"]),
        checkpoint=checkpoint_dir,
        config_source=str(config_path),
        training_decode_config=dict(config.get("teacher_decode", {})),
        evaluation_decode_config=dict(config.get("m1_evaluation", {}).get("evaluation_decode", {})),
        m1=unavailable("OpenLLM6 eval has not been merged yet"),
        m2=unavailable("OpenLLM6 eval entry does not compute M2"),
        m3=unavailable("OpenLLM6 eval entry does not compute M3"),
        m6=m6,
    )

    training_config_path = _write_json(output_dir / "training_config.json", training["training_config"])
    budget_manifest = {
        "schema_version": "stage1_budget_manifest_v1",
        "created_at": _utc_now_iso(),
        "attack": attack,
        "budget": budget,
        "config_path": str(config_path),
        "transcript_dir": str(transcript_dir),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "transcript_hash": transcript_identity["transcript_hash"],
        "ordering_hash": transcript_identity["ordering_hash"],
        "ledger_hash": transcript_identity["ledger_hash"],
        "training_config_path": str(training_config_path),
        "training_manifest_path": training["manifest_path"],
        "result_json_path": str(output_dir / "stage1_results_v1.json"),
        "training_manifest": training_manifest,
    }
    budget_manifest_path = _write_json(output_dir / "stage1_budget_manifest.json", budget_manifest)
    result_payload["budget_manifest_path"] = str(budget_manifest_path)
    result_path = save_unified_result(output_dir / "stage1_results_v1.json", result_payload)
    return {
        "attack": attack,
        "budget": budget,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_dir),
        "training_manifest_path": training["manifest_path"],
        "budget_manifest_path": str(budget_manifest_path),
        "result_json_path": str(result_path),
        "transcript_hash": transcript_identity["transcript_hash"],
        "ordering_hash": transcript_identity["ordering_hash"],
        "ledger_hash": transcript_identity["ledger_hash"],
        "wall_clock_seconds": wall_clock_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True, choices=["seqkd", "lord"])
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--transcript-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_stage1_budget(
        attack=args.attack,
        budget=args.budget,
        config_path=Path(args.config),
        transcript_dir=Path(args.transcript_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
