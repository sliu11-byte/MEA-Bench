from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.core.base import AttackRunConfig
from attacks.core.hf_query_pool import DEFAULT_HF_QUERY_POOL, hf_query_pool_for_budget, resolve_query_pool_and_ordering
from attacks.core.pipeline import AttackPipeline


DEFAULT_STAGE1_CONFIG = REPO_ROOT / "attacks" / "configs" / "stage1_budget.yaml"
DEFAULT_QUERY_POOL = DEFAULT_HF_QUERY_POOL
DEFAULT_QUERY_ORDERING = "auto"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "attacks"


def _optional_path(value: str | None) -> Path | None:
    return None if value is None else Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a unified stage-1 attacker pipeline.")
    parser.add_argument("--attack", default="seqkd", choices=["seqkd", "lord", "soda", "qedks", "model_leeching", "gad"])
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--query-pool", default=None, help="JSON query pool or hf:// spec. Defaults to the published HF tier matching --budget.")
    parser.add_argument("--query-ordering", default=str(DEFAULT_QUERY_ORDERING))
    parser.add_argument("--stage1-config", default=str(DEFAULT_STAGE1_CONFIG))
    parser.add_argument("--transcript-dir")
    parser.add_argument("--teacher-transcript-path", help="External JSONL with query/response teacher outputs, such as defenses/*/attack_input/training_transcript.jsonl.")
    parser.add_argument("--countermeasure", default=None, help="Optional countermeasure label when attack training uses countermeasure-processed teacher responses.")
    parser.add_argument("--countermeasure-manifest", default=None, help="Optional countermeasure_manifest.json associated with the countermeasure run.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execution-stage", choices=["all", "prepare", "train"], default="all")
    parser.add_argument("--prepared-run-dir", help="Run directory produced by an earlier --execution-stage prepare invocation.")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--teacher-backend", choices=["local_hf", "vllm_openai"])
    parser.add_argument("--teacher-model", help="Teacher model id or local path.")
    parser.add_argument("--teacher-endpoint-url", help="OpenAI-compatible teacher endpoint base URL, for endpoint-backed attacks.")
    parser.add_argument("--teacher-request-model", help="Teacher model name sent to the endpoint; defaults to --teacher-model when omitted.")
    parser.add_argument("--teacher-api-key")
    parser.add_argument("--teacher-mode", default="chat", choices=["chat", "completion"])
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-top-p", type=float, default=1.0)
    parser.add_argument("--teacher-max-tokens", type=int, default=512)

    parser.add_argument("--student-model", help="Student/base/warmup model id, local path, or checkpoint path.")
    parser.add_argument("--student-endpoint-url", help="OpenAI-compatible student endpoint base URL, for attacks that query the student.")
    parser.add_argument("--student-request-model", help="Student model name sent to the endpoint; defaults to --student-model when omitted.")
    parser.add_argument("--student-api-key", default="EMPTY")
    parser.add_argument("--student-mode", default="chat", choices=["chat", "completion"])
    parser.add_argument("--student-temperature", type=float, default=0.7)
    parser.add_argument("--student-top-p", type=float, default=1.0)
    parser.add_argument("--student-max-tokens", type=int, default=1536)

    parser.add_argument("--warmup-model", help="Explicit warmup/reference checkpoint. SODA auto-discovers a compatible completed SeqKD run when omitted.")
    parser.add_argument("--soda-beta", type=float, default=0.1)
    parser.add_argument("--soda-learning-rate", type=float, default=5e-6)
    parser.add_argument("--soda-epochs", type=float, default=1.0)
    parser.add_argument("--soda-per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--soda-gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--soda-max-length", type=int, default=3584)
    parser.add_argument("--soda-max-prompt-length", type=int, default=1024)
    parser.add_argument("--soda-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--soda-nonfinite-gradient-retries", type=int, default=3)
    parser.add_argument("--soda-student-negatives-jsonl", help="Existing SODA student negatives JSONL to reuse instead of querying the student endpoint.")
    parser.add_argument("--soda-preferences-jsonl", help="Existing SODA preferences JSONL to reuse instead of rebuilding preference pairs.")

    parser.add_argument("--qedks-max-followups-per-answer", type=int, default=4)
    parser.add_argument("--qedks-use-ppl-schedule", action="store_true")
    parser.add_argument("--qedks-ppl-device", default="cuda")
    parser.add_argument("--qedks-learning-rate", type=float, default=2e-4)
    parser.add_argument("--qedks-epochs", type=float, default=2.0)
    parser.add_argument("--qedks-lora-r", type=int, default=16)
    parser.add_argument("--qedks-lora-alpha", type=int, default=32)
    parser.add_argument("--qedks-lora-dropout", type=float, default=0.05)
    parser.add_argument("--qedks-per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--qedks-gradient-accumulation-steps", type=int, default=16)

    parser.add_argument("--model-leeching-learning-rate", type=float, default=2e-4)
    parser.add_argument("--model-leeching-epochs", type=float, default=2.0)
    parser.add_argument("--model-leeching-lora-r", type=int, default=16)
    parser.add_argument("--model-leeching-lora-alpha", type=int, default=32)
    parser.add_argument("--model-leeching-lora-dropout", type=float, default=0.05)
    parser.add_argument("--model-leeching-per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--model-leeching-gradient-accumulation-steps", type=int, default=16)

    parser.add_argument("--gad-group-size", type=int, default=8)
    parser.add_argument("--gad-kl-beta", type=float, default=0.001)
    parser.add_argument("--gad-learning-rate", type=float, default=1e-6)
    parser.add_argument("--gad-discriminator-learning-rate", type=float, default=1e-6)
    parser.add_argument("--gad-warmup-learning-rate", type=float, default=5e-6)
    parser.add_argument("--gad-warmup-epochs", type=float, default=1.0)
    parser.add_argument("--gad-epochs", type=float, default=2.0)
    parser.add_argument("--gad-discriminator-warmup-steps", type=int, default=10)
    parser.add_argument("--gad-max-steps", type=int, default=-1)
    parser.add_argument("--gad-per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gad-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gad-max-seq-length", type=int, default=3584)
    parser.add_argument("--gad-max-prompt-length", type=int, default=2048)
    parser.add_argument("--gad-max-response-length", type=int, default=1536)
    parser.add_argument("--gad-temperature", type=float, default=0.8)
    parser.add_argument("--gad-top-p", type=float, default=1.0)
    parser.add_argument("--gad-offload-inactive-models", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gad-memory-log-interval", type=int, default=25)
    parser.add_argument("--gad-generator-device", help="Device for GAD generator/student G, e.g. cuda:0.")
    parser.add_argument("--gad-discriminator-device", help="Device for GAD discriminator D, e.g. cuda:1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_pool_spec = hf_query_pool_for_budget(args.budget) if args.query_pool in (None, "auto") else args.query_pool
    query_pool_path, query_ordering_path = resolve_query_pool_and_ordering(query_pool_spec, args.query_ordering)

    config = AttackRunConfig(
        attack=args.attack,
        budget=args.budget,
        query_pool_path=query_pool_path,
        query_ordering_path=query_ordering_path,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        stage1_config_path=Path(args.stage1_config).expanduser().resolve(),
        transcript_dir=_optional_path(args.transcript_dir),
        teacher_transcript_path=_optional_path(args.teacher_transcript_path),
        teacher_backend=args.teacher_backend,
        teacher_model=args.teacher_model,
        teacher_endpoint_url=args.teacher_endpoint_url,
        teacher_request_model=args.teacher_request_model,
        teacher_api_key=args.teacher_api_key,
        teacher_mode=args.teacher_mode,
        teacher_temperature=args.teacher_temperature,
        teacher_top_p=args.teacher_top_p,
        teacher_max_tokens=args.teacher_max_tokens,
        student_model=args.student_model,
        student_endpoint_url=args.student_endpoint_url,
        student_request_model=args.student_request_model,
        student_api_key=args.student_api_key,
        student_mode=args.student_mode,
        student_temperature=args.student_temperature,
        student_top_p=args.student_top_p,
        student_max_tokens=args.student_max_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
        execution_stage=args.execution_stage,
        prepared_run_dir=_optional_path(args.prepared_run_dir),
        bf16=args.bf16,
        use_lora=args.use_lora,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        warmup_model=args.warmup_model,
        soda_beta=args.soda_beta,
        soda_learning_rate=args.soda_learning_rate,
        soda_epochs=args.soda_epochs,
        soda_per_device_train_batch_size=args.soda_per_device_train_batch_size,
        soda_gradient_accumulation_steps=args.soda_gradient_accumulation_steps,
        soda_max_length=args.soda_max_length,
        soda_max_prompt_length=args.soda_max_prompt_length,
        soda_max_grad_norm=args.soda_max_grad_norm,
        soda_nonfinite_gradient_retries=args.soda_nonfinite_gradient_retries,
        soda_student_negatives_jsonl=_optional_path(args.soda_student_negatives_jsonl),
        soda_preferences_jsonl=_optional_path(args.soda_preferences_jsonl),
        qedks_max_followups_per_answer=args.qedks_max_followups_per_answer,
        qedks_use_ppl_schedule=args.qedks_use_ppl_schedule,
        qedks_ppl_device=args.qedks_ppl_device,
        qedks_learning_rate=args.qedks_learning_rate,
        qedks_epochs=args.qedks_epochs,
        qedks_lora_r=args.qedks_lora_r,
        qedks_lora_alpha=args.qedks_lora_alpha,
        qedks_lora_dropout=args.qedks_lora_dropout,
        qedks_per_device_train_batch_size=args.qedks_per_device_train_batch_size,
        qedks_gradient_accumulation_steps=args.qedks_gradient_accumulation_steps,
        model_leeching_learning_rate=args.model_leeching_learning_rate,
        model_leeching_epochs=args.model_leeching_epochs,
        model_leeching_lora_r=args.model_leeching_lora_r,
        model_leeching_lora_alpha=args.model_leeching_lora_alpha,
        model_leeching_lora_dropout=args.model_leeching_lora_dropout,
        model_leeching_per_device_train_batch_size=args.model_leeching_per_device_train_batch_size,
        model_leeching_gradient_accumulation_steps=args.model_leeching_gradient_accumulation_steps,
        gad_group_size=args.gad_group_size,
        gad_kl_beta=args.gad_kl_beta,
        gad_learning_rate=args.gad_learning_rate,
        gad_discriminator_learning_rate=args.gad_discriminator_learning_rate,
        gad_warmup_learning_rate=args.gad_warmup_learning_rate,
        gad_warmup_epochs=args.gad_warmup_epochs,
        gad_epochs=args.gad_epochs,
        gad_discriminator_warmup_steps=args.gad_discriminator_warmup_steps,
        gad_max_steps=args.gad_max_steps,
        gad_per_device_train_batch_size=args.gad_per_device_train_batch_size,
        gad_gradient_accumulation_steps=args.gad_gradient_accumulation_steps,
        gad_max_seq_length=args.gad_max_seq_length,
        gad_max_prompt_length=args.gad_max_prompt_length,
        gad_max_response_length=args.gad_max_response_length,
        gad_temperature=args.gad_temperature,
        gad_top_p=args.gad_top_p,
        gad_offload_inactive_models=args.gad_offload_inactive_models,
        gad_memory_log_interval=args.gad_memory_log_interval,
        gad_generator_device=args.gad_generator_device,
        gad_discriminator_device=args.gad_discriminator_device,
        extra={
            "countermeasure": args.countermeasure,
            "countermeasure_manifest": None if args.countermeasure_manifest is None else str(_optional_path(args.countermeasure_manifest)),
            "countermeasure_enabled": args.countermeasure is not None,
        },
    )
    result = AttackPipeline().run(config)
    payload: dict[str, Any] = result.to_manifest()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
