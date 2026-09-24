from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any
import torch

from datasets import Dataset

from attacks.core.checkpoint_health import checkpoint_tensor_health
from .common import read_jsonl


def _missing_parameters(callable_obj: Any, required: set[str]) -> list[str]:
    parameters = inspect.signature(callable_obj).parameters
    return sorted(required - set(parameters))


def validate_soda_runtime_compatibility(
    *,
    DPOConfig: type,
    DPOTrainer: type,
    LoraConfig: type,
    PeftModel: type,
    TrainerCallback: type,
) -> dict[str, Any]:
    import accelerate
    import datasets
    import peft
    import transformers
    import trl

    expected_versions = {
        "transformers": "5.14.1",
        "trl": "1.9.2",
        "peft": "0.15.2",
        "accelerate": "1.14.0",
        "datasets": "5.0.1",
    }
    actual_versions = {
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
    }
    errors = [
        f"{name}=={actual_versions[name]} (expected {expected})"
        for name, expected in expected_versions.items()
        if actual_versions[name] != expected
    ]
    signature_requirements = {
        "DPOConfig": (
            DPOConfig,
            {
                "output_dir", "beta", "learning_rate", "num_train_epochs",
                "per_device_train_batch_size", "gradient_accumulation_steps",
                "max_length", "bf16", "gradient_checkpointing", "max_grad_norm",
                "lr_scheduler_type", "warmup_steps", "save_strategy", "logging_steps",
            },
        ),
        "DPOTrainer.__init__": (
            DPOTrainer.__init__,
            {"model", "ref_model", "args", "train_dataset", "processing_class", "callbacks", "peft_config"},
        ),
        "PeftModel.from_pretrained": (
            PeftModel.from_pretrained,
            {"model", "model_id", "adapter_name", "is_trainable"},
        ),
        "PeftModel.add_adapter": (
            PeftModel.add_adapter,
            {"adapter_name", "peft_config"},
        ),
    }
    for label, (callable_obj, required) in signature_requirements.items():
        missing = _missing_parameters(callable_obj, required)
        if missing:
            errors.append(f"{label} missing parameters: {missing}")
    if not hasattr(TrainerCallback, "on_pre_optimizer_step"):
        errors.append("TrainerCallback missing on_pre_optimizer_step")
    for method_name in ("disable_adapter", "set_adapter"):
        if not hasattr(PeftModel, method_name):
            errors.append(f"PeftModel missing {method_name}")
    trainer_source = inspect.getsource(DPOTrainer.__init__)
    lora_config_fields = set(getattr(LoraConfig, "__dataclass_fields__", {}))
    trl_lora_fields = set(re.findall(r"default_config\.([A-Za-z_][A-Za-z0-9_]*)", trainer_source))
    missing_lora_fields = trl_lora_fields - lora_config_fields
    unsupported_missing_lora_fields = missing_lora_fields - {"target_parameters"}
    if unsupported_missing_lora_fields:
        errors.append(
            "TRL reads unsupported PEFT LoraConfig fields: "
            f"{sorted(unsupported_missing_lora_fields)}"
        )
    if errors:
        raise RuntimeError("SODA runtime compatibility check failed:\n- " + "\n- ".join(errors))
    trl_reads_target_parameters = "target_parameters" in trl_lora_fields
    peft_has_target_parameters = "target_parameters" in lora_config_fields
    report = {
        "versions": actual_versions,
        "required_signatures": sorted(signature_requirements),
        "trainer_callback_on_pre_optimizer_step": True,
        "trl_lora_config_fields": sorted(trl_lora_fields),
        "missing_peft_lora_config_fields": sorted(missing_lora_fields),
        "trl_reads_target_parameters": trl_reads_target_parameters,
        "peft_has_target_parameters": peft_has_target_parameters,
        "requires_target_parameters_compat": trl_reads_target_parameters and not peft_has_target_parameters,
    }
    print(json.dumps({"soda_runtime_compatibility": report}, ensure_ascii=False), flush=True)
    return report


def assert_model_trainable_parameters_finite(model: torch.nn.Module) -> int:
    checked = 0
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        checked += 1
        if not torch.isfinite(parameter.detach()).all().item():
            bad.append(name)
    if bad:
        raise FloatingPointError(f"Non-finite trainable model parameters: {bad[:8]}")
    return checked


def truncate_text(tokenizer: object, text: str, max_tokens: int | None) -> str:
    if max_tokens is None or max_tokens <= 0:
        return text
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = tokenized.get("input_ids", [])
    if len(input_ids) <= max_tokens:
        return text
    return tokenizer.decode(input_ids[:max_tokens], skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SODA DPO from teacher/student preference pairs.")
    parser.add_argument("--preference-jsonl", required=True)
    parser.add_argument("--warmup-model", required=True, help="Seq-KD warmup checkpoint q_w; also used as DPO reference.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=3584)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--nonfinite-gradient-retries",
        type=int,
        default=3,
        help="Retry a micro-batch with a new dropout mask after restoring pre-backward accumulated gradients.",
    )
    args = parser.parse_args()

    try:
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl import DPOConfig, DPOTrainer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "SODA DPO training requires optional packages `trl` and `peft`. "
            "Install them in the training environment before running this script."
        ) from exc
    runtime_compatibility = validate_soda_runtime_compatibility(
        DPOConfig=DPOConfig,
        DPOTrainer=DPOTrainer,
        LoraConfig=LoraConfig,
        PeftModel=PeftModel,
        TrainerCallback=TrainerCallback,
    )

    warmup_path = Path(args.warmup_model).expanduser()
    adapter_config_path = warmup_path / "adapter_config.json"
    adapter_warmup = adapter_config_path.is_file()
    base_model = None
    if adapter_warmup:
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise SystemExit(f"SeqKD adapter has no base_model_name_or_path: {adapter_config_path}")
        if not args.use_lora:
            raise SystemExit("A LoRA SeqKD warmup requires --use-lora for SODA DPO training.")

    tokenizer_source = str(warmup_path) if warmup_path.exists() else args.warmup_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except (OSError, ValueError):
        if not base_model:
            raise
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = Dataset.from_list(
        [
            {
                "prompt": truncate_text(tokenizer, r["prompt"], args.max_prompt_length),
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for r in read_jsonl(args.preference_jsonl)
        ]
    )
    model_dtype = torch.bfloat16 if args.bf16 else "auto"
    if adapter_warmup:
        backbone = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=model_dtype,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            backbone,
            str(warmup_path),
            adapter_name="default",
            is_trainable=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.warmup_model,
            dtype=model_dtype,
            trust_remote_code=True,
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    peft_config = None
    peft_target_parameters_compat = False
    if args.use_lora and not adapter_warmup:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_warmup:
        default_adapter_config = model.peft_config["default"]
        if not hasattr(default_adapter_config, "target_parameters"):
            # TRL 1.9.2 reads this newer optional PEFT field when cloning the
            # warmup adapter as its frozen DPO reference. PEFT 0.15.2 has no
            # parameter-targeting mode, so its equivalent value is None.
            default_adapter_config.target_parameters = None
            peft_target_parameters_compat = True
            print(
                "[SODA compatibility] PEFT LoraConfig has no target_parameters; "
                "using the PEFT 0.15.2-equivalent default None.",
                flush=True,
            )

    if args.nonfinite_gradient_retries < 0:
        raise SystemExit("--nonfinite-gradient-retries must be non-negative")

    class TrainingSafetyCallback(TrainerCallback):
        def __init__(self) -> None:
            self.pre_step_learning_rates: list[float] = []

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, optimizer: Any = None, model: Any = None, **kwargs: Any) -> Any:
            if optimizer is None:
                raise RuntimeError("Trainer did not provide the optimizer to on_pre_optimizer_step")
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
            if not learning_rates or not all(math.isfinite(value) and value > 0.0 for value in learning_rates):
                raise FloatingPointError(f"Invalid optimizer learning rates before step {state.global_step + 1}: {learning_rates}")
            self.pre_step_learning_rates.append(max(learning_rates))
            if model is not None:
                bad = [name for name, parameter in model.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()]
                if bad:
                    raise FloatingPointError(f"Non-finite gradients immediately before optimizer step: {bad[:8]}")
            return control

    class FiniteGradientDPOTrainer(DPOTrainer):
        def __init__(self, *trainer_args: Any, nonfinite_gradient_retries: int, **trainer_kwargs: Any) -> None:
            self.nonfinite_gradient_retries = nonfinite_gradient_retries
            self.nonfinite_backward_attempts = 0
            self.recovered_micro_batches = 0
            super().__init__(*trainer_args, **trainer_kwargs)

        @staticmethod
        def _snapshot_gradients(model: torch.nn.Module) -> list[tuple[torch.nn.Parameter, torch.Tensor | None]]:
            return [
                (parameter, None if parameter.grad is None else parameter.grad.detach().clone())
                for parameter in model.parameters()
                if parameter.requires_grad
            ]

        @staticmethod
        def _restore_gradients(snapshot: list[tuple[torch.nn.Parameter, torch.Tensor | None]]) -> None:
            for parameter, previous in snapshot:
                if previous is None:
                    parameter.grad = None
                elif parameter.grad is None:
                    parameter.grad = previous
                else:
                    parameter.grad.copy_(previous)

        @staticmethod
        def _nonfinite_gradient_names(model: torch.nn.Module) -> list[str]:
            return [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
            ]

        def training_step(self, model: torch.nn.Module, inputs: dict[str, Any], *step_args: Any, **step_kwargs: Any) -> torch.Tensor:
            snapshot = self._snapshot_gradients(model)
            for attempt in range(self.nonfinite_gradient_retries + 1):
                loss = super().training_step(model, inputs, *step_args, **step_kwargs)
                bad = self._nonfinite_gradient_names(model)
                if torch.isfinite(loss.detach()).all().item() and not bad:
                    if attempt:
                        self.recovered_micro_batches += 1
                    return loss
                self.nonfinite_backward_attempts += 1
                self._restore_gradients(snapshot)
                print(
                    f"[SODA safety] non-finite backward attempt {attempt + 1}/"
                    f"{self.nonfinite_gradient_retries + 1}; loss={float(loss.detach().float().item())}; "
                    f"bad_gradients={bad[:4]}",
                    flush=True,
                )
            raise FloatingPointError(
                "SODA produced non-finite gradients after "
                f"{self.nonfinite_gradient_retries + 1} attempts on the same micro-batch"
            )

    safety_callback = TrainingSafetyCallback()
    config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="linear",
        warmup_steps=0,
        save_strategy="epoch",
        logging_steps=10,
    )
    trainer = FiniteGradientDPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[safety_callback],
        nonfinite_gradient_retries=args.nonfinite_gradient_retries,
    )
    train_result = trainer.train()
    if trainer.state.global_step <= 0:
        raise RuntimeError("SODA completed without any optimizer steps")
    if not safety_callback.pre_step_learning_rates:
        raise RuntimeError("SODA recorded no optimizer-step learning rates")
    if max(safety_callback.pre_step_learning_rates) <= 0.0:
        raise RuntimeError("SODA effective learning rate was zero for every optimizer step")
    trainable_parameter_tensors = assert_model_trainable_parameters_finite(trainer.model)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    checkpoint_health = checkpoint_tensor_health(Path(args.output_dir))
    if not checkpoint_health["all_finite"]:
        raise FloatingPointError(
            f"Saved SODA checkpoint contains non-finite tensors: {checkpoint_health['nonfinite_tensors'][:8]}"
        )
    safety_summary = {
        "requested_learning_rate": args.learning_rate,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "optimizer_steps": trainer.state.global_step,
        "pre_step_learning_rates": safety_callback.pre_step_learning_rates,
        "first_pre_step_learning_rate": safety_callback.pre_step_learning_rates[0],
        "last_pre_step_learning_rate": safety_callback.pre_step_learning_rates[-1],
        "max_pre_step_learning_rate": max(safety_callback.pre_step_learning_rates),
        "min_pre_step_learning_rate": min(safety_callback.pre_step_learning_rates),
        "nonfinite_backward_attempts": trainer.nonfinite_backward_attempts,
        "recovered_micro_batches": trainer.recovered_micro_batches,
        "nonfinite_gradient_retries": args.nonfinite_gradient_retries,
        "peft_target_parameters_compat": peft_target_parameters_compat,
        "runtime_compatibility": runtime_compatibility,
        "trainable_parameter_tensors": trainable_parameter_tensors,
        "checkpoint_health": checkpoint_health,
        "train_metrics": train_result.metrics,
    }
    (Path(args.output_dir) / "training_safety.json").write_text(
        json.dumps(safety_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"soda_training_safety": safety_summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


