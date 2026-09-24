from __future__ import annotations

"""Minimal text-only LoRD training loop for stage 1 transcripts."""

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterator, Mapping

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, set_seed

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None
    PeftModel = None
    get_peft_model = None

from .lord_primitives import (
    compute_black_box_pair_loss,
    compute_differentiable_black_box_loss,
    evaluate_lord_snapshot_transition,
    resolve_tau_value,
)
from .seqkd_train import _ensure_tokenizer_chat_defaults, _full_text_without_eos, _prompt_prefix_text
from .stage1_transcript import git_commit, load_transcript_bundle, transcript_hash_from_records


LORD_SCHEMA_VERSION = "lord_stage1_v1"
LORD_CODE_VERSION = "lord_train_stage1_v1"


class LoRDTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoRDTrainingConfig:
    transcript_dir: str
    output_dir: str
    base_student_model_id: str
    requested_budget: int
    seed: int
    periods: int = 1
    learning_rate: float = 3e-5
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 64
    num_return_sequences: int = 2
    tau1: float = 0.8
    tau2: float = 0.1
    tau_delta: float = 0.05
    lambda1: float = 0.5
    clip_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    seqkd_warm_start: bool = False
    seqkd_checkpoint: str | None = None
    resume_from_checkpoint: str | None = None
    generation_backend: str = "transformers"
    bf16: bool = True
    use_lora: bool = True
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.num_return_sequences != 2:
            raise LoRDTrainingError("LoRD requires num_return_sequences=2.")
        if self.periods < 1 or self.requested_budget < 1:
            raise LoRDTrainingError("periods and requested_budget must be positive.")
        if not 0.0 < self.lambda1 < 1.0:
            raise LoRDTrainingError("lambda1 must be in (0, 1).")
        if self.generation_backend not in {"transformers", "vllm"}:
            raise LoRDTrainingError("generation_backend must be 'transformers' or 'vllm'.")
        if self.seqkd_warm_start and not self.seqkd_checkpoint:
            raise LoRDTrainingError("seqkd_checkpoint is required when seqkd_warm_start=True.")
        if self.seqkd_checkpoint and not self.seqkd_warm_start:
            raise LoRDTrainingError("seqkd_checkpoint was supplied but seqkd_warm_start is disabled.")


@dataclass(frozen=True)
class LoRDTrainResult:
    checkpoint_dir: Path
    snapshot_dir: Path
    manifest_path: Path
    log_path: Path
    optimizer_steps: int
    candidate_generation_count: int
    ledger_hash_before: str
    ledger_hash_after: str
    max_gradient_norm: float
    parameter_delta_norm: float


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _model_device(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def _snapshot(model: PreTrainedModel) -> PreTrainedModel:
    result = copy.deepcopy(model).eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


def _capture_trainable_state(model: PreTrainedModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_state(model: PreTrainedModel, state: Mapping[str, torch.Tensor]) -> None:
    named_parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(named_parameters))
    if missing:
        raise LoRDTrainingError(f"snapshot state contains unknown trainable parameters: {missing[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = named_parameters[name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


@contextmanager
def _temporary_trainable_state(
    model: PreTrainedModel,
    snapshot_state: Mapping[str, torch.Tensor],
) -> Iterator[None]:
    current_state = _capture_trainable_state(model)
    _load_trainable_state(model, snapshot_state)
    try:
        yield
    finally:
        _load_trainable_state(model, current_state)


def _adapter_config_path(source: str | Path) -> Path | None:
    path = Path(source)
    config_path = path / "adapter_config.json"
    return config_path if path.is_dir() and config_path.exists() else None


def _load_causal_lm(
    source: str | Path,
    *,
    bf16: bool,
    base_model_id: str | Path | None = None,
) -> PreTrainedModel:
    torch_dtype = torch.bfloat16 if bf16 else "auto"
    adapter_config_path = _adapter_config_path(source)
    if adapter_config_path is not None:
        if PeftModel is None:
            raise LoRDTrainingError("peft is not available but the checkpoint is a PEFT adapter.")
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_source = str(base_model_id or adapter_config.get("base_model_name_or_path") or "").strip()
        if not base_source:
            raise LoRDTrainingError(
                f"PEFT adapter checkpoint {source} does not declare base_model_name_or_path; "
                "set the base student model explicitly."
            )
        print(f"Loading PEFT adapter checkpoint from {source} with base model {base_source}", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_source, torch_dtype=torch_dtype)
        model = PeftModel.from_pretrained(base_model, str(source))
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(str(source), torch_dtype=torch_dtype)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


def _maybe_apply_lora(model: PreTrainedModel, config: LoRDTrainingConfig) -> PreTrainedModel:
    if not config.use_lora:
        return model
    if LoraConfig is None or get_peft_model is None:
        raise LoRDTrainingError("peft is not available but use_lora=True was requested.")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def _maybe_enable_gradient_checkpointing(model: PreTrainedModel, config: LoRDTrainingConfig) -> None:
    if not config.gradient_checkpointing:
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False


def _response_sequence(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    response_text: str,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    prompt_text_rendered = _prompt_prefix_text(tokenizer, prompt_text)
    full_text = _full_text_without_eos(tokenizer, prompt_text, response_text)
    prompt_ids = tokenizer(prompt_text_rendered, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if not full_ids or full_ids[-1] != tokenizer.eos_token_id:
        full_ids.append(tokenizer.eos_token_id)
    if len(full_ids) <= len(prompt_ids):
        full_ids.append(tokenizer.eos_token_id)
    return torch.tensor([full_ids], dtype=torch.long, device=device), len(prompt_ids)


def student_sequence_mean_log_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    response_text: str,
) -> torch.Tensor:
    input_ids, response_start = _response_sequence(tokenizer, prompt_text, response_text, _model_device(model))
    if response_start == 0:
        response_start = 1
    was_training = model.training
    model.eval()
    logits = model(input_ids=input_ids).logits[:, :-1, :]
    if was_training:
        model.train()
    targets = input_ids[:, 1:]
    selected = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    first_prediction = max(response_start - 1, 0)
    response_log_probs = selected[:, first_prediction:]
    if response_log_probs.numel() == 0:
        raise LoRDTrainingError("response has no scoreable tokens.")
    return response_log_probs.mean()


def _generate_two_candidates(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    config: LoRDTrainingConfig,
    generation_seed: int,
) -> list[str]:
    device = _model_device(model)
    rendered = _prompt_prefix_text(tokenizer, prompt_text)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    torch.manual_seed(generation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(generation_seed)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            num_return_sequences=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return [tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip() for row in output]


def _torch_black_box_loss(
    positive_log_prob: torch.Tensor,
    negative_log_prob: torch.Tensor,
    teacher_anchor_log_prob: torch.Tensor,
    *,
    lambda1: float,
    epsilon: float,
) -> torch.Tensor:
    return compute_differentiable_black_box_loss(
        positive_log_prob=positive_log_prob,
        negative_log_prob=negative_log_prob,
        anchor_log_prob=teacher_anchor_log_prob,
        lambda1=lambda1,
        epsilon=epsilon,
    )


def _load_initial_model(
    config: LoRDTrainingConfig,
    model_factory: Callable[[], PreTrainedModel] | None,
) -> tuple[PreTrainedModel, str, int]:
    if config.resume_from_checkpoint:
        checkpoint = Path(config.resume_from_checkpoint)
        state = json.loads((checkpoint / "lord_state.json").read_text(encoding="utf-8"))
        return _load_causal_lm(checkpoint, bf16=config.bf16, base_model_id=config.base_student_model_id), "resume", int(state["completed_periods"])
    if model_factory is not None:
        return model_factory(), "base_student", 0
    source = config.seqkd_checkpoint if config.seqkd_warm_start else config.base_student_model_id
    init_kind = "seqkd_warm_start" if config.seqkd_warm_start else "base_student"
    return _load_causal_lm(str(source), bf16=config.bf16, base_model_id=config.base_student_model_id), init_kind, 0


def train_lord(
    config: LoRDTrainingConfig,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    model_factory: Callable[[], PreTrainedModel] | None = None,
    candidate_generator: Callable[[PreTrainedModel, PreTrainedTokenizerBase, str, LoRDTrainingConfig, int], list[str]] | None = None,
) -> LoRDTrainResult:
    config.validate()
    set_seed(config.seed)
    random.seed(config.seed)
    bundle = load_transcript_bundle(Path(config.transcript_dir), budget=config.requested_budget)
    ledger_path = Path(config.transcript_dir) / "ledger.json"
    ledger_hash_before = _file_hash(ledger_path)

    if tokenizer is None:
        tokenizer_source = config.resume_from_checkpoint or (
            config.seqkd_checkpoint if config.seqkd_warm_start else config.base_student_model_id
        )
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    _ensure_tokenizer_chat_defaults(tokenizer)
    model, initialization_source, completed_periods = _load_initial_model(config, model_factory)
    if model_factory is not None:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = _maybe_apply_lora(model, config)
    print(f"LoRD model device: {_model_device(model)}", flush=True)
    _maybe_enable_gradient_checkpointing(model, config)
    if config.generation_backend == "vllm" and candidate_generator is None:
        raise LoRDTrainingError("generation_backend='vllm' requires an optional candidate_generator callback.")
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.train()

    tau1 = resolve_tau_value(code_var="tau1", actual_trigger_value=config.tau1)
    tau2 = resolve_tau_value(code_var="tau2", actual_trigger_value=config.tau2)
    tau_delta = resolve_tau_value(code_var="tau_delta", actual_trigger_value=config.tau_delta)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "lord_training_log.jsonl"
    logs: list[dict[str, Any]] = []
    total_optimizer_steps = config.periods * len(bundle.records)
    log_interval = max(1, int(os.environ.get("LORD_PROGRESS_LOG_INTERVAL", "10")))
    print(
        f"LoRD training start: periods={config.periods}, records={len(bundle.records)}, "
        f"planned_optimizer_steps={total_optimizer_steps}, log_interval={log_interval}",
        flush=True,
    )
    optimizer_steps = 0
    candidate_generation_count = 0
    max_gradient_norm = 0.0
    started = time.perf_counter()

    snapshot_model = None if config.use_lora else _snapshot(model)
    snapshot_state = _capture_trainable_state(model) if config.use_lora else None
    for period_offset in range(config.periods):
        period = completed_periods + period_offset
        snapshot_id = f"snapshot_{period:06d}"
        for record_index, record in enumerate(bundle.records):
            generation_seed = config.seed + period * len(bundle.records) + record_index
            generate_candidates = candidate_generator or _generate_two_candidates
            if config.use_lora:
                if snapshot_state is None:
                    raise LoRDTrainingError("LoRD LoRA snapshot state was not initialized.")
                with _temporary_trainable_state(model, snapshot_state):
                    candidates = generate_candidates(model, tokenizer, record.prompt_text, config, generation_seed)
                    with torch.no_grad():
                        previous_log_probs = [
                            student_sequence_mean_log_prob(model, tokenizer, record.prompt_text, text)
                            for text in candidates
                        ]
            else:
                if snapshot_model is None:
                    raise LoRDTrainingError("LoRD snapshot model was not initialized.")
                candidates = generate_candidates(snapshot_model, tokenizer, record.prompt_text, config, generation_seed)
                with torch.no_grad():
                    previous_log_probs = [
                        student_sequence_mean_log_prob(snapshot_model, tokenizer, record.prompt_text, text)
                        for text in candidates
                    ]
            if len(candidates) != 2:
                raise LoRDTrainingError("student generation did not return exactly two candidates.")
            candidate_generation_count += 2
            model.train()
            current_log_probs = [
                student_sequence_mean_log_prob(model, tokenizer, record.prompt_text, text) for text in candidates
            ]
            previous_scores = [float(value.exp().item()) for value in previous_log_probs]
            current_scores = [float(value.detach().exp().item()) for value in current_log_probs]
            decision = evaluate_lord_snapshot_transition(
                previous_scores,
                current_scores,
                tau1=tau1,
                tau2=tau2,
                tau_delta=tau_delta,
            )
            anchor_log_prob = student_sequence_mean_log_prob(
                model, tokenizer, record.prompt_text, record.teacher_response
            )
            positive_log_prob = current_log_probs[decision.positive_index]
            if decision.teacher_anchor or period == 0:
                positive_log_prob = anchor_log_prob
            negative_log_prob = current_log_probs[decision.negative_index]
            loss = _torch_black_box_loss(
                positive_log_prob,
                negative_log_prob,
                anchor_log_prob,
                lambda1=config.lambda1,
                epsilon=config.clip_epsilon,
            )
            if not torch.isfinite(loss):
                raise LoRDTrainingError("LoRD loss is NaN or Inf.")
            primitive_loss = compute_black_box_pair_loss(
                positive_score=current_scores[decision.positive_index],
                negative_score=current_scores[decision.negative_index],
                anchor_score=float(anchor_log_prob.detach().exp().item()),
                epsilon=config.clip_epsilon,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite_gradients = [
                parameter.grad.detach() for parameter in model.parameters() if parameter.grad is not None
            ]
            if not finite_gradients or not all(torch.isfinite(gradient).all() for gradient in finite_gradients):
                raise LoRDTrainingError("LoRD gradients are missing or non-finite.")
            gradient_norm = float(torch.sqrt(sum(gradient.float().pow(2).sum() for gradient in finite_gradients)).item())
            if gradient_norm <= 0.0:
                raise LoRDTrainingError("LoRD gradient norm is zero.")
            max_gradient_norm = max(max_gradient_norm, gradient_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer_steps += 1
            if optimizer_steps == 1 or optimizer_steps % log_interval == 0 or optimizer_steps == total_optimizer_steps:
                print(
                    f"LoRD progress: step={optimizer_steps}/{total_optimizer_steps}, "
                    f"period={period}, record={record_index + 1}/{len(bundle.records)}, "
                    f"loss={float(loss.detach().item()):.6g}, grad_norm={gradient_norm:.6g}",
                    flush=True,
                )
            for candidate_id, candidate_text in enumerate(candidates):
                logs.append({
                    "period": period,
                    "prompt_id": record.prompt_id,
                    "snapshot_id": snapshot_id,
                    "candidate_id": f"{snapshot_id}:{record.prompt_id}:{candidate_id}",
                    "candidate_text": candidate_text,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "max_new_tokens": config.max_new_tokens,
                    "num_return_sequences": 2,
                    "seed": generation_seed,
                    "previous_score": previous_scores[candidate_id],
                    "current_score": current_scores[candidate_id],
                    "delta": decision.deltas[candidate_id],
                    "positive": candidate_id == decision.positive_index,
                    "teacher_anchor": bool(decision.teacher_anchor or period == 0),
                    "loss": float(loss.detach().item()),
                    "primitive_loss_audit": float(primitive_loss),
                    "gradient_norm": gradient_norm,
                })
        if config.use_lora:
            snapshot_state = _capture_trainable_state(model)
        else:
            snapshot_model = _snapshot(model)

    log_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in logs), encoding="utf-8")
    checkpoint_dir = output_dir / "checkpoint-final"
    snapshot_dir = output_dir / "snapshot-current"
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    if config.use_lora:
        if snapshot_state is None:
            raise LoRDTrainingError("LoRD LoRA snapshot state was not initialized.")
        with _temporary_trainable_state(model, snapshot_state):
            model.save_pretrained(str(snapshot_dir))
    else:
        if snapshot_model is None:
            raise LoRDTrainingError("LoRD snapshot model was not initialized.")
        snapshot_model.save_pretrained(str(snapshot_dir))
    tokenizer.save_pretrained(str(snapshot_dir))
    total_completed_periods = completed_periods + config.periods
    _save_json(checkpoint_dir / "lord_state.json", {
        "completed_periods": total_completed_periods,
        "snapshot_id": f"snapshot_{total_completed_periods:06d}",
        "snapshot_dir": str(snapshot_dir),
    })
    parameter_delta_norm = float(torch.sqrt(sum(
        (parameter.detach() - initial).float().pow(2).sum()
        for parameter, initial in zip((p for p in model.parameters() if p.requires_grad), initial_parameters)
    )).item())
    if parameter_delta_norm <= 0.0:
        raise LoRDTrainingError("optimizer steps did not change model parameters.")
    ledger_hash_after = _file_hash(ledger_path)
    if ledger_hash_before != ledger_hash_after:
        raise LoRDTrainingError("query ledger changed during LoRD training.")
    manifest = {
        "schema_version": LORD_SCHEMA_VERSION,
        "code_version": LORD_CODE_VERSION,
        "git_commit": git_commit(),
        "checkpoint_dir": str(checkpoint_dir),
        "snapshot_dir": str(snapshot_dir),
        "initialization_source": initialization_source,
        "base_student_model_id": config.base_student_model_id,
        "snapshot_mode": "trainable_state_cpu" if config.use_lora else "deepcopy_model",
        "seqkd_warm_start": config.seqkd_warm_start,
        "transcript_hash": bundle.manifest.transcript_hash,
        "transcript_prefix_hash": transcript_hash_from_records(bundle.records),
        "ordering_hash": bundle.manifest.prompt_ordering_hash,
        "teacher_access": "text_only_transcript",
        "teacher_logits_read": False,
        "optimizer_steps": optimizer_steps,
        "candidate_generation_count": candidate_generation_count,
        "period_count": config.periods,
        "sub_stage_count": optimizer_steps,
        "completed_periods": total_completed_periods,
        "ledger_hash_before": ledger_hash_before,
        "ledger_hash_after": ledger_hash_after,
        "max_gradient_norm": max_gradient_norm,
        "parameter_delta_norm": parameter_delta_norm,
        "wall_clock_seconds": time.perf_counter() - started,
        "config_source": "lord_training_config.json",
    }
    manifest_path = output_dir / "lord_manifest.json"
    _save_json(manifest_path, manifest)
    _save_json(output_dir / "lord_training_config.json", config.to_dict())
    return LoRDTrainResult(
        checkpoint_dir=checkpoint_dir,
        snapshot_dir=snapshot_dir,
        manifest_path=manifest_path,
        log_path=log_path,
        optimizer_steps=optimizer_steps,
        candidate_generation_count=candidate_generation_count,
        ledger_hash_before=ledger_hash_before,
        ledger_hash_after=ledger_hash_after,
        max_gradient_norm=max_gradient_norm,
        parameter_delta_norm=parameter_delta_norm,
    )


def load_trained_lord_model(checkpoint_dir: str | Path) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return (
        model,
        AutoTokenizer.from_pretrained(str(checkpoint_dir)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_student_model_id", required=True)
    parser.add_argument("--requested_budget", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--periods", default=1, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--tau1", default=0.8, type=float)
    parser.add_argument("--tau2", default=0.1, type=float)
    parser.add_argument("--tau_delta", default=0.05, type=float)
    parser.add_argument("--lambda1", default=0.5, type=float)
    parser.add_argument("--seqkd_warm_start", action="store_true")
    parser.add_argument("--seqkd_checkpoint")
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    args = parser.parse_args()
    train_lord(LoRDTrainingConfig(**vars(args)))


if __name__ == "__main__":
    main()
