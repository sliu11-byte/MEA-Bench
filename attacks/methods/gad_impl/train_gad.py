from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
except Exception:  # pragma: no cover - optional dependency
    AutoPeftModelForCausalLM = None
    LoraConfig = None
    get_peft_model = None

from attacks.methods.stage1_budget_impl.seqkd_train import _full_text_without_eos, _prompt_prefix_text
from attacks.methods.stage1_budget_impl.stage1_transcript import load_transcript_bundle


GAD_SCHEMA_VERSION = "gad_training_v1"


class GADError(RuntimeError):
    pass


@dataclass(frozen=True)
class GADTrainingConfig:
    transcript_dir: str
    output_dir: str
    generator_model_id: str
    discriminator_model_id: str
    requested_budget: int
    seed: int
    group_size: int = 8
    kl_beta: float = 0.001
    learning_rate: float = 1e-6
    discriminator_learning_rate: float = 1e-6
    gad_epochs: float = 2.0
    discriminator_warmup_steps: int = 10
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 3584
    max_prompt_length: int = 2048
    max_response_length: int = 1536
    temperature: float = 0.8
    top_p: float = 1.0
    bf16: bool = True
    use_lora: bool = True
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    offload_inactive_models: bool = False
    memory_log_interval: int = 25
    generator_device: str | None = None
    discriminator_device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GADTrainResult:
    checkpoint_dir: Path
    discriminator_dir: Path
    manifest_path: Path
    metrics: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _ensure_tokenizer(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise GADError("tokenizer must define pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_device(value: str | None) -> torch.device:
    if value:
        device = torch.device(value)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GADError(f"CUDA device requested but CUDA is not available: {device}")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise GADError(f"CUDA device index is outside visible devices: {device}")
    return device


def _cuda_memory_snapshot(stage: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"stage": stage}
    if not torch.cuda.is_available():
        payload["cuda_available"] = False
        return payload
    devices = []
    for device in range(torch.cuda.device_count()):
        devices.append(
            {
                "device": int(device),
                "allocated_gib": float(torch.cuda.memory_allocated(device) / 1024**3),
                "reserved_gib": float(torch.cuda.memory_reserved(device) / 1024**3),
                "max_allocated_gib": float(torch.cuda.max_memory_allocated(device) / 1024**3),
                "max_reserved_gib": float(torch.cuda.max_memory_reserved(device) / 1024**3),
            }
        )
    payload.update(
        {
            "cuda_available": True,
            "devices": devices,
        }
    )
    return payload


def _peak_cuda_metric(memory_trace: Sequence[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in memory_trace:
        for device_payload in row.get("devices", []):
            values.append(float(device_payload.get(key, 0.0)))
    return max(values, default=0.0)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _move_trainable_bundle(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.to(device)
    _move_optimizer_state(optimizer, device)


def _torch_dtype(bf16: bool) -> torch.dtype | str:
    return torch.bfloat16 if bf16 else "auto"


def _is_local_peft_adapter(model_id: str) -> bool:
    return (Path(model_id) / "adapter_config.json").exists()


def _maybe_enable_gradient_checkpointing(model: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False


def _maybe_apply_causal_lm_lora(model: nn.Module, config: GADTrainingConfig) -> nn.Module:
    if not config.use_lora:
        return model
    if LoraConfig is None or get_peft_model is None:
        raise GADError("peft is not available but use_lora=True was requested.")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def _maybe_apply_backbone_lora(model: nn.Module, config: GADTrainingConfig) -> nn.Module:
    if not config.use_lora:
        return model
    if LoraConfig is None or get_peft_model is None:
        raise GADError("peft is not available but use_lora=True was requested.")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(model, lora_config)


def _load_generator_model(model_id: str, config: GADTrainingConfig) -> nn.Module:
    if _is_local_peft_adapter(model_id):
        if AutoPeftModelForCausalLM is None:
            raise GADError("peft AutoPeftModelForCausalLM is required to load a LoRA warmup checkpoint.")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            is_trainable=True,
            torch_dtype=_torch_dtype(config.bf16),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=_torch_dtype(config.bf16))
        model = _maybe_apply_causal_lm_lora(model, config)
    _maybe_enable_gradient_checkpointing(model, config.gradient_checkpointing)
    return model


class SequenceDiscriminator(nn.Module):
    """LM backbone plus scalar score head, matching GAD's critic-as-D idea."""

    def __init__(self, model_id: str, config: GADTrainingConfig):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id, torch_dtype=_torch_dtype(config.bf16))
        self.backbone = _maybe_apply_backbone_lora(self.backbone, config)
        _maybe_enable_gradient_checkpointing(self.backbone, config.gradient_checkpointing)
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise GADError(f"discriminator backbone has no hidden_size: {model_id}")
        self.score_head = nn.Linear(int(hidden_size), 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = outputs.last_hidden_state
        last_indices = attention_mask.sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_indices, last_indices]
        last_hidden = last_hidden.to(dtype=self.score_head.weight.dtype)
        return self.score_head(last_hidden).squeeze(-1)

    def save_artifacts(self, output_dir: Path, tokenizer: Any) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        torch.save(self.score_head.state_dict(), output_dir / "score_head.pt")
        _write_json(
            output_dir / "gad_discriminator_head.json",
            {
                "head_type": "linear_scalar_score_head",
                "score_position": "last_valid_token_of_rendered_prompt_response",
                "loss": "bradley_terry_pairwise_preference",
            },
        )


def _score_texts(discriminator: SequenceDiscriminator, tokenizer: Any, texts: Sequence[str], max_length: int) -> torch.Tensor:
    device = _model_device(discriminator)
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return discriminator(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])


def _render_full_texts(tokenizer: Any, prompt_text: str, responses: Sequence[str]) -> list[str]:
    return [_full_text_without_eos(tokenizer, prompt_text, response) for response in responses]


def _sequence_mean_log_prob(
    model: nn.Module,
    tokenizer: Any,
    prompt_text: str,
    response_text: str,
    max_seq_length: int,
) -> torch.Tensor | None:
    device = _model_device(model)
    prompt_rendered = _prompt_prefix_text(tokenizer, prompt_text)
    full_rendered = _full_text_without_eos(tokenizer, prompt_text, response_text)
    prompt_ids = tokenizer(prompt_rendered, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_rendered, add_special_tokens=False).input_ids
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise GADError("chat template boundary mismatch while computing GAD log probability.")
    if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
        full_ids = [*full_ids, tokenizer.eos_token_id]
    full_ids = full_ids[:max_seq_length]
    if len(full_ids) <= len(prompt_ids):
        return None

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, use_cache=False)
    log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(1, input_ids.size(1), device=device)
    response_mask = positions >= min(len(prompt_ids), input_ids.size(1) - 1)
    if not bool(response_mask.any()):
        return None
    return token_log_probs[:, response_mask].mean()


def _sequence_mean_log_probs(
    model: nn.Module,
    tokenizer: Any,
    prompt_text: str,
    response_texts: Sequence[str],
    max_seq_length: int,
) -> list[torch.Tensor | None]:
    device = _model_device(model)
    prompt_rendered = _prompt_prefix_text(tokenizer, prompt_text)
    prompt_ids = tokenizer(prompt_rendered, add_special_tokens=False).input_ids
    if not response_texts:
        return []

    sequences: list[list[int]] = []
    response_starts: list[int] = []
    valid_indices: list[int] = []
    results: list[torch.Tensor | None] = [None] * len(response_texts)
    for index, response_text in enumerate(response_texts):
        full_rendered = _full_text_without_eos(tokenizer, prompt_text, response_text)
        full_ids = tokenizer(full_rendered, add_special_tokens=False).input_ids
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise GADError("chat template boundary mismatch while computing GAD log probability.")
        if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
            full_ids = [*full_ids, tokenizer.eos_token_id]
        full_ids = full_ids[:max_seq_length]
        if len(full_ids) <= len(prompt_ids):
            continue
        sequences.append(full_ids)
        response_starts.append(min(len(prompt_ids), len(full_ids) - 1))
        valid_indices.append(index)

    if not sequences:
        return results

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise GADError("tokenizer must define pad_token_id or eos_token_id.")
        pad_token_id = tokenizer.eos_token_id
    max_length = max(len(sequence) for sequence in sequences)
    padded = [sequence + [pad_token_id] * (max_length - len(sequence)) for sequence in sequences]
    attention_mask = [
        [1] * len(sequence) + [0] * (max_length - len(sequence))
        for sequence in sequences
    ]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention = torch.tensor(attention_mask, dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(1, input_ids.size(1), device=device).unsqueeze(0)
    valid_token_mask = attention[:, 1:].bool()
    response_start_tensor = torch.tensor(response_starts, dtype=torch.long, device=device).unsqueeze(1)
    response_mask = (positions >= response_start_tensor) & valid_token_mask

    for row_index, output_index in enumerate(valid_indices):
        mask = response_mask[row_index]
        if bool(mask.any()):
            results[output_index] = token_log_probs[row_index, mask].mean()
    return results


def _generate_group(model: nn.Module, tokenizer: Any, prompt_text: str, config: GADTrainingConfig) -> list[str]:
    device = _model_device(model)
    prompt = _prompt_prefix_text(tokenizer, prompt_text)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_prompt_length,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_response_length,
            num_return_sequences=config.group_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = input_ids.size(1)
    responses = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
    return [response.strip() or (tokenizer.eos_token or "") for response in responses]


def _bt_update(
    discriminator: SequenceDiscriminator,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    prompt_text: str,
    teacher_response: str,
    student_responses: Sequence[str],
    config: GADTrainingConfig,
) -> dict[str, float]:
    discriminator.train()
    teacher_texts = _render_full_texts(tokenizer, prompt_text, [teacher_response] * len(student_responses))
    student_texts = _render_full_texts(tokenizer, prompt_text, student_responses)
    teacher_scores = _score_texts(discriminator, tokenizer, teacher_texts, config.max_seq_length)
    student_scores = _score_texts(discriminator, tokenizer, student_texts, config.max_seq_length)
    loss = -F.logsigmoid(teacher_scores - student_scores).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        acc = (teacher_scores > student_scores).float().mean().item()
        return {
            "bt_loss": float(loss.detach().cpu()),
            "d_acc": float(acc),
            "teacher_score_mean": float(teacher_scores.mean().detach().cpu()),
            "student_score_mean": float(student_scores.mean().detach().cpu()),
        }


def _policy_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    prompt_text: str,
    responses: Sequence[str],
    old_log_probs: Sequence[torch.Tensor],
    advantages: torch.Tensor,
    config: GADTrainingConfig,
) -> float | None:
    model.train()
    losses: list[torch.Tensor] = []
    new_log_probs = _sequence_mean_log_probs(model, tokenizer, prompt_text, responses, config.max_seq_length)
    for new_log_prob, old_log_prob, advantage in zip(new_log_probs, old_log_probs, advantages):
        if new_log_prob is None:
            continue
        old_log_prob = old_log_prob.detach().to(new_log_prob.device)
        advantage = advantage.detach().to(new_log_prob.device)
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped = ratio.clamp(0.8, 1.2)
        pg_loss = -torch.minimum(ratio * advantage, clipped * advantage)
        kl_loss = (new_log_prob - old_log_prob).pow(2)
        losses.append(pg_loss + config.kl_beta * kl_loss)
    if not losses:
        return None
    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu())


def train_gad(config: GADTrainingConfig) -> GADTrainResult:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    generator_device = _resolve_device(config.generator_device)
    discriminator_device = _resolve_device(config.discriminator_device)
    cpu_device = torch.device("cpu")
    memory_trace: list[dict[str, Any]] = [_cuda_memory_snapshot("start")]

    bundle = load_transcript_bundle(Path(config.transcript_dir), budget=config.requested_budget)
    records = list(bundle.records)
    if not records:
        raise GADError("GAD requires a non-empty teacher transcript.")
    if config.group_size < 2:
        raise GADError("GAD requires group_size >= 2 for GRPO-style group normalization.")

    tokenizer = AutoTokenizer.from_pretrained(config.generator_model_id)
    _ensure_tokenizer(tokenizer)
    generator = _load_generator_model(config.generator_model_id, config).to(generator_device)
    if generator.config.pad_token_id is None:
        generator.config.pad_token_id = tokenizer.pad_token_id
    _cuda_cleanup()
    memory_trace.append(_cuda_memory_snapshot("after_generator_load"))

    discriminator = SequenceDiscriminator(config.discriminator_model_id, config).to(discriminator_device)
    d_optimizer = torch.optim.AdamW(
        [parameter for parameter in discriminator.parameters() if parameter.requires_grad],
        lr=config.discriminator_learning_rate,
    )
    g_optimizer = torch.optim.AdamW(
        [parameter for parameter in generator.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    _cuda_cleanup()
    memory_trace.append(_cuda_memory_snapshot("after_discriminator_load"))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_metrics: list[dict[str, float]] = []
    for step in range(max(0, config.discriminator_warmup_steps)):
        record = records[step % len(records)]
        if config.offload_inactive_models:
            _move_trainable_bundle(generator, g_optimizer, generator_device)
        student_responses = _generate_group(generator, tokenizer, record.prompt_text, config)
        if config.offload_inactive_models:
            _move_trainable_bundle(generator, g_optimizer, cpu_device)
            _cuda_cleanup()
            _move_trainable_bundle(discriminator, d_optimizer, discriminator_device)
        warmup_metrics.append(
            _bt_update(
                discriminator,
                d_optimizer,
                tokenizer,
                record.prompt_text,
                record.teacher_response,
                student_responses,
                config,
            )
        )
        if config.offload_inactive_models:
            _move_trainable_bundle(discriminator, d_optimizer, cpu_device)
            _cuda_cleanup()
        if config.memory_log_interval > 0 and step % config.memory_log_interval == 0:
            memory_trace.append(_cuda_memory_snapshot(f"discriminator_warmup_step_{step}"))
    _cuda_cleanup()
    memory_trace.append(_cuda_memory_snapshot("after_discriminator_warmup"))

    total_steps = config.max_steps if config.max_steps > 0 else max(1, int(round(len(records) * config.gad_epochs)))
    policy_losses: list[float] = []
    bt_metrics: list[dict[str, float]] = []
    reward_means: list[float] = []

    for step in range(total_steps):
        record = records[step % len(records)]
        if config.offload_inactive_models:
            _move_trainable_bundle(generator, g_optimizer, generator_device)
        responses = _generate_group(generator, tokenizer, record.prompt_text, config)
        with torch.no_grad():
            old_log_prob_candidates = _sequence_mean_log_probs(
                generator,
                tokenizer,
                record.prompt_text,
                responses,
                config.max_seq_length,
            )
            valid_rollouts = [
                (response, old_log_prob.detach().cpu(), index)
                for index, (response, old_log_prob) in enumerate(zip(responses, old_log_prob_candidates))
                if old_log_prob is not None
            ]
        if len(valid_rollouts) != len(responses):
            if config.offload_inactive_models:
                _move_trainable_bundle(generator, g_optimizer, cpu_device)
                _cuda_cleanup()
            continue
        valid_responses = [response for response, _, _ in valid_rollouts]
        old_log_probs = [old_log_prob for _, old_log_prob, _ in valid_rollouts]

        if config.offload_inactive_models:
            _move_trainable_bundle(generator, g_optimizer, cpu_device)
            _cuda_cleanup()
            _move_trainable_bundle(discriminator, d_optimizer, discriminator_device)
        discriminator.eval()
        with torch.no_grad():
            reward_scores = _score_texts(discriminator, tokenizer, _render_full_texts(tokenizer, record.prompt_text, valid_responses), config.max_seq_length)
        reward_mean = reward_scores.mean()
        reward_std = reward_scores.std(unbiased=False).clamp_min(1e-6)
        advantages = ((reward_scores - reward_mean) / reward_std).detach()
        reward_means.append(float(reward_mean.detach().cpu()))

        bt_metrics.append(
            _bt_update(
                discriminator,
                d_optimizer,
                tokenizer,
                record.prompt_text,
                record.teacher_response,
                valid_responses,
                config,
            )
        )
        if config.offload_inactive_models:
            _move_trainable_bundle(discriminator, d_optimizer, cpu_device)
            _cuda_cleanup()
            _move_trainable_bundle(generator, g_optimizer, generator_device)
        policy_loss = _policy_update(generator, g_optimizer, tokenizer, record.prompt_text, valid_responses, old_log_probs, advantages, config)
        if policy_loss is not None:
            policy_losses.append(policy_loss)
        if config.offload_inactive_models:
            _move_trainable_bundle(generator, g_optimizer, cpu_device)
            _cuda_cleanup()
        if config.memory_log_interval > 0 and step % config.memory_log_interval == 0:
            memory_trace.append(_cuda_memory_snapshot(f"joint_step_{step}"))

    checkpoint_dir = output_dir / "checkpoint-final"
    discriminator_dir = output_dir / "discriminator-final"
    if config.offload_inactive_models:
        _move_trainable_bundle(generator, g_optimizer, generator_device)
    generator.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    if config.offload_inactive_models:
        _move_trainable_bundle(generator, g_optimizer, cpu_device)
        _cuda_cleanup()
    if config.offload_inactive_models:
        _move_trainable_bundle(discriminator, d_optimizer, discriminator_device)
    discriminator.save_artifacts(discriminator_dir, tokenizer)
    _cuda_cleanup()
    memory_trace.append(_cuda_memory_snapshot("after_save"))

    def mean_metric(items: Sequence[dict[str, float]], key: str) -> float:
        values = [float(item[key]) for item in items if key in item]
        return sum(values) / max(1, len(values))

    metrics = {
        "discriminator_warmup_steps": len(warmup_metrics),
        "discriminator_warmup_bt_loss": mean_metric(warmup_metrics, "bt_loss"),
        "joint_steps": total_steps,
        "mean_policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
        "mean_bt_loss": mean_metric(bt_metrics, "bt_loss"),
        "mean_d_acc": mean_metric(bt_metrics, "d_acc"),
        "mean_reward": sum(reward_means) / max(1, len(reward_means)),
        "teacher_successful_queries": bundle.manifest.actual_successful_queries,
        "teacher_model_id": bundle.manifest.teacher_model_id,
        "generator_device": str(generator_device),
        "discriminator_device": str(discriminator_device),
        "peak_cuda_allocated_gib": _peak_cuda_metric(memory_trace, "max_allocated_gib"),
        "peak_cuda_reserved_gib": _peak_cuda_metric(memory_trace, "max_reserved_gib"),
    }
    manifest = {
        "schema_version": GAD_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "checkpoint_dir": str(checkpoint_dir),
        "discriminator_dir": str(discriminator_dir),
        "transcript_dir": config.transcript_dir,
        "config": config.to_dict(),
        "metrics": metrics,
        "memory_trace": memory_trace,
        "algorithm_notes": {
            "teacher_access": "offline_text_transcript_only",
            "generator_warmup": "shared_seqkd_sft_before_this_module",
            "discriminator_loss": "bradley_terry_pairwise_preference",
            "discriminator_score": "scalar score at rendered prompt-response last valid token",
            "generator_update": "grpo_style_group_normalized_ppo_on_discriminator_reward",
            "student_rollouts_per_prompt": config.group_size,
            "teacher_response_shared_within_group": True,
            "official_reference": "YTianZHU/verl gad branch critic-as-discriminator implementation",
        },
    }
    manifest_path = output_dir / "gad_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(output_dir / "gad_training_config.json", config.to_dict())
    return GADTrainResult(
        checkpoint_dir=checkpoint_dir,
        discriminator_dir=discriminator_dir,
        manifest_path=manifest_path,
        metrics=metrics,
    )


__all__ = ["GADError", "GADTrainingConfig", "GADTrainResult", "train_gad"]
