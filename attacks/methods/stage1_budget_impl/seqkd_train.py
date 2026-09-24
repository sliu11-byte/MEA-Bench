from __future__ import annotations

"""
======================================================================
SEQKD_TRAIN ---

Stage 2 Seq-KD training entry.

This module consumes the shared stage 1 transcript, builds response-only
causal-LM supervision, trains a student model, and writes a Hugging Face
checkpoint plus a training manifest.
======================================================================
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None
    get_peft_model = None

from .stage1_transcript import (
    CODE_VERSION as STAGE1_CODE_VERSION,
    SCHEMA_VERSION as STAGE1_SCHEMA_VERSION,
    TranscriptBundle,
    TranscriptConfigMismatchError,
    TranscriptRecord,
    TranscriptSchemaError,
    git_commit,
    load_transcript_bundle,
    stable_hash,
    transcript_hash_from_records,
)


SEQKD_SCHEMA_VERSION = "seqkd_v1"
SEQKD_CODE_VERSION = "seqkd_train_v1"

FORBIDDEN_TRANSCRIPT_FIELDS = {
    "teacher_logits",
    "logits",
    "token_probabilities",
    "top_k_probabilities",
    "hidden_states",
    "gradients",
}


class SeqKDError(RuntimeError):
    pass


class SeqKDTranscriptError(SeqKDError):
    pass


class SeqKDDatasetError(SeqKDError):
    pass


class SeqKDTrainingError(SeqKDError):
    pass


@dataclass(frozen=True)
class SeqKDTrainingConfig:
    transcript_dir: str
    output_dir: str
    student_model_id: str
    requested_budget: int
    seed: int
    max_seq_length: int = 2048
    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.0
    fp16: bool = False
    bf16: bool = False
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    allow_sampling_transcript: bool = False
    include_eos_in_loss: bool = True
    save_total_limit: int = 1
    logging_steps: int = 1
    save_strategy: str = "no"
    report_to: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class SeqKDDataStats:
    total_records: int
    valid_records: int
    skipped_records: int
    truncated_records: int
    eos_included_records: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SeqKDTrainingManifest:
    schema_version: str
    code_version: str
    git_commit: str
    created_at: str
    output_dir: str
    checkpoint_dir: str
    tokenizer_dir: str
    transcript_dir: str
    transcript_hash: str
    transcript_source_hash: str
    transcript_prefix_hash: str
    prompt_ordering_hash: str
    prompt_pool_id: str
    teacher_model_id: str
    decode_config: Mapping[str, Any]
    student_model_id: str
    requested_budget: int
    actual_successful_queries: int
    seed: int
    max_seq_length: int
    learning_rate: float
    num_train_epochs: float
    max_steps: int
    effective_batch_size: int
    gradient_accumulation_steps: int
    optimizer: str
    scheduler: str
    warmup_ratio: float
    precision: str
    use_lora: bool
    gradient_checkpointing: bool
    train_loss: float
    valid_train_samples: int
    skipped_samples: int
    truncated_samples: int
    eos_included_samples: int
    supervision_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeqKDTrainResult:
    manifest: SeqKDTrainingManifest
    manifest_path: Path
    checkpoint_dir: Path
    dataset_stats: SeqKDDataStats


class SeqKDTensorDataset(Dataset):
    def __init__(self, examples: Sequence[Mapping[str, torch.Tensor]]):
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        return self._examples[idx]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_decode_config(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json(dict(value)))


def _prompt_prefix_text(tokenizer: PreTrainedTokenizerBase, prompt_text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt_text}\nAssistant: "


def _full_text_without_eos(tokenizer: PreTrainedTokenizerBase, prompt_text: str, teacher_response: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": teacher_response},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return f"User: {prompt_text}\nAssistant: {teacher_response}"


def _ensure_tokenizer_chat_defaults(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise SeqKDDatasetError("tokenizer must define either pad_token_id or eos_token_id.")
    if tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"


def _encode_seqkd_example(
    tokenizer: PreTrainedTokenizerBase,
    record: TranscriptRecord,
    max_seq_length: int,
    include_eos_in_loss: bool,
) -> tuple[dict[str, torch.Tensor] | None, bool, bool, bool]:
    prompt_rendered = _prompt_prefix_text(tokenizer, record.prompt_text)
    full_rendered = _full_text_without_eos(tokenizer, record.prompt_text, record.teacher_response)

    prompt_ids = tokenizer(prompt_rendered, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_rendered, add_special_tokens=False).input_ids

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise SeqKDDatasetError(
            "chat template boundary mismatch: the prompt prefix is not a prefix of the full conversation."
        )

    eos_included = False
    if include_eos_in_loss and tokenizer.eos_token_id is not None and (
        not full_ids or full_ids[-1] != tokenizer.eos_token_id
    ):
        full_ids = list(full_ids) + [tokenizer.eos_token_id]
        eos_included = True

    if len(prompt_ids) >= max_seq_length:
        return None, False, False, eos_included

    truncated = len(full_ids) > max_seq_length
    full_ids = list(full_ids[:max_seq_length])
    if len(full_ids) <= len(prompt_ids):
        return None, truncated, False, eos_included

    labels = [-100] * len(full_ids)
    labels[len(prompt_ids) :] = full_ids[len(prompt_ids) :]

    attention_mask = [1] * len(full_ids)
    pad_len = max_seq_length - len(full_ids)
    if pad_len > 0:
        full_ids.extend([tokenizer.pad_token_id] * pad_len)
        attention_mask.extend([0] * pad_len)
        labels.extend([-100] * pad_len)

    example = {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    return example, truncated, True, eos_included


def build_seqkd_examples(
    bundle: TranscriptBundle,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    include_eos_in_loss: bool = True,
) -> tuple[list[dict[str, torch.Tensor]], SeqKDDataStats]:
    _ensure_tokenizer_chat_defaults(tokenizer)

    examples: list[dict[str, torch.Tensor]] = []
    skipped = 0
    truncated = 0
    eos_included = 0

    for record in bundle.records:
        example, was_truncated, was_valid, did_include_eos = _encode_seqkd_example(
            tokenizer,
            record,
            max_seq_length=max_seq_length,
            include_eos_in_loss=include_eos_in_loss,
        )
        if was_truncated:
            truncated += 1
        if did_include_eos:
            eos_included += 1
        if not was_valid or example is None:
            skipped += 1
            continue
        examples.append(example)

    stats = SeqKDDataStats(
        total_records=len(bundle.records),
        valid_records=len(examples),
        skipped_records=skipped,
        truncated_records=truncated,
        eos_included_records=eos_included,
    )
    return examples, stats


def build_seqkd_dataset(
    bundle: TranscriptBundle,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    include_eos_in_loss: bool = True,
) -> tuple[SeqKDTensorDataset, SeqKDDataStats]:
    examples, stats = build_seqkd_examples(
        bundle=bundle,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        include_eos_in_loss=include_eos_in_loss,
    )
    if not examples:
        raise SeqKDDatasetError("no valid training examples remained after preprocessing.")
    return SeqKDTensorDataset(examples), stats


def compute_seqkd_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if response_mask is None:
        response_mask = labels != -100
    shift_mask = response_mask[..., 1:].contiguous().to(dtype=shift_logits.dtype)
    safe_labels = shift_labels.masked_fill(shift_mask == 0, 0)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        safe_labels.view(-1),
    ).view_as(shift_labels)
    loss = (token_loss * shift_mask).sum()
    token_count = shift_mask.sum().clamp(min=1.0)
    return loss / token_count


def _inspect_prompt_ids(bundle: TranscriptBundle) -> list[str]:
    return [record.prompt_id for record in bundle.records]


def _validate_transcript_for_seqkd(
    bundle: TranscriptBundle,
    requested_budget: int,
    allow_sampling_transcript: bool,
) -> None:
    manifest = bundle.manifest
    if manifest.schema_version != STAGE1_SCHEMA_VERSION:
        raise SeqKDTranscriptError(
            f"unexpected transcript schema version: {manifest.schema_version!r}"
        )

    if requested_budget > manifest.requested_budget:
        raise SeqKDTranscriptError(
            f"requested_budget={requested_budget} exceeds available transcript budget={manifest.requested_budget}."
        )

    if len(bundle.records) != requested_budget:
        raise SeqKDTranscriptError(
            f"loaded transcript prefix has {len(bundle.records)} records, expected requested_budget={requested_budget}."
        )

    if bundle.manifest.prompt_ordering_hash != bundle.ordering.ordering_hash:
        raise SeqKDTranscriptError("prompt ordering hash mismatch between manifest and ordering payload.")

    if transcript_hash_from_records(bundle.records) != manifest.transcript_hash:
        raise SeqKDTranscriptError("transcript hash mismatch.")

    if len({record.prompt_id for record in bundle.records}) != len(bundle.records):
        raise SeqKDTranscriptError("each prompt must appear exactly once.")

    if any(record.query_status != "success" for record in bundle.records):
        raise SeqKDTranscriptError("all transcript records must have query_status == 'success'.")

    if len({record.query_id for record in bundle.records}) != len(bundle.records):
        raise SeqKDTranscriptError("query_id must be unique per successful transcript record.")

    expected_indices = list(range(len(bundle.records)))
    if [record.order_index for record in bundle.records] != expected_indices:
        raise SeqKDTranscriptError("record order_index values must be contiguous and start at 0.")

    decode_config = _normalize_decode_config(manifest.decode_config)
    strategy = str(decode_config.get("strategy", "")).lower()
    settings = decode_config.get("settings", {})

    if strategy == "sampling" and not allow_sampling_transcript:
        raise SeqKDTranscriptError(
            "sampling transcripts are rejected in the default Seq-KD mode."
        )
    if strategy == "beam":
        num_beams = settings.get("num_beams", settings.get("beam_size"))
        if int(num_beams) != 5:
            raise SeqKDTranscriptError(
                f"beam transcript must use num_beams=5 in default Seq-KD mode, got {num_beams!r}."
            )
    elif strategy == "greedy":
        pass
    elif strategy == "external_jsonl":
        pass
    elif strategy == "sampling":
        if allow_sampling_transcript:
            return
    else:
        raise SeqKDTranscriptError(f"unsupported decode strategy: {strategy!r}")


def load_seqkd_bundle(
    transcript_dir: str | Path,
    requested_budget: int,
    *,
    allow_sampling_transcript: bool = False,
) -> TranscriptBundle:
    bundle = load_transcript_bundle(Path(transcript_dir), budget=requested_budget)
    _validate_transcript_for_seqkd(
        bundle=bundle,
        requested_budget=requested_budget,
        allow_sampling_transcript=allow_sampling_transcript,
    )
    return bundle


def _train_arguments(
    config: SeqKDTrainingConfig,
) -> TrainingArguments:
    output_dir = str(Path(config.output_dir))
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        bf16=config.bf16,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        report_to=list(config.report_to),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )


def _build_model(
    student_model_id: str,
    config: SeqKDTrainingConfig,
    model_factory: Callable[[], PreTrainedModel] | None = None,
) -> PreTrainedModel:
    if model_factory is not None:
        model = model_factory()
    else:
        torch_dtype = torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else "auto"
        model = AutoModelForCausalLM.from_pretrained(student_model_id, torch_dtype=torch_dtype)
    return model


def _maybe_apply_lora(model: PreTrainedModel, config: SeqKDTrainingConfig) -> PreTrainedModel:
    if not config.use_lora:
        return model
    if LoraConfig is None or get_peft_model is None:
        raise SeqKDTrainingError("peft is not available but use_lora=True was requested.")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def _maybe_enable_gradient_checkpointing(model: PreTrainedModel, config: SeqKDTrainingConfig) -> None:
    if not config.gradient_checkpointing:
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


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(dict(payload)), encoding="utf-8")


def train_seqkd(
    config: SeqKDTrainingConfig,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    model_factory: Callable[[], PreTrainedModel] | None = None,
) -> SeqKDTrainResult:
    set_seed(config.seed)

    bundle = load_seqkd_bundle(
        config.transcript_dir,
        requested_budget=config.requested_budget,
        allow_sampling_transcript=config.allow_sampling_transcript,
    )

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.student_model_id)
    _ensure_tokenizer_chat_defaults(tokenizer)

    dataset, stats = build_seqkd_dataset(
        bundle=bundle,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        include_eos_in_loss=config.include_eos_in_loss,
    )

    model = _build_model(config.student_model_id, config=config, model_factory=model_factory)
    model = _maybe_apply_lora(model, config)
    _maybe_enable_gradient_checkpointing(model, config)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = _train_arguments(config)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    train_output = trainer.train()

    checkpoint_dir = output_dir / "checkpoint-final"
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    transcript_hash = bundle.manifest.transcript_hash
    transcript_prefix_hash = transcript_hash_from_records(bundle.records)
    precision = "bf16" if config.bf16 else "fp16" if config.fp16 else "fp32"
    manifest = SeqKDTrainingManifest(
        schema_version=SEQKD_SCHEMA_VERSION,
        code_version=SEQKD_CODE_VERSION,
        git_commit=git_commit(),
        created_at=_utc_now_iso(),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        tokenizer_dir=str(checkpoint_dir),
        transcript_dir=str(Path(config.transcript_dir)),
        transcript_hash=transcript_hash,
        transcript_source_hash=bundle.manifest.transcript_hash,
        transcript_prefix_hash=transcript_prefix_hash,
        prompt_ordering_hash=bundle.manifest.prompt_ordering_hash,
        prompt_pool_id=bundle.manifest.prompt_pool_id,
        teacher_model_id=bundle.manifest.teacher_model_id,
        decode_config=_normalize_decode_config(bundle.manifest.decode_config),
        student_model_id=config.student_model_id,
        requested_budget=config.requested_budget,
        actual_successful_queries=bundle.manifest.actual_successful_queries,
        seed=config.seed,
        max_seq_length=config.max_seq_length,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        effective_batch_size=config.effective_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optimizer=config.optim,
        scheduler=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        precision=precision,
        use_lora=config.use_lora,
        gradient_checkpointing=config.gradient_checkpointing,
        train_loss=float(train_output.training_loss),
        valid_train_samples=stats.valid_records,
        skipped_samples=stats.skipped_records,
        truncated_samples=stats.truncated_records,
        eos_included_samples=stats.eos_included_records,
        supervision_scope="response_only",
    )
    manifest_path = output_dir / "seqkd_manifest.json"
    _save_json(manifest_path, manifest.to_dict())
    _save_json(output_dir / "seqkd_training_config.json", config.to_dict())
    _save_json(output_dir / "seqkd_dataset_stats.json", stats.to_dict())
    return SeqKDTrainResult(
        manifest=manifest,
        manifest_path=manifest_path,
        checkpoint_dir=checkpoint_dir,
        dataset_stats=stats,
    )


def load_trained_seqkd_model(
    checkpoint_dir: str | Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir))
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--student_model_id", required=True, type=str)
    parser.add_argument("--requested_budget", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=1.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--per_device_train_batch_size", default=1, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--optim", default="adamw_torch", type=str)
    parser.add_argument("--lr_scheduler_type", default="linear", type=str)
    parser.add_argument("--warmup_ratio", default=0.0, type=float)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--allow_sampling_transcript", action="store_true")
    parser.set_defaults(include_eos_in_loss=True)
    parser.add_argument("--include_eos_in_loss", dest="include_eos_in_loss", action="store_true")
    parser.add_argument("--no_include_eos_in_loss", dest="include_eos_in_loss", action="store_false")
    parser.add_argument("--save_total_limit", default=1, type=int)
    parser.add_argument("--logging_steps", default=1, type=int)
    parser.add_argument("--save_strategy", default="no", type=str)
    args = parser.parse_args()

    config = SeqKDTrainingConfig(
        transcript_dir=args.transcript_dir,
        output_dir=args.output_dir,
        student_model_id=args.student_model_id,
        requested_budget=args.requested_budget,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        allow_sampling_transcript=args.allow_sampling_transcript,
        include_eos_in_loss=args.include_eos_in_loss,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
    )
    train_seqkd(config)


__all__ = [
    "SEQKD_SCHEMA_VERSION",
    "SeqKDDatasetError",
    "SeqKDError",
    "SeqKDDataStats",
    "SeqKDTrainResult",
    "SeqKDTrainingConfig",
    "SeqKDTrainingError",
    "SeqKDTrainingManifest",
    "SeqKDTensorDataset",
    "SeqKDTranscriptError",
    "build_seqkd_dataset",
    "build_seqkd_examples",
    "compute_seqkd_loss",
    "load_seqkd_bundle",
    "load_trained_seqkd_model",
    "train_seqkd",
]
