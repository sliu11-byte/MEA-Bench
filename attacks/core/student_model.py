from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_lora_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def infer_lora_base_model(checkpoint: str | Path) -> str | None:
    config_path = Path(checkpoint) / "adapter_config.json"
    if not config_path.exists():
        return None
    value = load_json(config_path).get("base_model_name_or_path")
    return None if value in (None, "") else str(value)


@dataclass(frozen=True)
class StudentCheckpointInfo:
    checkpoint: str
    checkpoint_type: str
    base_model: str
    student_name: str = "student"
    source_manifest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = False
    use_chat_template: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StudentCheckpointModel:
    """Loaded attack student checkpoint with a simple query interface.

    This wrapper supports both full Hugging Face causal-LM checkpoints and PEFT
    LoRA adapter directories. It is intentionally small so evaluation code can
    depend on it without depending on any attacker-specific implementation.
    """

    def __init__(self, *, model: Any, tokenizer: Any, info: StudentCheckpointInfo) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self.model.eval()

    def encode_query(self, query: str, *, use_chat_template: bool = False):
        if use_chat_template and getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": query}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        return self.tokenizer(query, return_tensors="pt")["input_ids"]

    def generate_one(
        self,
        query: str,
        *,
        generation_config: GenerationConfig | None = None,
        **overrides: Any,
    ) -> str:
        import torch

        config = _merge_generation_config(generation_config, overrides)
        input_ids = self.encode_query(query, use_chat_template=config.use_chat_template).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if config.do_sample:
            kwargs.update({"temperature": config.temperature, "top_p": config.top_p})
        with torch.inference_mode():
            output_ids = self.model.generate(**kwargs)
        new_tokens = output_ids[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(
        self,
        queries: Sequence[str],
        *,
        generation_config: GenerationConfig | None = None,
        **overrides: Any,
    ) -> list[str]:
        return [self.generate_one(query, generation_config=generation_config, **overrides) for query in queries]

    def generate_records(
        self,
        queries: Iterable[str] | Iterable[dict[str, Any]],
        *,
        generation_config: GenerationConfig | None = None,
        query_field: str = "query",
        **overrides: Any,
    ) -> list[dict[str, Any]]:
        config = _merge_generation_config(generation_config, overrides)
        records: list[dict[str, Any]] = []
        for index, item in enumerate(queries):
            if isinstance(item, dict):
                query = str(item[query_field])
                record = dict(item)
            else:
                query = str(item)
                record = {"query_id": str(index), "query": query}
            record["response"] = self.generate_one(query, generation_config=config)
            record["student"] = self.info.to_dict()
            record["generation_config"] = config.to_dict()
            records.append(record)
        return records


def _merge_generation_config(config: GenerationConfig | None, overrides: dict[str, Any]) -> GenerationConfig:
    payload = {} if config is None else config.to_dict()
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return GenerationConfig(**payload)


def _load_tokenizer(checkpoint: Path, base_model: str | None):
    from transformers import AutoTokenizer

    candidates = [str(checkpoint)]
    if base_model:
        candidates.append(base_model)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as exc:  # pragma: no cover - runtime dependency path
            last_error = exc
    raise RuntimeError(f"Could not load tokenizer from {candidates}") from last_error


def load_student_from_checkpoint(
    checkpoint: str | Path,
    *,
    base_model: str | None = None,
    student_name: str = "student",
    dtype: str = "auto",
    device_map: str = "auto",
    source_manifest: str | Path | None = None,
) -> StudentCheckpointModel:
    from transformers import AutoModelForCausalLM

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    resolved_base = base_model or infer_lora_base_model(checkpoint_path)
    tokenizer = _load_tokenizer(checkpoint_path, resolved_base)

    if is_lora_checkpoint(checkpoint_path):
        from attacks.core.checkpoint_health import checkpoint_tensor_health

        health = checkpoint_tensor_health(checkpoint_path)
        if not health["all_finite"]:
            raise FloatingPointError(
                f"LoRA checkpoint contains non-finite tensors: {health['nonfinite_tensors'][:8]}"
            )
        if not resolved_base:
            raise ValueError(
                "LoRA adapter checkpoint detected, but no base model was provided and "
                "adapter_config.json does not contain base_model_name_or_path."
            )
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("Loading a LoRA checkpoint requires `peft` in the environment.") from exc
        model = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(checkpoint_path))
        info = StudentCheckpointInfo(
            checkpoint=str(checkpoint_path),
            checkpoint_type="lora_adapter",
            base_model=resolved_base,
            student_name=student_name,
            source_manifest=None if source_manifest is None else str(source_manifest),
        )
        return StudentCheckpointModel(model=model, tokenizer=tokenizer, info=info)

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    info = StudentCheckpointInfo(
        checkpoint=str(checkpoint_path),
        checkpoint_type="full_model",
        base_model=str(checkpoint_path),
        student_name=student_name,
        source_manifest=None if source_manifest is None else str(source_manifest),
    )
    return StudentCheckpointModel(model=model, tokenizer=tokenizer, info=info)


def load_student_from_manifest(
    manifest_path: str | Path,
    *,
    base_model: str | None = None,
    student_name: str | None = None,
    dtype: str = "auto",
    device_map: str = "auto",
) -> StudentCheckpointModel:
    manifest = load_json(manifest_path)
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    checkpoint = result.get("checkpoint_dir") or manifest.get("checkpoint_dir")
    if not checkpoint:
        raise ValueError(f"Manifest has no checkpoint_dir: {manifest_path}")
    resolved_name = student_name or str(result.get("attack") or manifest.get("attack") or "student")
    return load_student_from_checkpoint(
        checkpoint,
        base_model=base_model,
        student_name=resolved_name,
        dtype=dtype,
        device_map=device_map,
        source_manifest=manifest_path,
    )


__all__ = [
    "GenerationConfig",
    "StudentCheckpointInfo",
    "StudentCheckpointModel",
    "infer_lora_base_model",
    "is_lora_checkpoint",
    "load_student_from_checkpoint",
    "load_student_from_manifest",
]
