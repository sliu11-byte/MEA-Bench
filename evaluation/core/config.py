from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    pass


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return ""

    return ENV_PATTERN.sub(replace, value)


def expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


@dataclass
class ModelSpec:
    key: str
    model_id: str
    role: str
    family: str
    size: str
    checkpoint_type: str
    base_url: str
    api_key: str
    request_model_name: str
    prompt_rendering_mode: str
    backend: str
    dtype: str = "auto"
    quantization_setting: str | None = None
    tensor_parallel_size: int | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    model_revision: str | None = None

    def sanitized(self) -> dict[str, Any]:
        data = copy.deepcopy(self.__dict__)
        data.pop("api_key", None)
        return data


@dataclass
class DatasetSpec:
    key: str
    display_name: str
    adapter: str
    source: str
    identifier: str
    config_name: str | None
    revision: str | None
    requested_splits: list[str]
    task_type: str
    trust_remote_code: bool = False
    local_path: str | None = None
    official_split_output_name: str | None = None
    provenance_note: str | None = None


@dataclass
class GenerationSpec:
    seed: int = 42
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0
    max_tokens_multiple_choice: int = 32
    max_tokens_gsm8k: int = 512
    max_tokens_domain_qa: int = 256
    max_tokens_humaneval: int = 1024
    concurrency: int = 1
    stop_sequences: list[str] = field(default_factory=list)

    def max_tokens_for(self, task_type: str) -> int:
        if task_type == "numeric_qa":
            return self.max_tokens_gsm8k
        if task_type == "code_generation":
            return self.max_tokens_humaneval
        if task_type == "multiple_choice":
            return self.max_tokens_multiple_choice
        return self.max_tokens_domain_qa


@dataclass
class RetrySpec:
    max_attempts: int = 4
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    multiplier: float = 2.0
    request_timeout_seconds: float = 120.0


@dataclass
class SamplingSpec:
    selection_seed: int = 42
    manifest_pool_size: int = 100
    smoke_examples_per_split: int = 2
    pilot_examples_per_split: int = 10
    humaneval_pilot_examples: int = 10


@dataclass
class M1Config:
    source_path: Path
    version: int
    output_root: Path
    models: dict[str, ModelSpec]
    datasets: dict[str, DatasetSpec]
    generation: GenerationSpec
    retry: RetrySpec
    sampling: SamplingSpec
    raw_resolved: dict[str, Any]

    def sanitized_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(self.raw_resolved)
        for model in result.get("models", {}).values():
            model.pop("api_key", None)
            base_url = model.get("base_url", "")
            if base_url:
                parts = urlsplit(base_url)
                host = parts.hostname or ""
                netloc = f"{host}:{parts.port}" if parts.port else host
                model["base_url"] = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
        return result


def load_config(path: str | Path) -> M1Config:
    source_path = Path(path).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("M1 config must be a YAML mapping")
    resolved = expand_environment(raw)

    models = {
        key: ModelSpec(key=key, **value)
        for key, value in resolved.get("models", {}).items()
    }
    datasets = {
        key: DatasetSpec(key=key, **value)
        for key, value in resolved.get("datasets", {}).items()
    }
    config = M1Config(
        source_path=source_path,
        version=int(resolved.get("version", 1)),
        output_root=Path(resolved.get("output_root", "results/m1_rollout")),
        models=models,
        datasets=datasets,
        generation=GenerationSpec(**resolved.get("generation", {})),
        retry=RetrySpec(**resolved.get("retry", {})),
        sampling=SamplingSpec(**resolved.get("sampling", {})),
        raw_resolved=resolved,
    )
    validate_config(config)
    return config


def validate_config(config: M1Config) -> None:
    expected_models = {
        "qwen_72b_target",
        "qwen_7b_same_family",
        "mistral_7b_cross_family",
    }
    expected_datasets = {
        "arc_challenge", "hellaswag", "mmlu", "truthfulqa", "winogrande", "gsm8k",
        "medqa", "pubmedqa", "chemprot", "fomc", "headline", "fpb", "humaneval",
    }
    if set(config.models) != expected_models:
        raise ConfigError(f"M1 requires exactly these model keys: {sorted(expected_models)}")
    if set(config.datasets) != expected_datasets:
        raise ConfigError(f"M1 requires exactly these dataset keys: {sorted(expected_datasets)}")
    if config.generation.temperature < 0:
        raise ConfigError("temperature must be non-negative")
    if not 0 < config.generation.top_p <= 1:
        raise ConfigError("top_p must be in (0, 1]")
    if config.generation.concurrency < 1:
        raise ConfigError("concurrency must be at least 1")
    if config.retry.max_attempts < 1:
        raise ConfigError("retry.max_attempts must be at least 1")
    if config.sampling.manifest_pool_size < 1:
        raise ConfigError("sampling.manifest_pool_size must be at least 1")
    if config.sampling.smoke_examples_per_split < 1:
        raise ConfigError("sampling.smoke_examples_per_split must be at least 1")
    if config.sampling.pilot_examples_per_split < 1:
        raise ConfigError("sampling.pilot_examples_per_split must be at least 1")
    if config.sampling.humaneval_pilot_examples < 1:
        raise ConfigError("sampling.humaneval_pilot_examples must be at least 1")
    for model in config.models.values():
        if model.prompt_rendering_mode not in {"chat_completions", "completions"}:
            raise ConfigError(f"Unsupported prompt_rendering_mode for {model.key}")


def apply_cli_overrides(config: M1Config, args: Any) -> None:
    names = [
        "seed", "temperature", "top_p", "top_k", "repetition_penalty",
        "max_tokens_multiple_choice", "max_tokens_gsm8k", "max_tokens_domain_qa",
        "max_tokens_humaneval", "concurrency",
    ]
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            setattr(config.generation, name, value)
            config.raw_resolved.setdefault("generation", {})[name] = value

    retry_names = [
        "max_attempts", "initial_backoff_seconds", "max_backoff_seconds",
        "multiplier", "request_timeout_seconds",
    ]
    for name in retry_names:
        value = getattr(args, f"retry_{name}", None)
        if value is not None:
            setattr(config.retry, name, value)
            config.raw_resolved.setdefault("retry", {})[name] = value

    sampling_names = [
        "manifest_pool_size", "smoke_examples_per_split", "pilot_examples_per_split",
        "humaneval_pilot_examples",
    ]
    for name in sampling_names:
        value = getattr(args, name, None)
        if value is not None:
            setattr(config.sampling, name, value)
            config.raw_resolved.setdefault("sampling", {})[name] = value

    for override in getattr(args, "endpoint", []) or []:
        key, separator, value = override.partition("=")
        if not separator or key not in config.models:
            raise ConfigError(f"Invalid --endpoint override: {override}")
        config.models[key].base_url = value
        config.raw_resolved["models"][key]["base_url"] = value

    for override in getattr(args, "request_model", []) or []:
        key, separator, value = override.partition("=")
        if not separator or key not in config.models:
            raise ConfigError(f"Invalid --request-model override: {override}")
        config.models[key].request_model_name = value
        config.raw_resolved["models"][key]["request_model_name"] = value

    validate_config(config)
