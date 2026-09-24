from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class AttackRunConfig:
    attack: str
    budget: int
    query_pool_path: Path
    output_dir: Path
    stage1_config_path: Path
    query_ordering_path: Path | None = None
    transcript_dir: Path | None = None
    teacher_transcript_path: Path | None = None
    teacher_backend: str | None = None
    teacher_model: str | None = None
    teacher_endpoint_url: str | None = None
    teacher_request_model: str | None = None
    teacher_api_key: str | None = None
    teacher_mode: str = "chat"
    teacher_temperature: float = 0.0
    teacher_top_p: float = 1.0
    teacher_max_tokens: int = 512
    student_model: str | None = None
    student_endpoint_url: str | None = None
    student_request_model: str | None = None
    student_api_key: str = "EMPTY"
    student_mode: str = "chat"
    student_temperature: float = 0.7
    student_top_p: float = 1.0
    student_max_tokens: int = 1536
    seed: int | None = None
    dry_run: bool = False
    execution_stage: str = "all"
    prepared_run_dir: Path | None = None
    bf16: bool = True
    use_lora: bool = True
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_model: str | None = None
    soda_beta: float = 0.1
    soda_learning_rate: float = 5e-6
    soda_epochs: float = 1.0
    soda_per_device_train_batch_size: int = 1
    soda_gradient_accumulation_steps: int = 32
    soda_max_length: int = 3584
    soda_max_prompt_length: int = 1024
    soda_max_grad_norm: float = 1.0
    soda_nonfinite_gradient_retries: int = 3
    soda_student_negatives_jsonl: Path | None = None
    soda_preferences_jsonl: Path | None = None
    qedks_max_followups_per_answer: int = 4
    qedks_use_ppl_schedule: bool = False
    qedks_ppl_device: str = "cuda"
    qedks_learning_rate: float = 2e-4
    qedks_epochs: float = 2.0
    qedks_lora_r: int = 16
    qedks_lora_alpha: int = 32
    qedks_lora_dropout: float = 0.05
    qedks_per_device_train_batch_size: int = 1
    qedks_gradient_accumulation_steps: int = 16
    model_leeching_learning_rate: float = 2e-4
    model_leeching_epochs: float = 2.0
    model_leeching_lora_r: int = 16
    model_leeching_lora_alpha: int = 32
    model_leeching_lora_dropout: float = 0.05
    model_leeching_per_device_train_batch_size: int = 1
    model_leeching_gradient_accumulation_steps: int = 16
    gad_group_size: int = 8
    gad_kl_beta: float = 0.001
    gad_learning_rate: float = 1e-6
    gad_discriminator_learning_rate: float = 1e-6
    gad_warmup_learning_rate: float = 5e-6
    gad_warmup_epochs: float = 1.0
    gad_epochs: float = 2.0
    gad_discriminator_warmup_steps: int = 10
    gad_max_steps: int = -1
    gad_per_device_train_batch_size: int = 1
    gad_gradient_accumulation_steps: int = 1
    gad_max_seq_length: int = 3584
    gad_max_prompt_length: int = 2048
    gad_max_response_length: int = 1536
    gad_temperature: float = 0.8
    gad_top_p: float = 1.0
    gad_offload_inactive_models: bool = False
    gad_memory_log_interval: int = 25
    gad_generator_device: str | None = None
    gad_discriminator_device: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


@dataclass(frozen=True)
class AttackResult:
    attack: str
    budget: int
    run_id: str
    output_dir: Path
    checkpoint_dir: Path | None
    status: str
    manifest_path: Path | None = None
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "attack": self.attack,
            "budget": self.budget,
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "checkpoint_dir": None if self.checkpoint_dir is None else str(self.checkpoint_dir),
            "status": self.status,
            "manifest_path": None if self.manifest_path is None else str(self.manifest_path),
            "artifacts": dict(self.artifacts),
            "metrics": dict(self.metrics),
        }


class BaseAttacker(Protocol):
    name: str

    def run(self, config: AttackRunConfig, *, run_id: str, run_dir: Path) -> AttackResult:
        """Run one attack and return its student checkpoint location."""
        ...
