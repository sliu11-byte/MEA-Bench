from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GeneratorResult:
    defense: str
    run_id: str
    output_dir: Path
    oracle_dir: Path
    oracle_base_url: str
    oracle_manifest_path: Path
    teacher_query_log_path: Path
    defended_transcript_path: Path
    defense_artifacts_dir: Path
    attack_output_dir: Path
    attack_manifest_path: Optional[Path]
    student_checkpoint_path: Optional[Path]
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, dict):
                payload[key] = {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in value.items()
                }
        return payload
