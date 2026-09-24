from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DetectorResult:
    detector: str
    status: str
    output_dir: Optional[Path] = None
    report_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    log_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, dict):
                payload[key] = {k: str(v) if isinstance(v, Path) else v for k, v in value.items()}
        return payload
