from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from defenses.core.detector.result import DetectorResult
from defenses.core.generator.result import GeneratorResult


@dataclass(frozen=True)
class DefenseRunResult:
    defense: str
    run_id: str
    output_dir: Path
    generator_result: GeneratorResult
    detector_result: Optional[DetectorResult] = None
    status: str = "ok"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["generator_result"] = self.generator_result.to_manifest()
        payload["detector_result"] = None if self.detector_result is None else self.detector_result.to_manifest()
        return payload
