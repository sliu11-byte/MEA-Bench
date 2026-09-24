"""Shared defense / detector manifest schemas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]

_METHOD_CONFIG_KEY = {
    "adfp": "adfp_config",
    "ads": "ads_config",
    "doge": "doge_config",
    "ginsew": "ginsew_config",
    "radioactivity": "radioactivity_config",
    "trace_rewriting": "trace_rewriting_config",
}

# Matrix M4 accepts only these labels; aliases map onto them.
_POSITIVE_LABELS = {
    "positive",
    "pos",
    "1",
    "true",
    "watermarked",
    "wm",
    "radioactive",
    "defended",
    "ads",
    "adfp",
    "duffin",
    "mmd",
    "prada",
    "seat",
}
_NEGATIVE_LABELS = {
    "negative",
    "neg",
    "0",
    "false",
    "clean",
    "baseline",
    "unwatermarked",
    "bare_base",
    "none",
}

# Default Matrix ``direction`` per detector when not already set.
_DETECTOR_DIRECTION = {
    "adfp": "higher",
    "duffin": "higher",
    "ginsew": "higher",
    "mmd": "higher",
    "prada": "higher",
    "radioactivity": "lower",
    "seat": "higher",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_matrix_label(label: Any) -> Optional[str]:
    """Map common aliases to Matrix M4 ``positive`` / ``negative``."""
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in _POSITIVE_LABELS:
        return "positive"
    if s in _NEGATIVE_LABELS:
        return "negative"
    if s in {"positive", "negative"}:
        return s
    return None


def resolve_matrix_direction(
    *,
    detector: str,
    direction: Any = None,
    score_direction: Any = None,
) -> str:
    """Return Matrix M4 ``direction``: ``higher`` or ``lower``."""
    for raw in (direction, score_direction):
        if raw is None:
            continue
        s = str(raw).strip().lower()
        if s in {"higher", "lower"}:
            return s
        if "lower" in s:
            return "lower"
        if "higher" in s:
            return "higher"
    return _DETECTOR_DIRECTION.get(detector, "higher")


@dataclass
class DefenseManifest:
    defense: str
    type: str
    teacher_model: str
    query_pool: str
    output_transcript: str
    num_queries: int
    rewriter_model: Optional[str] = None
    input_clean_transcript: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with both ``config`` and method-specific alias keys."""
        data = asdict(self)
        key = _METHOD_CONFIG_KEY.get(self.defense)
        if key:
            data[key] = dict(self.config or {})
        return data

    def save(self, path: PathLike) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: PathLike) -> "DefenseManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        defense = data.get("defense")
        alias = _METHOD_CONFIG_KEY.get(defense or "", "")
        if not data.get("config") and alias and data.get(alias):
            data["config"] = data[alias]
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class DetectorManifest:
    detector: str
    defense_run_id: str
    attack_run_id: Optional[str]
    student_checkpoint: Optional[str]
    watermark_artifacts: Dict[str, str] = field(default_factory=dict)
    output_report: str = ""
    cost: Dict[str, Any] = field(default_factory=dict)
    probe_queries: Optional[str] = None
    num_probe_queries: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    students: List[Dict[str, Any]] = field(default_factory=list)
    student_outputs: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: PathLike) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: PathLike) -> "DetectorManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def finalize_detector_row(
    row: Dict[str, Any],
    *,
    detector: str,
    defense_run_id: Optional[str],
    attack_run_id: Optional[str] = None,
    row_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Stamp shared ids and Matrix-M4-compatible ``label`` / ``direction`` / ``score``.

    Emits fields Matrix ``m4_provenance.detector`` can read directly:
      id, label (positive|negative), score, direction (higher|lower), detector

    Also keeps defense-scope helpers: score_direction, model_id, defense_run_id,
    attack_run_id, and optional label_raw / p_value / green_*.
    """
    out = dict(row)
    model_id = (
        out.get("model_id")
        or out.get("student")
        or out.get("student_checkpoint")
        or out.get("student_outputs")
    )
    out["model_id"] = model_id
    out["detector"] = detector
    out["defense_run_id"] = defense_run_id
    out["attack_run_id"] = attack_run_id
    out["id"] = row_id or out.get("id") or f"{detector}:{model_id}"

    raw_label = out.get("label")
    normalized = normalize_matrix_label(raw_label)
    if raw_label is not None and normalized is not None and str(raw_label).strip().lower() != normalized:
        out["label_raw"] = raw_label
    if normalized is not None:
        out["label"] = normalized
    elif raw_label is not None:
        out["label_raw"] = raw_label

    # Radioactivity: Matrix primary score is p-value (lower = more radioactive).
    if detector == "radioactivity" and out.get("p_value") is not None:
        if "green_score" not in out and out.get("score") is not None:
            try:
                if abs(float(out["score"]) - float(out["p_value"])) > 1e-15:
                    out.setdefault("green_score", float(out["score"]))
            except (TypeError, ValueError):
                pass
        out["score"] = float(out["p_value"])

    direction = resolve_matrix_direction(
        detector=detector,
        direction=out.get("direction"),
        score_direction=out.get("score_direction"),
    )
    out["direction"] = direction
    # Defense-scope md uses score_direction with short higher/lower values.
    out["score_direction"] = direction
    return out
