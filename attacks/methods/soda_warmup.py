from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from attacks.core.base import AttackRunConfig
from attacks.core.manifest import file_sha256


class SODAWarmupError(RuntimeError):
    pass


@dataclass(frozen=True)
class SODAWarmupResolution:
    checkpoint: str
    source: str
    manifest_path: str | None = None
    run_id: str | None = None
    transcript_dir: str | None = None


def _resolved_path(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def _checkpoint_complete(path: Path) -> bool:
    if (path / "adapter_config.json").is_file():
        return (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    if not (path / "config.json").is_file():
        return False
    return any(path.glob("model*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def _local_checkpoint_from_manifest(manifest_path: Path, manifest: dict[str, Any], budget: int) -> Path | None:
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    configured = result.get("checkpoint_dir")
    if configured:
        configured_path = Path(str(configured)).expanduser()
        if configured_path.is_dir() and _checkpoint_complete(configured_path):
            return configured_path.resolve()

    run_dir = manifest_path.parent
    candidates = [
        run_dir / "training" / "seqkd" / f"budget_{budget}" / "seqkd" / "checkpoint-final",
        run_dir / "checkpoint-final",
    ]
    for candidate in candidates:
        if candidate.is_dir() and _checkpoint_complete(candidate):
            return candidate.resolve()
    return None


def _teacher_bundle(manifest: dict[str, Any]) -> str | None:
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    teacher = artifacts.get("teacher_transcript") if isinstance(artifacts.get("teacher_transcript"), dict) else {}
    value = teacher.get("bundle_dir")
    if not value:
        run_config = manifest.get("run_config") if isinstance(manifest.get("run_config"), dict) else {}
        value = run_config.get("transcript_dir")
    return None if not value else _resolved_path(str(value))


def _teacher_transcript_matches(candidate_bundle: str | None, expected_bundle: str) -> bool:
    if candidate_bundle == expected_bundle:
        return True
    if candidate_bundle is None:
        return False
    candidate_jsonl = Path(candidate_bundle) / "transcript.jsonl"
    expected_jsonl = Path(expected_bundle) / "transcript.jsonl"
    return (
        candidate_jsonl.is_file()
        and expected_jsonl.is_file()
        and file_sha256(candidate_jsonl) == file_sha256(expected_jsonl)
    )


def _ordering_identity(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ordering_hash = payload.get("ordering_hash")
    return str(ordering_hash) if ordering_hash else None


def _candidate_ordering_identity(manifest: dict[str, Any]) -> str | None:
    bundle = _teacher_bundle(manifest)
    if bundle:
        identity = _ordering_identity(Path(bundle) / "ordering.json")
        if identity:
            return identity
    run_config = manifest.get("run_config") if isinstance(manifest.get("run_config"), dict) else {}
    ordering_path = run_config.get("query_ordering_path")
    return _ordering_identity(None if not ordering_path else Path(str(ordering_path)).expanduser())


def _validate_explicit_warmup(config: AttackRunConfig) -> SODAWarmupResolution:
    assert config.warmup_model is not None
    if config.student_model and config.warmup_model == config.student_model:
        raise SODAWarmupError(
            "SODA --warmup-model resolves to the untrained base student. Provide a completed SeqKD "
            "checkpoint, or omit --warmup-model to auto-discover a compatible SeqKD run."
        )
    path = Path(config.warmup_model).expanduser()
    if path.exists():
        if not path.is_dir() or not _checkpoint_complete(path):
            raise SODAWarmupError(f"Explicit SODA warmup checkpoint is incomplete: {path}")
        value = str(path.resolve())
    else:
        value = config.warmup_model
    return SODAWarmupResolution(checkpoint=value, source="explicit")


def resolve_soda_warmup(config: AttackRunConfig, *, transcript_dir: Path | None) -> SODAWarmupResolution:
    if config.warmup_model:
        return _validate_explicit_warmup(config)

    seqkd_root = config.output_dir / "seqkd"
    if not seqkd_root.is_dir():
        raise SODAWarmupError(
            f"No SeqKD output directory found at {seqkd_root}. Run SeqKD budget={config.budget} first "
            "or pass --warmup-model explicitly."
        )

    expected_pool_hash = file_sha256(config.query_pool_path)
    expected_ordering_hash = None if config.query_ordering_path is None else file_sha256(config.query_ordering_path)
    expected_ordering_identity = _ordering_identity(config.query_ordering_path)
    expected_transcript = None if transcript_dir is None else _resolved_path(transcript_dir)
    compatible: list[tuple[str, Path, dict[str, Any], Path]] = []
    rejected: list[str] = []

    manifests = sorted(seqkd_root.glob(f"*_seqkd_b{config.budget}/attack_manifest.json"), reverse=True)
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rejected.append(f"{manifest_path.parent.name}: unreadable manifest ({exc})")
            continue

        run_config = manifest.get("run_config") if isinstance(manifest.get("run_config"), dict) else {}
        result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
        reasons: list[str] = []
        if result.get("status") != "completed":
            reasons.append(f"status={result.get('status')!r}")
        if run_config.get("budget") != config.budget:
            reasons.append(f"budget={run_config.get('budget')!r}")
        if run_config.get("student_model") != config.student_model:
            reasons.append(f"student_model={run_config.get('student_model')!r}")
        if manifest.get("query_pool_sha256") != expected_pool_hash:
            reasons.append("query pool differs")
        candidate_ordering_identity = _candidate_ordering_identity(manifest)
        if expected_ordering_identity and candidate_ordering_identity:
            if candidate_ordering_identity != expected_ordering_identity:
                reasons.append("query ordering differs")
        elif manifest.get("query_ordering_sha256") != expected_ordering_hash:
            reasons.append("query ordering differs")
        candidate_transcript = _teacher_bundle(manifest)
        if candidate_transcript is None or not (Path(candidate_transcript) / "transcript.jsonl").is_file():
            reasons.append("teacher transcript missing")
        elif expected_transcript is not None and not _teacher_transcript_matches(candidate_transcript, expected_transcript):
            reasons.append("teacher transcript differs")

        checkpoint = _local_checkpoint_from_manifest(manifest_path, manifest, config.budget)
        if checkpoint is None:
            reasons.append("checkpoint missing or incomplete")
        if reasons:
            rejected.append(f"{manifest_path.parent.name}: {', '.join(reasons)}")
            continue
        compatible.append((str(manifest.get("created_at") or manifest_path.parent.name), manifest_path, manifest, checkpoint))

    if not compatible:
        detail = "\n  - ".join(rejected[:8]) if rejected else "no matching SeqKD manifests"
        raise SODAWarmupError(
            "SODA could not auto-discover a compatible completed SeqKD warmup. Required: same budget, "
            "base student, query pool/order, and teacher transcript, with a complete checkpoint.\n"
            f"Searched: {seqkd_root}\n  - {detail}"
        )

    _, manifest_path, manifest, checkpoint = max(compatible, key=lambda item: item[0])
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    return SODAWarmupResolution(
        checkpoint=str(checkpoint),
        source="auto_seqkd",
        manifest_path=str(manifest_path.resolve()),
        run_id=str(result.get("run_id") or manifest_path.parent.name),
        transcript_dir=_teacher_bundle(manifest),
    )


__all__ = ["SODAWarmupError", "SODAWarmupResolution", "resolve_soda_warmup"]
