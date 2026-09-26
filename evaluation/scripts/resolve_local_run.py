from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")
    return payload


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def infer_identity_from_path(path: Path) -> tuple[str | None, int | None, str | None]:
    """Recover the canonical defense/attack/budget tuple from an output path."""
    parts = path.parts
    for index, part in enumerate(parts):
        if not (part.startswith("b") and part[1:].isdigit()):
            continue
        attack = parts[index - 1] if index >= 1 else None
        defense = parts[index - 2] if index >= 2 else None
        return attack, int(part[1:]), defense
    return None, None, None


def local_path(value: Any, manifest: Path, storage: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value).expanduser()
    candidates = [raw, manifest.parent / raw]
    if "outputs" in raw.parts:
        suffix = raw.parts[raw.parts.index("outputs") :]
        candidates.extend((storage.joinpath(*suffix), manifest.parents[0].joinpath(*suffix)))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def checkpoint_complete(path: Path) -> bool:
    if (path / "adapter_config.json").is_file():
        return any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    if not (path / "config.json").is_file():
        return False
    return any(path.glob("model*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def resolve_checkpoint(manifest: Path, storage: Path, *values: Any) -> Path | None:
    for value in values:
        candidate = local_path(value, manifest, storage)
        if candidate and candidate.is_dir() and checkpoint_complete(candidate):
            return candidate
    candidates = []
    for config in manifest.parent.rglob("adapter_config.json"):
        if checkpoint_complete(config.parent):
            candidates.append(config.parent.resolve())
    for config in manifest.parent.rglob("config.json"):
        if checkpoint_complete(config.parent):
            candidates.append(config.parent.resolve())
    unique = sorted(set(candidates), key=str)
    return unique[0] if len(unique) == 1 else None


def resolve_report(manifest: Path, storage: Path, *values: Any) -> Path | None:
    for value in values:
        candidate = local_path(value, manifest, storage)
        if candidate and candidate.is_file():
            return candidate
    names = ("detector_report.json", "comparison_report.json", "summary.json")
    candidates = [path.resolve() for name in names for path in manifest.parent.rglob(name)]
    unique = sorted(set(candidates), key=str)
    return unique[0] if len(unique) == 1 else None


def resolve(manifest_path: Path, storage: Path) -> dict[str, Any]:
    manifest = manifest_path.expanduser().resolve()
    payload = read_json(manifest)
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    config = mapping(payload.get("run_config")) or mapping(payload.get("config"))
    generator = payload.get("generator_result") if isinstance(payload.get("generator_result"), dict) else {}
    detector = payload.get("detector_result") if isinstance(payload.get("detector_result"), dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    generator_metadata = mapping(generator.get("metadata"))
    generator_attack = mapping(generator_metadata.get("attack"))
    path_attack, path_budget, path_defense = infer_identity_from_path(manifest)

    attack = result.get("attack") or generator_attack.get("attack") or payload.get("attack") or config.get("attack") or path_attack
    budget = result.get("budget") or payload.get("budget") or config.get("budget") or generator_attack.get("budget") or path_budget
    defense = payload.get("defense") or path_defense
    run_id = result.get("run_id") or payload.get("run_id") or manifest.parent.name
    schema = str(payload.get("schema_version") or "")

    comparison = local_path(metadata.get("comparison") or generator_metadata.get("comparison"), manifest, storage)
    countermeasure = generator_metadata.get("countermeasure") or metadata.get("countermeasure")
    if comparison or countermeasure not in (None, False, "", "none"):
        kind = "adaptive_attack"
    elif generator:
        kind = "defense"
    elif payload.get("detector") or payload.get("output_report") or manifest.name == "detector_manifest.json":
        kind = "detector"
        defense = payload.get("detector") or defense
    elif schema.startswith("attack_manifest") or result.get("attack"):
        kind = "attack"
    else:
        kind = str(payload.get("run_type") or "unknown")

    if kind == "detector":
        source_run_id = payload.get("attack_run_id") or run_id
        run_id = f"{defense}_{attack}_b{budget}_{source_run_id}"

    checkpoint = resolve_checkpoint(
        manifest,
        storage,
        result.get("checkpoint_dir"),
        payload.get("checkpoint_dir"),
        generator.get("student_checkpoint_path"),
        payload.get("student_checkpoint"),
    )
    report = resolve_report(
        manifest,
        storage,
        payload.get("output_report"),
        detector.get("report_path"),
        metadata.get("comparison"),
        generator_metadata.get("comparison"),
    )

    if kind in {"attack", "defense"} and checkpoint is None:
        raise FileNotFoundError(f"No unique complete local checkpoint could be resolved from {manifest}")
    if kind in {"adaptive_attack", "detector"} and report is None:
        raise FileNotFoundError(f"No unique local report could be resolved from {manifest}")
    if not attack or budget is None:
        raise ValueError(f"Manifest does not identify attack and budget: {manifest}")

    return {
        "manifest": str(manifest),
        "kind": kind,
        "attack": str(attack),
        "budget": int(budget),
        "defense": None if defense is None else str(defense),
        "run_id": str(run_id),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "report": None if report is None else str(report),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve local artifacts recorded by one benchmark run manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--storage-root", type=Path, default=Path(os.environ.get("STORAGE_ROOT", ".")))
    args = parser.parse_args()
    print(json.dumps(resolve(args.manifest, args.storage_root.expanduser().resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
