#!/usr/bin/env python3
"""Resolve local inputs for the corrected clean Qwen SODA baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from runs.defense.prepare_soda_seqkd_reuse import read_jsonl, validate_checkpoint


CLEAN_PREFERENCE_ROWS = 999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_preferences(path: Path) -> int:
    rows = read_jsonl(path)
    if len(rows) != CLEAN_PREFERENCE_ROWS:
        raise ValueError(
            f"Expected {CLEAN_PREFERENCE_ROWS} published clean SODA preferences, "
            f"got {len(rows)} from {path}"
        )
    for index, row in enumerate(rows):
        if not all(isinstance(row.get(key), str) and row[key] for key in ("prompt", "chosen", "rejected")):
            raise ValueError(f"Invalid preference row {index} in {path}")
    return len(rows)


def first_local_checkpoint(storage: Path, explicit: Path | None) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    attack_root = storage / "outputs" / "attacks_full" / "seqkd"
    if attack_root.is_dir():
        candidates.extend(sorted(attack_root.rglob("checkpoint-final"), reverse=True))
    inputs = storage / "inputs"
    if inputs.is_dir():
        candidates.extend(path for path in sorted(inputs.rglob("checkpoint-final"), reverse=True) if "clean" in path.parts)
    for candidate in candidates:
        try:
            return validate_checkpoint(candidate)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
    return None


def first_local_preferences(storage: Path, explicit: Path | None) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    for root in (storage / "outputs" / "attacks_full" / "soda", storage / "inputs"):
        if root.is_dir():
            candidates.extend(
                path for path in sorted(root.rglob("preferences.jsonl"), reverse=True)
                if "clean" in str(path).lower()
            )
    for candidate in candidates:
        try:
            valid_preferences(candidate)
            return candidate.expanduser().resolve()
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", required=True, type=Path)
    parser.add_argument("--seqkd-checkpoint", type=Path)
    parser.add_argument("--preferences", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    storage = args.storage_root.expanduser().resolve()
    checkpoint = first_local_checkpoint(storage, args.seqkd_checkpoint)
    if checkpoint is None:
        raise FileNotFoundError(
            "No local SeqKD checkpoint found; pass --seqkd-checkpoint or place it under STORAGE_ROOT."
        )

    preferences = first_local_preferences(storage, args.preferences)
    if preferences is None:
        raise FileNotFoundError(
            "No local SODA preferences found; pass --preferences or place them under STORAGE_ROOT."
        )

    result = {
        "warmup_checkpoint": str(checkpoint),
        "warmup_source": "local",
        "preferences": str(preferences),
        "preferences_source": "local",
        "preferences_sha256": sha256(preferences),
        "preferences_rows": valid_preferences(preferences),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
