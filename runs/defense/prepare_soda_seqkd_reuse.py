#!/usr/bin/env python3
"""Resolve the matching SeqKD defense artifacts used to warm-start SODA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHODS = ("ads", "doge", "trace_rewriting", "adfp", "ginsew", "radioactivity")
DETECTOR_METHODS = {"adfp", "ginsew", "radioactivity"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def aligned_query(row: dict[str, object]) -> tuple[str, str]:
    item_id = row.get("query_id") or row.get("prompt_id") or row.get("id")
    prompt = row.get("query") or row.get("prompt") or row.get("prompt_text")
    if item_id is None or prompt is None:
        raise ValueError(f"Row lacks a query id or prompt: {sorted(row)}")
    return str(item_id), str(prompt)


def first_existing(root: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"None of the expected paths exist under {root}: {candidates}")


def first_optional(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path.resolve()
    return None


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def align_negative_rows(
    transcript_data: list[dict[str, object]],
    negative_data: list[dict[str, object]],
) -> list[dict[str, object]]:
    transcript_queries = [aligned_query(row) for row in transcript_data]
    usable_negatives = negative_data[:len(transcript_data)]
    negative_queries = [aligned_query(row) for row in usable_negatives]
    if len(negative_queries) != len(transcript_queries):
        raise ValueError("SODA negatives contain fewer rows than the SeqKD transcript")
    if [prompt for _, prompt in transcript_queries] != [prompt for _, prompt in negative_queries]:
        mismatch = next(
            (i for i, pair in enumerate(zip(transcript_queries, negative_queries, strict=True)) if pair[0][1] != pair[1][1]),
            None,
        )
        raise ValueError(f"SeqKD transcript and SODA negative prompts are not aligned; first mismatch={mismatch}")
    transcript_ids = [item_id for item_id, _ in transcript_queries]
    if len(set(transcript_ids)) != len(transcript_ids):
        raise ValueError("SeqKD defended transcript contains duplicate query ids")
    aligned_rows = []
    for (teacher_id, _), negative in zip(transcript_queries, usable_negatives, strict=True):
        row = dict(negative)
        original_id = row.get("prompt_id") or row.get("query_id") or row.get("id")
        row["source_prompt_id"] = original_id
        row["prompt_id"] = teacher_id
        aligned_rows.append(row)
    return aligned_rows


def validate_checkpoint(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    adapter_config = checkpoint / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(f"Missing SeqKD adapter checkpoint: {checkpoint}")
    if not (checkpoint / "adapter_model.safetensors").is_file() and not (checkpoint / "adapter_model.bin").is_file():
        raise FileNotFoundError(f"Missing SeqKD adapter weights: {checkpoint}")
    adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
    if adapter.get("base_model_name_or_path") != "Qwen/Qwen2.5-7B":
        raise ValueError(
            f"Unexpected SeqKD base model in {checkpoint}: "
            f"{adapter.get('base_model_name_or_path')}"
        )
    return checkpoint


def _unique_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.exists() and resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _local_method_roots(method: str, storage_root: Path, explicit_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    if explicit_root is not None:
        roots.extend((explicit_root, explicit_root / method))

    defense_outputs = storage_root / "outputs" / "defenses"
    roots.extend((
        defense_outputs / "seqkd_b1000" / method,
        storage_root / "inputs" / "seqkd_defenses" / method,
        storage_root / "inputs" / "mea-seqkd-defenses" / method,
    ))
    if defense_outputs.is_dir():
        roots.extend(path / method for path in sorted(defense_outputs.glob("seqkd_b1000*")))

    inputs = storage_root / "inputs"
    if inputs.is_dir():
        roots.extend(path / method for path in sorted(inputs.glob("*/mea-seqkd-defenses")))
        roots.extend(path / method for path in sorted(inputs.glob("mea-seqkd-defenses*")))
    return _unique_existing(roots)


def _manifest_paths(method_root: Path) -> dict[str, Path]:
    """Read usable paths from a completed local defense manifest when available."""
    for manifest in (
        method_root / "defense_run_manifest.json",
        method_root / "full" / "defense_run_manifest.json",
    ):
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        generator = payload.get("generator_result") or {}
        candidates = {
            "checkpoint": generator.get("student_checkpoint_path"),
            "transcript": generator.get("defended_transcript_path"),
            "artifacts": generator.get("defense_artifacts_dir"),
            "query_log": generator.get("teacher_query_log_path"),
            "oracle_manifest": generator.get("oracle_manifest_path"),
        }
        result: dict[str, Path] = {}
        for key, value in candidates.items():
            if value and Path(value).expanduser().exists():
                result[key] = Path(value).expanduser().resolve()
        if "oracle_manifest" not in result:
            result["oracle_manifest"] = manifest.resolve()
        return result
    return {}


def find_local_seqkd_bundle(
    method: str,
    storage_root: Path,
    explicit_root: Path | None = None,
) -> dict[str, Path] | None:
    for method_root in _local_method_roots(method, storage_root, explicit_root):
        from_manifest = _manifest_paths(method_root)
        checkpoint = from_manifest.get("checkpoint") or first_optional(method_root, (
            "checkpoint-final",
            "full/checkpoint-final",
        ))
        if checkpoint is None:
            checkpoints = sorted(method_root.glob("attack/seqkd/**/checkpoint-final"))
            checkpoint = checkpoints[-1].resolve() if checkpoints else None

        transcript = from_manifest.get("transcript") or first_optional(method_root, (
            "oracle/transcripts/defended_teacher.jsonl",
            "oracle/defended_teacher_transcript.jsonl",
            "full/oracle/transcripts/defended_teacher.jsonl",
            "full/oracle/defended_teacher_transcript.jsonl",
        ))
        artifacts = from_manifest.get("artifacts") or first_optional(method_root, (
            "oracle/artifacts",
            "full/oracle/artifacts",
        ))
        query_log = from_manifest.get("query_log") or first_optional(method_root, (
            "oracle/teacher_received_queries.jsonl",
            "full/oracle/teacher_received_queries.jsonl",
        ))
        oracle_manifest = from_manifest.get("oracle_manifest") or first_optional(method_root, (
            "oracle/defense_manifest.json",
            "full/oracle/defense_manifest.json",
            "full/defense_run_manifest.json",
            "defense_run_manifest.json",
        ))
        required = (checkpoint, transcript, query_log, oracle_manifest)
        if any(path is None or not path.exists() for path in required):
            continue
        if method in DETECTOR_METHODS and (artifacts is None or not artifacts.is_dir()):
            continue
        try:
            checkpoint = validate_checkpoint(checkpoint)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
        return {
            "root": method_root,
            "checkpoint": checkpoint,
            "transcript": transcript,
            "artifacts": artifacts or transcript.parent / "artifacts",
            "query_log": query_log,
            "oracle_manifest": oracle_manifest,
        }
    return None


def _negative_candidates(storage_root: Path, explicit_path: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.extend((
        storage_root / "outputs" / "defenses" / "soda_b1000" / "shared" / "student_negatives.jsonl",
        storage_root / "inputs" / "soda_defenses" / "shared" / "student_negatives.jsonl",
        storage_root / "inputs" / "mea-soda-defenses" / "shared" / "student_negatives.jsonl",
    ))
    search_roots = (
        storage_root / "outputs" / "defenses" / "soda_b1000",
        storage_root / "outputs" / "attacks_full" / "soda",
        storage_root / "inputs",
    )
    for root in search_roots:
        if root.is_dir():
            candidates.extend(sorted(root.rglob("student_negatives.jsonl")))
    return _unique_existing(candidates)


def find_local_negatives(
    transcript_data: list[dict[str, object]],
    storage_root: Path,
    explicit_path: Path | None = None,
) -> tuple[Path, list[dict[str, object]]] | None:
    for path in _negative_candidates(storage_root, explicit_path):
        try:
            rows = read_jsonl(path)
            if len(rows) < len(transcript_data):
                continue
            aligned = align_negative_rows(transcript_data, rows)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        return path, aligned
    return None


def resolve_artifacts(
    method: str,
    aligned_negatives: Path,
    storage_root: Path,
    seqkd_local_root: Path | None = None,
    student_negatives: Path | None = None,
) -> dict[str, object]:
    bundle = find_local_seqkd_bundle(method, storage_root, seqkd_local_root)
    if bundle is None:
        raise FileNotFoundError(
            f"No local SeqKD defense bundle found for {method}; "
            "pass --seqkd-local-root or place it under STORAGE_ROOT."
        )

    transcript = bundle["transcript"]
    transcript_data = read_jsonl(transcript)
    transcript_rows = len(transcript_data)
    if transcript_rows != 1000:
        raise ValueError(f"Expected 1000 transcript rows; got {transcript_rows} from {transcript}")

    local_negatives = find_local_negatives(transcript_data, storage_root, student_negatives)
    if local_negatives is None:
        raise FileNotFoundError(
            "No local SODA student negatives found; pass --student-negatives "
            "or place them under STORAGE_ROOT."
        )
    negatives, aligned_rows = local_negatives
    negative_data = read_jsonl(negatives)
    negative_rows = len(negative_data)
    write_jsonl(aligned_negatives, aligned_rows)
    return {
        "method": method,
        "seqkd_source": "local",
        "seqkd_local_root": str(bundle["root"]),
        "student_negatives_source": "local",
        "warmup_checkpoint": str(bundle["checkpoint"]),
        "defended_transcript": str(transcript),
        "transcript_sha256": sha256(transcript),
        "transcript_rows": transcript_rows,
        "defense_artifacts": str(bundle["artifacts"]),
        "oracle_manifest": str(bundle["oracle_manifest"]),
        "teacher_query_log": str(bundle["query_log"]),
        "student_negatives": str(aligned_negatives.resolve()),
        "source_student_negatives": str(negatives),
        "student_negative_rows": negative_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--storage-root", required=True, type=Path)
    parser.add_argument("--seqkd-local-root", type=Path)
    parser.add_argument("--student-negatives", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    aligned_negatives = args.output.parent / "student_negatives_aligned.jsonl"
    result = resolve_artifacts(
        args.method,
        aligned_negatives,
        args.storage_root.expanduser().resolve(),
        args.seqkd_local_root,
        args.student_negatives,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(args.output.resolve()),
        "seqkd_source": result["seqkd_source"],
        "student_negatives_source": result["student_negatives_source"],
        "warmup_checkpoint": result["warmup_checkpoint"],
        "source_student_negatives": result["source_student_negatives"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
