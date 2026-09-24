from __future__ import annotations

"""
======================================================================
STAGE1_TRANSCRIPT ---

Shared transcript, ordering, ledger, and manifest helpers for stage 1.

This module keeps teacher-oracle logic out of task-specific loaders and
stores only text-level artifacts.
======================================================================
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stage1_transcript_v1"
CODE_VERSION = "stage1_transcript_v1"
FORBIDDEN_RECORD_KEYS = {
    "teacher_logits",
    "logits",
    "token_probabilities",
    "top_k_probabilities",
    "hidden_states",
    "gradients",
}


class Stage1TranscriptError(RuntimeError):
    pass


class TranscriptSchemaError(Stage1TranscriptError):
    pass


class TranscriptConfigMismatchError(Stage1TranscriptError):
    pass


class TranscriptBudgetError(Stage1TranscriptError):
    pass


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_normalize_jsonable(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _normalize_jsonable(vars(value))
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "unknown"


def _nearest_git_commit(start_dir: Path) -> str:
    current = start_dir.resolve()
    for candidate in [current, *current.parents]:
        git_dir = candidate / ".git"
        if git_dir.exists():
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(candidate),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
            break
    return "unknown"


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    prompt_text: str


@dataclass(frozen=True)
class PromptOrdering:
    prompt_pool_id: str
    seed: int
    prompt_pool_hash: str
    ordering_hash: str
    ordered_prompt_ids: tuple[str, ...]
    ordered_indices: tuple[int, ...]

    def prefix(self, budget: int) -> "PromptOrdering":
        if budget < 0:
            raise TranscriptBudgetError("budget must be non-negative.")
        if budget > len(self.ordered_indices):
            raise TranscriptBudgetError(
                f"budget={budget} exceeds available prompts={len(self.ordered_indices)}."
            )
        return PromptOrdering(
            prompt_pool_id=self.prompt_pool_id,
            seed=self.seed,
            prompt_pool_hash=self.prompt_pool_hash,
            ordering_hash=self.ordering_hash,
            ordered_prompt_ids=self.ordered_prompt_ids[:budget],
            ordered_indices=self.ordered_indices[:budget],
        )


@dataclass(frozen=True)
class DecodeConfig:
    strategy: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        return stable_hash({"strategy": self.strategy, "settings": self.settings})


@dataclass(frozen=True)
class TranscriptRecord:
    prompt_id: str
    prompt_text: str
    teacher_response: str
    teacher_model_id: str
    decode_config: Mapping[str, Any]
    seed: int
    order_index: int
    query_id: str
    query_status: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "teacher_response": self.teacher_response,
            "teacher_model_id": self.teacher_model_id,
            "decode_config": dict(self.decode_config),
            "seed": self.seed,
            "order_index": self.order_index,
            "query_id": self.query_id,
            "query_status": self.query_status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TranscriptRecord":
        validate_transcript_record(payload)
        return cls(
            prompt_id=str(payload["prompt_id"]),
            prompt_text=str(payload["prompt_text"]),
            teacher_response=str(payload["teacher_response"]),
            teacher_model_id=str(payload["teacher_model_id"]),
            decode_config=dict(payload["decode_config"]),
            seed=int(payload["seed"]),
            order_index=int(payload["order_index"]),
            query_id=str(payload["query_id"]),
            query_status=str(payload["query_status"]),
            created_at=str(payload["created_at"]),
        )


@dataclass(frozen=True)
class QueryAttempt:
    query_id: str
    prompt_id: str
    order_index: int
    attempt_index: int
    status: str
    created_at: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "query_id": self.query_id,
            "prompt_id": self.prompt_id,
            "order_index": self.order_index,
            "attempt_index": self.attempt_index,
            "status": self.status,
            "created_at": self.created_at,
        }
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        return payload


@dataclass
class QueryLedger:
    successful_queries: int = 0
    api_attempts: int = 0
    failed_attempts: int = 0
    attempts: list[QueryAttempt] = field(default_factory=list)

    def record_success(self, attempt: QueryAttempt) -> None:
        self.api_attempts += 1
        self.successful_queries += 1
        self.attempts.append(attempt)

    def record_failure(self, attempt: QueryAttempt) -> None:
        self.api_attempts += 1
        self.failed_attempts += 1
        self.attempts.append(attempt)

    def summary(self) -> dict[str, int]:
        return {
            "successful_queries": self.successful_queries,
            "api_attempts": self.api_attempts,
            "failed_attempts": self.failed_attempts,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "successful_queries": self.successful_queries,
            "api_attempts": self.api_attempts,
            "failed_attempts": self.failed_attempts,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QueryLedger":
        attempts = [
            QueryAttempt(
                query_id=str(item["query_id"]),
                prompt_id=str(item["prompt_id"]),
                order_index=int(item["order_index"]),
                attempt_index=int(item["attempt_index"]),
                status=str(item["status"]),
                created_at=str(item["created_at"]),
                error_message=(
                    None if item.get("error_message") is None else str(item["error_message"])
                ),
            )
            for item in payload.get("attempts", [])
        ]
        return cls(
            successful_queries=int(payload.get("successful_queries", 0)),
            api_attempts=int(payload.get("api_attempts", 0)),
            failed_attempts=int(payload.get("failed_attempts", 0)),
            attempts=attempts,
        )


@dataclass(frozen=True)
class TranscriptManifest:
    schema_version: str
    code_version: str
    git_commit: str
    prompt_pool_id: str
    prompt_pool_hash: str
    prompt_ordering_hash: str
    transcript_hash: str
    teacher_model_id: str
    decode_config: Mapping[str, Any]
    seed: int
    requested_budget: int
    actual_successful_queries: int
    api_attempts: int
    created_at: str
    bundle_id: str
    transcript_path: str
    ordering_path: str
    ledger_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "code_version": self.code_version,
            "git_commit": self.git_commit,
            "prompt_pool_id": self.prompt_pool_id,
            "prompt_pool_hash": self.prompt_pool_hash,
            "prompt_ordering_hash": self.prompt_ordering_hash,
            "transcript_hash": self.transcript_hash,
            "teacher_model_id": self.teacher_model_id,
            "decode_config": dict(self.decode_config),
            "seed": self.seed,
            "requested_budget": self.requested_budget,
            "actual_successful_queries": self.actual_successful_queries,
            "api_attempts": self.api_attempts,
            "created_at": self.created_at,
            "bundle_id": self.bundle_id,
            "transcript_path": self.transcript_path,
            "ordering_path": self.ordering_path,
            "ledger_path": self.ledger_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TranscriptManifest":
        expected_keys = {
            "schema_version",
            "code_version",
            "git_commit",
            "prompt_pool_id",
            "prompt_pool_hash",
            "prompt_ordering_hash",
            "transcript_hash",
            "teacher_model_id",
            "decode_config",
            "seed",
            "requested_budget",
            "actual_successful_queries",
            "api_attempts",
            "created_at",
            "bundle_id",
            "transcript_path",
            "ordering_path",
            "ledger_path",
        }
        missing = expected_keys.difference(payload.keys())
        if missing:
            raise TranscriptSchemaError(f"manifest is missing required keys: {sorted(missing)}")
        return cls(
            schema_version=str(payload["schema_version"]),
            code_version=str(payload["code_version"]),
            git_commit=str(payload["git_commit"]),
            prompt_pool_id=str(payload["prompt_pool_id"]),
            prompt_pool_hash=str(payload["prompt_pool_hash"]),
            prompt_ordering_hash=str(payload["prompt_ordering_hash"]),
            transcript_hash=str(payload["transcript_hash"]),
            teacher_model_id=str(payload["teacher_model_id"]),
            decode_config=dict(payload["decode_config"]),
            seed=int(payload["seed"]),
            requested_budget=int(payload["requested_budget"]),
            actual_successful_queries=int(payload["actual_successful_queries"]),
            api_attempts=int(payload["api_attempts"]),
            created_at=str(payload["created_at"]),
            bundle_id=str(payload["bundle_id"]),
            transcript_path=str(payload["transcript_path"]),
            ordering_path=str(payload["ordering_path"]),
            ledger_path=str(payload["ledger_path"]),
        )


@dataclass(frozen=True)
class TranscriptBundle:
    manifest: TranscriptManifest
    ordering: PromptOrdering
    records: tuple[TranscriptRecord, ...]
    ledger: QueryLedger

    def prefix(self, budget: int) -> "TranscriptBundle":
        if budget < 0:
            raise TranscriptBudgetError("budget must be non-negative.")
        if budget > len(self.records):
            raise TranscriptBudgetError(
                f"budget={budget} exceeds available transcript records={len(self.records)}."
            )
        return TranscriptBundle(
            manifest=self.manifest,
            ordering=self.ordering.prefix(budget),
            records=self.records[:budget],
            ledger=self.ledger,
        )

    def budget_summary(self) -> dict[str, int]:
        return self.ledger.summary()


def build_prompt_specs(
    prompt_texts: Sequence[str],
    prompt_ids: Sequence[str] | None = None,
) -> list[PromptSpec]:
    if prompt_ids is not None and len(prompt_ids) != len(prompt_texts):
        raise ValueError("prompt_ids and prompt_texts must have the same length.")
    specs: list[PromptSpec] = []
    for idx, prompt_text in enumerate(prompt_texts):
        prompt_id = prompt_ids[idx] if prompt_ids is not None else f"prompt_{idx:06d}"
        specs.append(PromptSpec(prompt_id=str(prompt_id), prompt_text=str(prompt_text)))
    return specs


def build_prompt_ordering(
    prompt_specs: Sequence[PromptSpec],
    seed: int,
    prompt_pool_id: str,
) -> PromptOrdering:
    index_list = list(range(len(prompt_specs)))
    random.Random(seed).shuffle(index_list)
    ordered_prompt_ids = tuple(prompt_specs[idx].prompt_id for idx in index_list)
    prompt_pool_hash = stable_hash(
        [(spec.prompt_id, spec.prompt_text) for spec in prompt_specs]
    )
    ordering_hash = stable_hash(
        {
            "prompt_pool_id": prompt_pool_id,
            "seed": seed,
            "prompt_pool_hash": prompt_pool_hash,
            "ordered_prompt_ids": ordered_prompt_ids,
        }
    )
    return PromptOrdering(
        prompt_pool_id=prompt_pool_id,
        seed=seed,
        prompt_pool_hash=prompt_pool_hash,
        ordering_hash=ordering_hash,
        ordered_prompt_ids=ordered_prompt_ids,
        ordered_indices=tuple(index_list),
    )


def prompt_specs_in_order(
    prompt_specs: Sequence[PromptSpec],
    ordering: PromptOrdering,
) -> list[PromptSpec]:
    return [prompt_specs[idx] for idx in ordering.ordered_indices]


def select_budget_prefix(
    ordered_prompt_specs: Sequence[PromptSpec],
    budget: int,
) -> list[PromptSpec]:
    if budget < 0:
        raise TranscriptBudgetError("budget must be non-negative.")
    if budget > len(ordered_prompt_specs):
        raise TranscriptBudgetError(
            f"budget={budget} exceeds available prompts={len(ordered_prompt_specs)}."
        )
    return list(ordered_prompt_specs[:budget])


def validate_transcript_record(payload: Mapping[str, Any]) -> None:
    missing = {
        "prompt_id",
        "prompt_text",
        "teacher_response",
        "teacher_model_id",
        "decode_config",
        "seed",
        "order_index",
        "query_id",
        "query_status",
        "created_at",
    }.difference(payload.keys())
    if missing:
        raise TranscriptSchemaError(f"record is missing required keys: {sorted(missing)}")

    forbidden = FORBIDDEN_RECORD_KEYS.intersection(payload.keys())
    if forbidden:
        raise TranscriptSchemaError(
            f"record contains forbidden fields: {sorted(forbidden)}"
        )

    if not isinstance(payload["decode_config"], Mapping):
        raise TranscriptSchemaError("record.decode_config must be a mapping.")


def _record_line(record: TranscriptRecord) -> str:
    return _canonical_json(record.to_dict())


def transcript_hash_from_records(records: Sequence[TranscriptRecord]) -> str:
    joined = "\n".join(_record_line(record) for record in records)
    return stable_hash({"jsonl": joined})


def write_transcript_jsonl(
    path: Path,
    records: Sequence[TranscriptRecord],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(_record_line(record) for record in records)
    path.write_text(payload, encoding="utf-8")
    return stable_hash({"jsonl": payload})


def read_transcript_jsonl(path: Path) -> list[TranscriptRecord]:
    if not path.exists():
        raise FileNotFoundError(f"transcript file does not exist: {path}")
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []
    records: list[TranscriptRecord] = []
    # JSON strings may legally contain Unicode line separators such as U+2028
    # and U+2029. str.splitlines() treats those characters as record boundaries,
    # so split only on the ASCII newline used by write_transcript_jsonl().
    for line_number, line in enumerate(content.split("\n"), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise TranscriptSchemaError(f"line {line_number} is not a JSON object.")
        records.append(TranscriptRecord.from_dict(payload))
    return records


def _identity_payload(
    prompt_pool_id: str,
    prompt_pool_hash: str,
    prompt_ordering_hash: str,
    teacher_model_id: str,
    decode_config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "prompt_pool_id": prompt_pool_id,
        "prompt_pool_hash": prompt_pool_hash,
        "prompt_ordering_hash": prompt_ordering_hash,
        "teacher_model_id": teacher_model_id,
        "decode_config": decode_config,
        "seed": seed,
        "schema_version": SCHEMA_VERSION,
    }


def bundle_id_from_identity(
    prompt_pool_id: str,
    prompt_pool_hash: str,
    prompt_ordering_hash: str,
    teacher_model_id: str,
    decode_config: Mapping[str, Any],
    seed: int,
) -> str:
    return stable_hash(
        _identity_payload(
            prompt_pool_id,
            prompt_pool_hash,
            prompt_ordering_hash,
            teacher_model_id,
            decode_config,
            seed,
        )
    )[:24]


def resolve_bundle_dir(
    base_dir: Path,
    prompt_pool_id: str,
    prompt_pool_hash: str,
    prompt_ordering_hash: str,
    teacher_model_id: str,
    decode_config: Mapping[str, Any],
    seed: int,
) -> Path:
    bundle_id = bundle_id_from_identity(
        prompt_pool_id,
        prompt_pool_hash,
        prompt_ordering_hash,
        teacher_model_id,
        decode_config,
        seed,
    )
    return (
        base_dir
        / _sanitize_component(prompt_pool_id)
        / _sanitize_component(teacher_model_id)
        / bundle_id
    )


def _manifest_paths(bundle_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        bundle_dir / "manifest.json",
        bundle_dir / "ordering.json",
        bundle_dir / "transcript.jsonl",
        bundle_dir / "ledger.json",
    )


def save_transcript_bundle(
    bundle_dir: Path,
    prompt_pool_id: str,
    ordering: PromptOrdering,
    records: Sequence[TranscriptRecord],
    teacher_model_id: str,
    decode_config: Mapping[str, Any],
    seed: int,
    requested_budget: int,
    ledger: QueryLedger,
    prompt_pool_hash: str | None = None,
) -> TranscriptManifest:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, ordering_path, transcript_path, ledger_path = _manifest_paths(bundle_dir)

    transcript_hash = write_transcript_jsonl(transcript_path, records)
    prompt_pool_hash = prompt_pool_hash or ordering.prompt_pool_hash
    bundle_id = bundle_id_from_identity(
        prompt_pool_id,
        prompt_pool_hash,
        ordering.ordering_hash,
        teacher_model_id,
        decode_config,
        seed,
    )
    manifest = TranscriptManifest(
        schema_version=SCHEMA_VERSION,
        code_version=CODE_VERSION,
        git_commit=git_commit(),
        prompt_pool_id=prompt_pool_id,
        prompt_pool_hash=prompt_pool_hash,
        prompt_ordering_hash=ordering.ordering_hash,
        transcript_hash=transcript_hash,
        teacher_model_id=teacher_model_id,
        decode_config=dict(decode_config),
        seed=seed,
        requested_budget=requested_budget,
        actual_successful_queries=ledger.successful_queries,
        api_attempts=ledger.api_attempts,
        created_at=_utc_now_iso(),
        bundle_id=bundle_id,
        transcript_path=str(transcript_path),
        ordering_path=str(ordering_path),
        ledger_path=str(ledger_path),
    )
    ordering_payload = {
        "prompt_pool_id": ordering.prompt_pool_id,
        "seed": ordering.seed,
        "prompt_pool_hash": ordering.prompt_pool_hash,
        "ordering_hash": ordering.ordering_hash,
        "ordered_prompt_ids": list(ordering.ordered_prompt_ids),
        "ordered_indices": list(ordering.ordered_indices),
    }
    ordering_path.write_text(_canonical_json(ordering_payload), encoding="utf-8")
    ledger_path.write_text(_canonical_json(ledger.to_dict()), encoding="utf-8")
    manifest_path.write_text(_canonical_json(manifest.to_dict()), encoding="utf-8")
    return manifest


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TranscriptSchemaError(f"file is not a JSON object: {path}")
    return dict(payload)


def load_transcript_bundle(
    bundle_dir: Path,
    expected_prompt_pool_id: str | None = None,
    expected_prompt_pool_hash: str | None = None,
    expected_prompt_ordering_hash: str | None = None,
    expected_teacher_model_id: str | None = None,
    expected_decode_config: Mapping[str, Any] | None = None,
    expected_seed: int | None = None,
    budget: int | None = None,
) -> TranscriptBundle:
    manifest_path, ordering_path, transcript_path, ledger_path = _manifest_paths(bundle_dir)
    manifest = TranscriptManifest.from_dict(_read_json_file(manifest_path))
    ordering_payload = _read_json_file(ordering_path)
    ordering = PromptOrdering(
        prompt_pool_id=str(ordering_payload["prompt_pool_id"]),
        seed=int(ordering_payload["seed"]),
        prompt_pool_hash=str(ordering_payload["prompt_pool_hash"]),
        ordering_hash=str(ordering_payload["ordering_hash"]),
        ordered_prompt_ids=tuple(str(x) for x in ordering_payload["ordered_prompt_ids"]),
        ordered_indices=tuple(int(x) for x in ordering_payload["ordered_indices"]),
    )
    records = tuple(read_transcript_jsonl(transcript_path))
    ledger = QueryLedger.from_dict(_read_json_file(ledger_path))
    if transcript_hash_from_records(records) != manifest.transcript_hash:
        raise TranscriptSchemaError("transcript hash does not match manifest.")

    _validate_loaded_bundle(
        manifest=manifest,
        ordering=ordering,
        expected_prompt_pool_id=expected_prompt_pool_id,
        expected_prompt_pool_hash=expected_prompt_pool_hash,
        expected_prompt_ordering_hash=expected_prompt_ordering_hash,
        expected_teacher_model_id=expected_teacher_model_id,
        expected_decode_config=expected_decode_config,
        expected_seed=expected_seed,
    )

    if budget is not None:
        if budget > len(records):
            raise TranscriptBudgetError(
                f"budget={budget} exceeds available transcript records={len(records)}."
            )
        records = records[:budget]
        ordering = ordering.prefix(budget)

    if transcript_hash_from_records(records) != manifest.transcript_hash and budget is None:
        raise TranscriptSchemaError("transcript hash does not match manifest.")

    return TranscriptBundle(manifest=manifest, ordering=ordering, records=records, ledger=ledger)


def _validate_loaded_bundle(
    *,
    manifest: TranscriptManifest,
    ordering: PromptOrdering,
    expected_prompt_pool_id: str | None,
    expected_prompt_pool_hash: str | None,
    expected_prompt_ordering_hash: str | None,
    expected_teacher_model_id: str | None,
    expected_decode_config: Mapping[str, Any] | None,
    expected_seed: int | None,
) -> None:
    if expected_prompt_pool_id is not None and manifest.prompt_pool_id != expected_prompt_pool_id:
        raise TranscriptConfigMismatchError(
            f"prompt_pool_id mismatch: manifest={manifest.prompt_pool_id!r}, expected={expected_prompt_pool_id!r}"
        )
    if expected_prompt_pool_hash is not None and manifest.prompt_pool_hash != expected_prompt_pool_hash:
        raise TranscriptConfigMismatchError(
            "prompt_pool_hash mismatch: "
            f"manifest={manifest.prompt_pool_hash!r}, expected={expected_prompt_pool_hash!r}"
        )
    if expected_prompt_ordering_hash is not None and manifest.prompt_ordering_hash != expected_prompt_ordering_hash:
        raise TranscriptConfigMismatchError(
            "prompt_ordering_hash mismatch: "
            f"manifest={manifest.prompt_ordering_hash!r}, expected={expected_prompt_ordering_hash!r}"
        )
    if expected_teacher_model_id is not None and manifest.teacher_model_id != expected_teacher_model_id:
        raise TranscriptConfigMismatchError(
            f"teacher_model_id mismatch: manifest={manifest.teacher_model_id!r}, expected={expected_teacher_model_id!r}"
        )
    if expected_seed is not None and manifest.seed != expected_seed:
        raise TranscriptConfigMismatchError(
            f"seed mismatch: manifest={manifest.seed!r}, expected={expected_seed!r}"
        )
    if expected_decode_config is not None and dict(manifest.decode_config) != dict(expected_decode_config):
        raise TranscriptConfigMismatchError("decode_config mismatch between manifest and expected config.")
    if manifest.prompt_ordering_hash != ordering.ordering_hash:
        raise TranscriptSchemaError("manifest/orderings disagree on ordering hash.")


def collect_transcript_bundle(
    prompt_specs: Sequence[PromptSpec],
    *,
    prompt_pool_id: str,
    seed: int,
    teacher_model_id: str,
    decode_config: DecodeConfig,
    requested_budget: int,
    oracle: Callable[[str], str],
    base_dir: Path | None = None,
    max_retries: int = 0,
    force_regenerate: bool = False,
) -> TranscriptBundle:
    if requested_budget < 0:
        raise TranscriptBudgetError("requested_budget must be non-negative.")
    if requested_budget > len(prompt_specs):
        raise TranscriptBudgetError(
            f"requested_budget={requested_budget} exceeds available prompts={len(prompt_specs)}."
        )
    base_dir = base_dir or default_transcript_root()
    ordering = build_prompt_ordering(prompt_specs, seed=seed, prompt_pool_id=prompt_pool_id)
    bundle_dir = resolve_bundle_dir(
        base_dir=base_dir,
        prompt_pool_id=prompt_pool_id,
        prompt_pool_hash=ordering.prompt_pool_hash,
        prompt_ordering_hash=ordering.ordering_hash,
        teacher_model_id=teacher_model_id,
        decode_config={"strategy": decode_config.strategy, "settings": dict(decode_config.settings)},
        seed=seed,
    )
    if bundle_dir.exists() and not force_regenerate:
        loaded = load_transcript_bundle(
            bundle_dir,
            expected_prompt_pool_id=prompt_pool_id,
            expected_prompt_pool_hash=ordering.prompt_pool_hash,
            expected_prompt_ordering_hash=ordering.ordering_hash,
            expected_teacher_model_id=teacher_model_id,
            expected_decode_config={"strategy": decode_config.strategy, "settings": dict(decode_config.settings)},
            expected_seed=seed,
        )
        if len(loaded.records) >= requested_budget:
            return loaded.prefix(requested_budget)
        raise TranscriptBudgetError(
            f"existing transcript only has {len(loaded.records)} records, cannot satisfy budget={requested_budget}."
        )

    selected_specs = select_budget_prefix(prompt_specs_in_order(prompt_specs, ordering), requested_budget)
    ledger = QueryLedger()
    records: list[TranscriptRecord] = []
    for order_index, spec in enumerate(selected_specs):
        query_id = f"{ordering.ordering_hash[:12]}:{order_index:06d}"
        attempt_index = 0
        while True:
            attempt_index += 1
            try:
                teacher_response = oracle(spec.prompt_text)
                ledger.record_success(
                    QueryAttempt(
                        query_id=query_id,
                        prompt_id=spec.prompt_id,
                        order_index=order_index,
                        attempt_index=attempt_index,
                        status="success",
                        created_at=_utc_now_iso(),
                    )
                )
                records.append(
                    TranscriptRecord(
                        prompt_id=spec.prompt_id,
                        prompt_text=spec.prompt_text,
                        teacher_response=teacher_response,
                        teacher_model_id=teacher_model_id,
                        decode_config={"strategy": decode_config.strategy, "settings": dict(decode_config.settings)},
                        seed=seed,
                        order_index=order_index,
                        query_id=query_id,
                        query_status="success",
                        created_at=_utc_now_iso(),
                    )
                )
                break
            except Exception as exc:  # pragma: no cover - surfaced through tests via success path
                ledger.record_failure(
                    QueryAttempt(
                        query_id=query_id,
                        prompt_id=spec.prompt_id,
                        order_index=order_index,
                        attempt_index=attempt_index,
                        status="failed",
                        created_at=_utc_now_iso(),
                        error_message=str(exc),
                    )
                )
                if attempt_index > max_retries:
                    raise
    manifest = save_transcript_bundle(
        bundle_dir=bundle_dir,
        prompt_pool_id=prompt_pool_id,
        ordering=ordering,
        records=records,
        teacher_model_id=teacher_model_id,
        decode_config={"strategy": decode_config.strategy, "settings": dict(decode_config.settings)},
        seed=seed,
        requested_budget=requested_budget,
        ledger=ledger,
        prompt_pool_hash=ordering.prompt_pool_hash,
    )
    return TranscriptBundle(manifest=manifest, ordering=ordering.prefix(requested_budget), records=tuple(records), ledger=ledger)


def default_transcript_root() -> Path:
    return Path(__file__).resolve().parents[1] / "stage1_artifacts" / "transcripts"


def transcript_bundle_summary(bundle: TranscriptBundle) -> dict[str, Any]:
    return {
        "manifest": bundle.manifest.to_dict(),
        "ordering_hash": bundle.ordering.ordering_hash,
        "transcript_hash": bundle.manifest.transcript_hash,
        "budget_summary": bundle.budget_summary(),
    }


def git_commit() -> str:
    return _nearest_git_commit(Path(__file__).resolve().parent)


def build_manifest_identity(
    *,
    prompt_pool_id: str,
    prompt_pool_hash: str,
    prompt_ordering_hash: str,
    teacher_model_id: str,
    decode_config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    return _identity_payload(
        prompt_pool_id=prompt_pool_id,
        prompt_pool_hash=prompt_pool_hash,
        prompt_ordering_hash=prompt_ordering_hash,
        teacher_model_id=teacher_model_id,
        decode_config=decode_config,
        seed=seed,
    )


__all__ = [
    "CODE_VERSION",
    "SCHEMA_VERSION",
    "DecodeConfig",
    "PromptOrdering",
    "PromptSpec",
    "QueryAttempt",
    "QueryLedger",
    "Stage1TranscriptError",
    "TranscriptബudgetError",
]

__all__ = [
    "CODE_VERSION",
    "SCHEMA_VERSION",
    "DecodeConfig",
    "PromptOrdering",
    "PromptSpec",
    "QueryAttempt",
    "QueryLedger",
    "Stage1TranscriptError",
    "TranscriptBudgetError",
    "TranscriptBundle",
    "TranscriptConfigMismatchError",
    "TranscriptManifest",
    "TranscriptRecord",
    "TranscriptSchemaError",
    "build_manifest_identity",
    "build_prompt_ordering",
    "build_prompt_specs",
    "bundle_id_from_identity",
    "collect_transcript_bundle",
    "default_transcript_root",
    "git_commit",
    "load_transcript_bundle",
    "prompt_specs_in_order",
    "read_transcript_jsonl",
    "resolve_bundle_dir",
    "save_transcript_bundle",
    "select_budget_prefix",
    "stable_hash",
    "transcript_bundle_summary",
    "transcript_hash_from_records",
    "validate_transcript_record",
    "write_transcript_jsonl",
]
