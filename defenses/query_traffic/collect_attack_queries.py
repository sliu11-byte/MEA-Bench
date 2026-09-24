"""Collect attack teacher-query logs into query-traffic detector inputs.

The caller gives an explicit local attack-output root, for example:

  /path/to/outputs/attacks_full

The collector does not modify ``attacks/`` outputs. It copies/extracts the
teacher queries it finds under the given root into a uniform
``teacher_received_queries.jsonl`` layout for query-traffic detectors.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from attacks.core.hf_query_pool import (
    hf_query_pool_for_budget,
    load_query_pool_records,
    record_prompt_id,
    record_prompt_text,
    resolve_query_pool_and_ordering,
)
from defenses.core.io_utils import ensure_dir, read_jsonl, write_jsonl
from defenses.query_traffic.common import pick_query


DIRECT_QUERY_LOG_NAMES = {
    "teacher_received_queries.jsonl",
}

TRANSCRIPT_CANDIDATE_NAMES = {
    "transcript.jsonl",
    "teacher_raw.jsonl",
    "combined_teacher.jsonl",
    "seed_teacher.jsonl",
    "template_teacher.jsonl",
    "followup_teacher.jsonl",
    "templated_queries.jsonl",
    "preferences.jsonl",
}

MANIFEST_CANDIDATE_NAMES = {
    "attack_manifest.json",
}

CANDIDATE_NAMES = DIRECT_QUERY_LOG_NAMES | TRANSCRIPT_CANDIDATE_NAMES | MANIFEST_CANDIDATE_NAMES


class AttackQueryCollectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    display: str
    local_root: Path | None = None


def parse_source_root(raw: str) -> SourceSpec:
    raw = raw.strip()
    if raw.startswith(("hf://", "hf-model://", "http://", "https://")):
        raise AttackQueryCollectionError("Attack outputs must be provided as a local filesystem path")

    return SourceSpec(kind="local", display=raw, local_root=Path(raw).expanduser())


def path_parts_after_anchor(parts: tuple[str, ...]) -> tuple[str, str]:
    """Infer (attack, run_id) from a candidate file path."""
    if "attacks" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("attacks")
        rest = parts[idx + 1 :]
    else:
        rest = parts
    attack = "unknown_attack"
    run_id = "unknown_run"
    if len(rest) >= 3:
        attack = rest[0]
        run_id = rest[1]
    elif len(rest) >= 2:
        attack = rest[0]
        run_id = Path(rest[1]).stem
    return sanitize_component(attack), sanitize_component(run_id)


def sanitize_component(value: str) -> str:
    value = str(value).strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_") or "unknown"


def is_candidate_relative_path(path: str) -> bool:
    return Path(path).name in CANDIDATE_NAMES


def list_local_candidates(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        if root.name not in CANDIDATE_NAMES:
            raise AttackQueryCollectionError(f"Local source file is not a supported query/transcript file: {root}")
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name in CANDIDATE_NAMES)


def source_priority(name: str) -> int:
    order = {
        "teacher_received_queries.jsonl": 0,
        "combined_teacher.jsonl": 1,
        "teacher_raw.jsonl": 2,
        "transcript.jsonl": 3,
        "seed_teacher.jsonl": 10,
        "template_teacher.jsonl": 11,
        "followup_teacher.jsonl": 12,
        "templated_queries.jsonl": 20,
        "preferences.jsonl": 30,
        "attack_manifest.json": 50,
    }
    return order.get(name, 99)


def select_group_sources(entries: list[tuple[Path, str, tuple[str, ...]]]) -> list[tuple[Path, str, tuple[str, ...]]]:
    entries = sorted(entries, key=lambda item: source_priority(item[0].name))
    names = {path.name for path, _, _ in entries}
    for preferred in ("teacher_received_queries.jsonl", "combined_teacher.jsonl", "teacher_raw.jsonl", "transcript.jsonl", "preferences.jsonl"):
        if preferred in names:
            return [entry for entry in entries if entry[0].name == preferred]
    if "attack_manifest.json" in names:
        return [entry for entry in entries if entry[0].name == "attack_manifest.json"]
    return entries


def _manifest_attack_and_budget(manifest: dict[str, Any]) -> tuple[str, int | None, str]:
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    run_config = manifest.get("run_config") if isinstance(manifest.get("run_config"), dict) else {}
    attack = str(result.get("attack") or run_config.get("attack") or manifest.get("attack") or "unknown_attack")
    budget_raw = result.get("budget") or run_config.get("budget") or manifest.get("budget")
    budget = int(budget_raw) if budget_raw is not None else None
    run_id = str(result.get("run_id") or run_config.get("run_id") or manifest.get("run_id") or "unknown_run")
    return sanitize_component(attack), budget, sanitize_component(run_id)


def _query_pool_spec_from_manifest(manifest: dict[str, Any], budget: int) -> str:
    run_config = manifest.get("run_config") if isinstance(manifest.get("run_config"), dict) else {}
    query_pool = run_config.get("query_pool") or run_config.get("query_pool_path")
    if isinstance(query_pool, str) and query_pool.startswith("hf://"):
        return query_pool
    if isinstance(query_pool, str) and Path(query_pool).expanduser().exists():
        return str(Path(query_pool).expanduser().resolve())
    return hf_query_pool_for_budget(budget)


def normalize_seqkd_manifest(
    *,
    manifest_path: Path,
    source_label: str,
    output_dir: Path,
    limit: int | None,
) -> dict[str, Any] | None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attack, budget, run_id = _manifest_attack_and_budget(manifest)
    if attack != "seqkd" or budget is None:
        return None
    num_queries = min(limit, budget) if limit is not None else budget
    query_pool_spec = _query_pool_spec_from_manifest(manifest, budget)
    query_pool_path, _ = resolve_query_pool_and_ordering(query_pool_spec, "auto")
    pool_records, pool_meta = load_query_pool_records(query_pool_path)
    selected = pool_records[:num_queries]
    if len(selected) < num_queries:
        raise AttackQueryCollectionError(
            f"SeqKD manifest requests {num_queries} queries, but {query_pool_path} has only {len(pool_records)} rows"
        )

    out_path = output_dir / attack / run_id / "teacher_received_queries.jsonl"
    ensure_dir(out_path.parent)
    records = []
    for index, record in enumerate(selected):
        qid = record_prompt_id(record, index)
        records.append(
            {
                "id": qid,
                "query_id": qid,
                "query": record_prompt_text(record).strip(),
                "attack": attack,
                "attack_run_id": run_id,
                "source_index": index,
                "source_path": source_label,
                "source_file_name": manifest_path.name,
                "source_query_pool": str(query_pool_path),
                "source_bank": record.get("bank"),
                "source_dataset": record.get("source"),
                "source_split": record.get("split"),
            }
        )
    write_jsonl(records, out_path)
    result = {
        "path": str(out_path.resolve()),
        "attack": attack,
        "run_id": run_id,
        "num_queries": len(records),
        "skipped_records": 0,
        "sources": [
            {
                "path": source_label,
                "file_name": manifest_path.name,
                "num_queries": len(records),
                "fallback": "seqkd_query_pool_from_attack_manifest",
                "query_pool": str(query_pool_path),
                "query_pool_meta": pool_meta,
            }
        ],
        "available_source_files": [source_label],
    }
    manifest_out = out_path.parent / "attack_query_collection_manifest.json"
    manifest_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["collection_manifest"] = str(manifest_out.resolve())
    return result


def normalize_group(
    *,
    entries: list[tuple[Path, str, tuple[str, ...]]],
    output_dir: Path,
    limit: int | None,
) -> dict[str, Any] | None:
    if not entries:
        return None
    attack, run_id = path_parts_after_anchor(entries[0][2])
    selected = select_group_sources(entries)
    out_path = output_dir / attack / run_id / "teacher_received_queries.jsonl"
    ensure_dir(out_path.parent)

    records: list[dict[str, Any]] = []
    skipped = 0
    source_summaries: list[dict[str, Any]] = []
    for source_path, source_label, _ in selected:
        if source_path.name == "attack_manifest.json":
            return normalize_seqkd_manifest(
                manifest_path=source_path,
                source_label=source_label,
                output_dir=output_dir,
                limit=limit,
            )
        before = len(records)
        for source_index, record in enumerate(read_jsonl(source_path)):
            query = pick_query(record)
            if query is None:
                skipped += 1
                continue
            qid = record.get("query_id") or record.get("id") or record.get("idx") or record.get("example_id")
            if qid is None:
                qid = f"{attack}_{run_id}_{len(records):08d}"
            records.append(
                {
                    "id": str(qid),
                    "query_id": str(qid),
                    "query": query,
                    "attack": attack,
                    "attack_run_id": run_id,
                    "source_index": source_index,
                    "source_path": source_label,
                    "source_file_name": source_path.name,
                }
            )
            if limit is not None and len(records) >= limit:
                break
        source_summaries.append(
            {
                "path": source_label,
                "file_name": source_path.name,
                "num_queries": len(records) - before,
            }
        )
        if limit is not None and len(records) >= limit:
            break

    if not records:
        return None
    write_jsonl(records, out_path)
    result = {
        "path": str(out_path.resolve()),
        "attack": attack,
        "run_id": run_id,
        "num_queries": len(records),
        "skipped_records": skipped,
        "sources": source_summaries,
        "available_source_files": [label for _, label, _ in entries],
    }
    manifest_path = out_path.parent / "attack_query_collection_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["collection_manifest"] = str(manifest_path.resolve())
    return result


def group_entries(entries: Iterable[tuple[Path, str, tuple[str, ...]]]) -> dict[tuple[str, str], list[tuple[Path, str, tuple[str, ...]]]]:
    groups: dict[tuple[str, str], list[tuple[Path, str, tuple[str, ...]]]] = {}
    for entry in entries:
        key = path_parts_after_anchor(entry[2])
        groups.setdefault(key, []).append(entry)
    return groups


def collect_from_local(spec: SourceSpec, output_dir: Path, *, limit: int | None) -> list[dict[str, Any]]:
    assert spec.local_root is not None
    root = spec.local_root.expanduser().resolve()
    candidates = list_local_candidates(root)
    entries = []
    for candidate in candidates:
        rel_parts = candidate.parts if root.is_file() else candidate.relative_to(root).parts
        entries.append((candidate, str(candidate.resolve()), rel_parts))

    results: list[dict[str, Any]] = []
    for _, group in sorted(group_entries(entries).items()):
        result = normalize_group(entries=group, output_dir=output_dir, limit=limit)
        if result:
            print(f"[OK] {result['attack']} {result['run_id']} -> {result['path']} ({result['num_queries']} queries)")
            results.append(result)
        else:
            print(f"[WARN] no query text extracted from {[label for _, label, _ in group]}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect teacher-query logs from an explicit local attack-output root.")
    parser.add_argument(
        "--source-root",
        required=True,
        help="Local attack output path.",
    )
    parser.add_argument("--output-dir", required=True, help="Where normalized teacher_received_queries.jsonl files are written.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max queries per source file.")
    parser.add_argument("--index-name", default="attack_query_collection_index.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = parse_source_root(args.source_root)
    output_dir = ensure_dir(Path(args.output_dir).expanduser())
    results = collect_from_local(spec, output_dir, limit=args.limit)

    index = {
        "source_root": args.source_root,
        "source_kind": spec.kind,
        "output_dir": str(output_dir.resolve()),
        "num_collected": len(results),
        "results": results,
    }
    index_path = output_dir / args.index_name
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[index] {index_path.resolve()}")
    if not results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
