from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

from attacks.core.hf_query_pool import (
    DEFAULT_DATASET_ID,
    file_sha256,
    load_query_pool_records,
    record_prompt_id,
    record_prompt_text,
    resolve_query_pool_and_ordering,
)
from attacks.core.manifest import stable_hash


DEFAULT_QUERY_POOL = f"hf://{DEFAULT_DATASET_ID}/query_pool_100000.json"


def default_output_dir() -> str:
    storage_root = os.environ.get("STORAGE_ROOT")
    if storage_root:
        return str(Path(storage_root).expanduser() / "outputs" / "heldout_queries")
    return "outputs/heldout_queries"


class HeldoutBuildError(RuntimeError):
    pass


def ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def parse_block_ids(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    block_ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        block_ids.append(int(item))
    if not block_ids:
        raise HeldoutBuildError("--block-ids did not contain any block ids")
    return block_ids


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_heldout(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_train_budget < 0:
        raise HeldoutBuildError("--max-train-budget must be non-negative")
    if args.block_size <= 0:
        raise HeldoutBuildError("--block-size must be positive")
    if args.num_blocks <= 0:
        raise HeldoutBuildError("--num-blocks must be positive")

    query_pool_path, _ = resolve_query_pool_and_ordering(args.query_pool, None)
    records, meta = load_query_pool_records(query_pool_path)
    if not records:
        raise HeldoutBuildError(f"query pool has no records: {query_pool_path}")
    if args.max_train_budget >= len(records):
        raise HeldoutBuildError(
            f"--max-train-budget={args.max_train_budget} leaves no heldout records in pool of size {len(records)}"
        )

    explicit_block_ids = parse_block_ids(args.block_ids)
    if explicit_block_ids is None:
        start_block = args.start_block if args.start_block is not None else ceil_div(args.max_train_budget, args.block_size)
        block_ids = list(range(start_block, start_block + args.num_blocks))
    else:
        block_ids = explicit_block_ids

    train_ids = {record_prompt_id(record, idx) for idx, record in enumerate(records[: args.max_train_budget])}
    selected_rows: list[dict[str, Any]] = []
    selected_source_indices: list[int] = []

    for block_id in block_ids:
        if block_id < 0:
            raise HeldoutBuildError(f"block id must be non-negative: {block_id}")
        start = block_id * args.block_size
        end = start + args.block_size
        if start < args.max_train_budget:
            raise HeldoutBuildError(
                f"block {block_id} starts at source index {start}, which overlaps max_train_budget={args.max_train_budget}"
            )
        if end > len(records):
            raise HeldoutBuildError(
                f"block {block_id} [{start}:{end}] exceeds query pool length {len(records)}"
            )
        for source_index, record in enumerate(records[start:end], start):
            item_id = record_prompt_id(record, source_index)
            if item_id in train_ids:
                raise HeldoutBuildError(f"heldout id overlaps train prefix: {item_id}")
            prompt = record_prompt_text(record)
            if not prompt:
                raise HeldoutBuildError(f"record has empty prompt at source index {source_index}")
            row = {
                "id": item_id,
                "prompt": prompt,
                "source_index": source_index,
                "block_id": block_id,
                "bank": record.get("bank"),
                "source": record.get("source"),
                "split": record.get("split"),
                "slice": record.get("slice"),
                "rank": record.get("rank", source_index),
            }
            selected_rows.append(row)
            selected_source_indices.append(source_index)

    output_dir = Path(args.output_dir).expanduser().resolve()
    prompts_output = Path(args.prompts_output).expanduser().resolve() if args.prompts_output else output_dir / "heldout_prompts.jsonl"
    manifest_output = Path(args.manifest_output).expanduser().resolve() if args.manifest_output else output_dir / "heldout_manifest.json"

    selected_ids = [row["id"] for row in selected_rows]
    bank_counts = Counter(str(row.get("bank")) for row in selected_rows)
    source_counts = Counter(str(row.get("source")) for row in selected_rows)
    heldout_hash = stable_hash([(row["id"], row["prompt"]) for row in selected_rows])

    manifest = {
        "schema_version": "heldout_queries_v1",
        "query_pool": args.query_pool,
        "query_pool_path": str(query_pool_path),
        "query_pool_sha256": file_sha256(query_pool_path),
        "query_pool_meta": meta,
        "max_train_budget": args.max_train_budget,
        "block_size": args.block_size,
        "block_ids": block_ids,
        "source_index_start": min(selected_source_indices),
        "source_index_end_exclusive": max(selected_source_indices) + 1,
        "num_records": len(selected_rows),
        "id_hash": stable_hash(selected_ids),
        "heldout_hash": heldout_hash,
        "bank_counts": dict(sorted(bank_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "disjoint_from_train_prefix": True,
        "prompts_output": str(prompts_output),
    }

    write_jsonl(prompts_output, selected_rows)
    write_json(manifest_output, manifest)
    return {"prompts_output": str(prompts_output), "manifest_output": str(manifest_output), **manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build heldout evaluation prompts from complete query-pool blocks. "
            "The default selects blocks 10, 11, and 12 from query_pool_100000, "
            "which is disjoint from the benchmark's max attack budget of 10000."
        )
    )
    parser.add_argument("--query-pool", default=DEFAULT_QUERY_POOL)
    parser.add_argument("--max-train-budget", type=int, default=10000)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--start-block", type=int, help="First heldout block id. Defaults to ceil(max_train_budget / block_size).")
    parser.add_argument("--block-ids", help="Comma-separated explicit block ids, e.g. 10,11,12. Overrides --start-block/--num-blocks.")
    parser.add_argument(
        "--output-dir",
        default=default_output_dir(),
        help="Directory for heldout_prompts.jsonl and heldout_manifest.json. Defaults to $STORAGE_ROOT/outputs/heldout_queries when STORAGE_ROOT is set; otherwise outputs/heldout_queries.",
    )
    parser.add_argument("--prompts-output", help="Explicit heldout_prompts.jsonl output path.")
    parser.add_argument("--manifest-output", help="Explicit heldout_manifest.json output path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_heldout(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
