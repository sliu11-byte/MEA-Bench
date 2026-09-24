"""Prepare benign reference query logs for query-traffic detectors."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from attacks.core.hf_query_pool import (
    DEFAULT_DATASET_ID,
    ensure_hf_file,
    hf_query_pool_for_budget,
    load_query_pool_records,
    record_prompt_id,
    record_prompt_text,
)
from defenses.core.io_utils import ensure_dir, write_jsonl


DEFAULT_CANDIDATE_POOL = f"hf://{DEFAULT_DATASET_ID}/query_pool_100000.json"
DEFAULT_USER_DATASET = "OpenLeecher/lmsys_chat_1m_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benign query reference logs for MMD/PRADA/SEAT.")
    parser.add_argument("--mode", choices=["user_traffic", "matched_mix", "both"], default="both")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--attack_query_pool", default="auto")
    parser.add_argument(
        "--attack_budget",
        type=int,
        default=None,
        help="Prefix of --attack_query_pool used as the attack distribution. Defaults to --num_queries.",
    )
    parser.add_argument("--candidate_query_pool", default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--user_dataset", default=DEFAULT_USER_DATASET)
    parser.add_argument("--user_split", default="train")
    parser.add_argument(
        "--user_max_scan",
        type=int,
        default=200000,
        help="Maximum streamed rows scanned from --user_dataset before giving up.",
    )
    parser.add_argument(
        "--fallback_user_from_candidate_lmsys",
        action="store_true",
        help="If the LMSYS HF dataset cannot be streamed, use disjoint LMSYS rows from --candidate_query_pool.",
    )
    return parser.parse_args()


def _resolve_pool_path(spec: str) -> Path:
    if spec.startswith("hf://"):
        return ensure_hf_file(spec)
    return Path(spec).expanduser().resolve()


def _record_key(record: dict[str, Any], index: int) -> tuple[str, str]:
    return record_prompt_id(record, index), record_prompt_text(record).strip()


def _load_attack_records(path: Path, attack_budget: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, meta = load_query_pool_records(path)
    selected = records[:attack_budget]
    if len(selected) < attack_budget:
        raise ValueError(f"Attack query pool has {len(records)} rows, fewer than attack_budget={attack_budget}")
    return selected, meta


def _excluded_sets(records: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    prompts: set[str] = set()
    for index, record in enumerate(records):
        prompt_id, prompt = _record_key(record, index)
        ids.add(prompt_id)
        prompts.add(prompt)
    return ids, prompts


def _largest_remainder_counts(weights: dict[str, int], total: int) -> dict[str, int]:
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Cannot allocate counts from empty weights")
    raw = {key: total * value / weight_sum for key, value in weights.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda key: (raw[key] - counts[key], weights[key], key), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _format_pool_record(record: dict[str, Any], *, source_role: str, index: int) -> dict[str, Any]:
    prompt = record_prompt_text(record).strip()
    prompt_id = record_prompt_id(record, index)
    return {
        "query_id": f"{source_role}_{index:06d}",
        "query": prompt,
        "source_role": source_role,
        "source_prompt_id": prompt_id,
        "source_bank": record.get("bank"),
        "source_dataset": record.get("source"),
        "source_split": record.get("split"),
        "source_rank": record.get("rank"),
    }


def build_matched_mix(
    *,
    attack_records: list[dict[str, Any]],
    excluded_ids: set[str],
    excluded_prompts: set[str],
    candidate_pool: Path,
    num_queries: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, candidate_meta = load_query_pool_records(candidate_pool)
    target_weights = Counter(str(record.get("bank") or "unknown") for record in attack_records)
    target_counts = _largest_remainder_counts(dict(target_weights), num_queries)
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(candidates):
        prompt_id, prompt = _record_key(record, index)
        if prompt_id in excluded_ids or prompt in excluded_prompts:
            continue
        bank = str(record.get("bank") or "unknown")
        if bank in target_counts:
            grouped[bank].append((index, record))

    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    shortfalls: dict[str, int] = {}
    for bank, count in sorted(target_counts.items()):
        pool = grouped.get(bank, [])
        if len(pool) < count:
            shortfalls[bank] = count - len(pool)
            count = len(pool)
        chosen = rng.sample(pool, count)
        output.extend(_format_pool_record(record, source_role="benign_matched_mix", index=index) for index, record in chosen)

    if len(output) < num_queries:
        raise ValueError(
            f"Matched-mix benign reference only found {len(output)} rows, need {num_queries}. "
            f"Shortfalls by bank: {shortfalls}"
        )
    rng.shuffle(output)
    output = output[:num_queries]
    for i, row in enumerate(output):
        row["query_id"] = f"benign_matched_mix_{i:06d}"
    meta = {
        "mode": "matched_mix",
        "num_queries": len(output),
        "candidate_pool": str(candidate_pool),
        "candidate_pool_meta": candidate_meta,
        "target_bank_counts": dict(target_counts),
        "actual_bank_counts": dict(Counter(row.get("source_bank") for row in output)),
        "excluded_attack_records": len(attack_records),
        "seed": seed,
    }
    return output, meta


def _prompt_from_lmsys_row(row: dict[str, Any]) -> str:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("from") or turn.get("role") or "").lower()
            if speaker in {"human", "user"}:
                value = turn.get("value") or turn.get("content")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    for key in ("prompt", "query", "instruction", "question", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _stream_lmsys_rows(dataset: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to stream user_traffic from Hugging Face") from exc
    dataset_iter = load_dataset(dataset, split=split, streaming=True)
    for row in dataset_iter:
        yield dict(row)


def build_user_traffic_from_lmsys(
    *,
    dataset: str,
    split: str,
    excluded_prompts: set[str],
    num_queries: int,
    max_scan: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scanned = 0
    for row in _stream_lmsys_rows(dataset, split):
        scanned += 1
        prompt = _prompt_from_lmsys_row(row)
        if prompt and prompt not in excluded_prompts:
            output.append(
                {
                    "query_id": f"benign_user_traffic_{len(output):06d}",
                    "query": prompt,
                    "source_role": "benign_user_traffic",
                    "source_dataset": dataset,
                    "source_split": split,
                    "source_prompt_id": row.get("id"),
                    "source_category": row.get("category"),
                    "source_flaw": row.get("flaw"),
                }
            )
            if len(output) >= num_queries:
                break
        if scanned >= max_scan:
            break
    if len(output) < num_queries:
        raise ValueError(
            f"User-traffic benign reference only found {len(output)} rows after scanning {scanned}; "
            f"need {num_queries}."
        )
    meta = {
        "mode": "user_traffic",
        "num_queries": len(output),
        "user_dataset": dataset,
        "user_split": split,
        "scanned_rows": scanned,
    }
    return output, meta


def build_user_traffic_from_candidate_lmsys(
    *,
    excluded_ids: set[str],
    excluded_prompts: set[str],
    candidate_pool: Path,
    num_queries: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, candidate_meta = load_query_pool_records(candidate_pool)
    rows: list[tuple[int, dict[str, Any]]] = []
    for index, record in enumerate(candidates):
        if record.get("bank") != "LMSYS-Chat-1M":
            continue
        prompt_id, prompt = _record_key(record, index)
        if prompt_id not in excluded_ids and prompt not in excluded_prompts:
            rows.append((index, record))
    if len(rows) < num_queries:
        raise ValueError(f"Candidate pool has {len(rows)} disjoint LMSYS rows, need {num_queries}")
    rng = random.Random(seed)
    chosen = rng.sample(rows, num_queries)
    output = [
        _format_pool_record(record, source_role="benign_user_traffic_candidate_lmsys", index=index)
        for index, record in chosen
    ]
    for i, row in enumerate(output):
        row["query_id"] = f"benign_user_traffic_{i:06d}"
    meta = {
        "mode": "user_traffic",
        "source_variant": "candidate_lmsys_fallback",
        "num_queries": len(output),
        "candidate_pool": str(candidate_pool),
        "candidate_pool_meta": candidate_meta,
        "seed": seed,
    }
    return output, meta


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    attack_budget = args.attack_budget or args.num_queries
    attack_pool_spec = hf_query_pool_for_budget(attack_budget) if args.attack_query_pool in (None, "auto") else args.attack_query_pool
    attack_pool = _resolve_pool_path(attack_pool_spec)
    candidate_pool = _resolve_pool_path(args.candidate_query_pool)
    attack_records, attack_meta = _load_attack_records(attack_pool, attack_budget)
    excluded_ids, excluded_prompts = _excluded_sets(attack_records)

    modes = ["user_traffic", "matched_mix"] if args.mode == "both" else [args.mode]
    summary: dict[str, Any] = {
        "schema_version": "query_traffic_benign_reference_v1",
        "attack_query_pool": str(attack_pool),
        "attack_query_pool_spec": attack_pool_spec,
        "attack_query_pool_meta": attack_meta,
        "attack_budget": attack_budget,
        "num_queries": args.num_queries,
        "outputs": {},
    }
    for mode in modes:
        if mode == "matched_mix":
            rows, meta = build_matched_mix(
                attack_records=attack_records,
                excluded_ids=excluded_ids,
                excluded_prompts=excluded_prompts,
                candidate_pool=candidate_pool,
                num_queries=args.num_queries,
                seed=args.seed,
            )
        elif mode == "user_traffic":
            try:
                rows, meta = build_user_traffic_from_lmsys(
                    dataset=args.user_dataset,
                    split=args.user_split,
                    excluded_prompts=excluded_prompts,
                    num_queries=args.num_queries,
                    max_scan=args.user_max_scan,
                )
            except Exception as exc:
                if not args.fallback_user_from_candidate_lmsys:
                    raise
                print(
                    "[query_traffic] user_traffic HF streaming failed; "
                    f"falling back to disjoint LMSYS rows from candidate pool: {exc}",
                    file=sys.stderr,
                )
                rows, meta = build_user_traffic_from_candidate_lmsys(
                    excluded_ids=excluded_ids,
                    excluded_prompts=excluded_prompts,
                    candidate_pool=candidate_pool,
                    num_queries=args.num_queries,
                    seed=args.seed + 17,
                )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        path = out_dir / f"benign_{mode}.jsonl"
        write_jsonl(rows, path)
        meta_path = out_dir / f"benign_{mode}.meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["outputs"][mode] = {"path": str(path), "meta_path": str(meta_path), "meta": meta}

    summary_path = out_dir / "benign_reference_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
