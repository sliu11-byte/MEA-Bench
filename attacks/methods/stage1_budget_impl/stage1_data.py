from __future__ import annotations

"""Shared stage-1 data mixing, deduplication, and decontamination."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
import unicodedata
from typing import Any, Mapping, Sequence

from .stage1_transcript import stable_hash


@dataclass(frozen=True)
class DataRecord:
    prompt_id: str
    prompt_text: str
    source: str
    category: str
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedData:
    records: tuple[DataRecord, ...]
    budget_prefixes: Mapping[int, tuple[DataRecord, ...]]
    ordering_hash: str
    report_path: Path
    records_path: Path
    report: Mapping[str, Any]


def normalize_prompt(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    return re.sub(r"\s+", " ", text)


def _source_counts(records: Sequence[DataRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.source] = counts.get(record.source, 0) + 1
    return dict(sorted(counts.items()))


def _category_counts(records: Sequence[DataRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.category] = counts.get(record.category, 0) + 1
    return dict(sorted(counts.items()))


def _mix_to_ratio(
    general: Sequence[DataRecord],
    special: Sequence[DataRecord],
    target_general_ratio: float,
) -> list[DataRecord]:
    if not 0.0 <= target_general_ratio <= 1.0:
        raise ValueError("target_general_ratio must be in [0, 1].")
    if not general or not special:
        return list(general) + list(special)
    special_ratio = 1.0 - target_general_ratio
    total_from_general = int(len(general) / target_general_ratio) if target_general_ratio else len(special)
    total_from_special = int(len(special) / special_ratio) if special_ratio else len(general)
    target_total = max(1, min(total_from_general, total_from_special))
    general_count = min(len(general), round(target_total * target_general_ratio))
    special_count = min(len(special), target_total - general_count)
    return list(general[:general_count]) + list(special[:special_count])


def prepare_stage1_data(
    *,
    general_records: Sequence[DataRecord],
    special_records: Sequence[DataRecord],
    m1_eval_prompts: Sequence[DataRecord],
    output_dir: str | Path,
    seed: int,
    target_general_ratio: float = 0.8,
    budgets: Sequence[int] = (100, 1000, 10000),
) -> PreparedData:
    """Run the required pipeline in order without synthesizing samples."""
    input_records = list(general_records) + list(special_records)
    mixed = _mix_to_ratio(general_records, special_records, target_general_ratio)

    deduplicated: list[DataRecord] = []
    seen: set[str] = set()
    dedup_deleted: list[str] = []
    for record in mixed:
        key = normalize_prompt(record.prompt_text)
        if key in seen:
            dedup_deleted.append(record.prompt_id)
            continue
        seen.add(key)
        deduplicated.append(record)

    eval_keys = {normalize_prompt(record.prompt_text) for record in m1_eval_prompts}
    decontaminated: list[DataRecord] = []
    contamination_deleted: list[str] = []
    for record in deduplicated:
        if normalize_prompt(record.prompt_text) in eval_keys:
            contamination_deleted.append(record.prompt_id)
        else:
            decontaminated.append(record)

    rng = random.Random(seed)
    ordered = list(decontaminated)
    rng.shuffle(ordered)
    ordering_hash = stable_hash([record.prompt_id for record in ordered])
    budget_prefixes = {
        int(budget): tuple(ordered[: min(int(budget), len(ordered))]) for budget in budgets
    }
    final_categories = _category_counts(ordered)
    final_total = len(ordered)
    actual_general = final_categories.get("general_chat", 0) / final_total if final_total else 0.0
    report: dict[str, Any] = {
        "pipeline_order": [
            "load_and_mix",
            "normalized_exact_dedup",
            "m1_exact_match_decontamination",
            "fixed_seed_ordering",
            "nested_budget_prefixes",
        ],
        "seed": seed,
        "target_general_ratio": target_general_ratio,
        "input_source_counts": _source_counts(input_records),
        "mixed_source_counts": _source_counts(mixed),
        "final_source_counts": _source_counts(ordered),
        "final_category_counts": final_categories,
        "actual_general_ratio": actual_general,
        "actual_special_ratio": 1.0 - actual_general if final_total else 0.0,
        "dedup_report": {
            "input_count": len(mixed),
            "output_count": len(deduplicated),
            "deleted_prompt_ids": dedup_deleted,
        },
        "contamination_report": {
            "input_count": len(deduplicated),
            "output_count": len(decontaminated),
            "deleted_prompt_ids": contamination_deleted,
            "eval_prompt_count": len(m1_eval_prompts),
        },
        "ordering_hash": ordering_hash,
        "ordered_prompt_ids": [record.prompt_id for record in ordered],
        "budget_prefixes": {
            str(budget): {
                "requested": budget,
                "actual": len(prefix),
                "prompt_ids": [record.prompt_id for record in prefix],
                "insufficient_data": len(prefix) < budget,
            }
            for budget, prefix in budget_prefixes.items()
        },
        "deleted_prompt_ids": dedup_deleted + contamination_deleted,
        "sample_replication": False,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "prepared_data.jsonl"
    records_path.write_text(
        "".join(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for record in ordered),
        encoding="utf-8",
    )
    report_path = output_dir / "data_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return PreparedData(
        records=tuple(ordered),
        budget_prefixes=budget_prefixes,
        ordering_hash=ordering_hash,
        report_path=report_path,
        records_path=records_path,
        report=report,
    )


__all__ = ["DataRecord", "PreparedData", "normalize_prompt", "prepare_stage1_data"]
