from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_jsonl


REJECT_MARKERS = {"", "UNSURE", "UNKNOWN", "N/A", "NONE", "NULL"}


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def clean_record(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    raw = str(record.get("teacher_response") or record.get("raw_response") or "").strip()
    if not raw:
        return None, "empty_response"
    parsed = parse_json_object(raw)
    if parsed is None:
        return None, "json_parse_failed"
    answer = str(parsed.get("answer", "")).strip()
    if answer.upper() in REJECT_MARKERS:
        return None, "rejected_or_unsure"
    sentence = str(parsed.get("sentence", "")).strip()
    return (
        {
            "prompt_id": record.get("prompt_id") or record.get("query_id"),
            "query_id": record.get("query_id"),
            "source_prompt": record.get("source_prompt") or record.get("query"),
            "templated_query": record.get("query"),
            "response": answer,
            "supporting_sentence": sentence,
            "raw_response": raw,
            "parsed_response": parsed,
            "method": "model_leeching",
            "template_id": record.get("template_id"),
            "source_record": record.get("source_record"),
        },
        "kept",
    )


def clean_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for record in records:
        cleaned, reason = clean_record(record)
        reasons[reason] += 1
        if cleaned is None:
            rejected.append(
                {
                    "query_id": record.get("query_id"),
                    "prompt_id": record.get("prompt_id"),
                    "reason": reason,
                    "raw_response": record.get("teacher_response") or record.get("raw_response"),
                }
            )
        else:
            kept.append(cleaned)
    total = len(records)
    stats = {
        "schema_version": "model_leeching_cleaning_stats_v1",
        "total_records": total,
        "kept_records": len(kept),
        "rejected_records": len(rejected),
        "keep_rate": 0.0 if total == 0 else len(kept) / total,
        "reject_rate": 0.0 if total == 0 else len(rejected) / total,
        "reasons": dict(reasons),
    }
    return kept, stats, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse, clean, and validate Model Leeching teacher responses.")
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--stats-json", required=True)
    parser.add_argument("--rejected-jsonl")
    args = parser.parse_args()

    kept, stats, rejected = clean_records(read_jsonl(args.teacher_jsonl))
    write_jsonl(Path(args.output_jsonl), kept)
    Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_json).write_text(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if args.rejected_jsonl:
        write_jsonl(Path(args.rejected_jsonl), rejected)
    print(f"Kept {len(kept)} / {stats['total_records']} Model Leeching records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
