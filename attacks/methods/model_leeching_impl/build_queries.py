from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import record_id, record_prompt, write_jsonl


DEFAULT_TEMPLATE = """You are an expert assistant. Complete the user task below and return only a valid JSON object.

User task:
{prompt}

Rules:
1. Use exactly these JSON keys: "answer", "sentence".
2. Put the best concise answer in "answer".
3. Put a short supporting sentence, rationale, or empty string in "sentence".
4. Do not include markdown, code fences, or any text outside the JSON object.
5. If the task cannot be answered, set "answer" to "UNSURE".
"""


def read_query_pool(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("data", [])
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"query pool must be a JSON object or list: {path}")
    if not isinstance(records, list):
        raise ValueError("query pool data must be a list")
    return [dict(record) for record in records]


def read_ordering(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    ordered = payload.get("ordered_prompt_ids") if isinstance(payload, dict) else None
    if not isinstance(ordered, list):
        raise ValueError("ordering file must contain ordered_prompt_ids")
    return [str(item) for item in ordered]


def select_records(records: list[dict[str, Any]], ordering: list[str] | None, budget: int) -> list[dict[str, Any]]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if ordering is None:
        return records[:budget]
    by_id = {record_id(record, idx): record for idx, record in enumerate(records)}
    selected = []
    for prompt_id in ordering:
        record = by_id.get(prompt_id)
        if record is not None:
            selected.append(record)
        if len(selected) >= budget:
            break
    return selected


def render_prompt(prompt: str, template: str) -> str:
    return template.format(prompt=prompt)


def build_query_records(
    records: list[dict[str, Any]],
    *,
    template: str = DEFAULT_TEMPLATE,
) -> list[dict[str, Any]]:
    output = []
    for idx, record in enumerate(records):
        prompt_id = record_id(record, idx)
        source_prompt = record_prompt(record)
        output.append(
            {
                "query_id": prompt_id,
                "prompt_id": prompt_id,
                "source_prompt": source_prompt,
                "query": render_prompt(source_prompt, template),
                "method": "model_leeching",
                "template_id": "json_answer_v1",
                "source_record": record,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Model Leeching templated teacher queries from a query pool.")
    parser.add_argument("--query-pool", required=True)
    parser.add_argument("--query-ordering")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--template-file")
    args = parser.parse_args()

    template = DEFAULT_TEMPLATE
    if args.template_file:
        template = Path(args.template_file).read_text(encoding="utf-8")
    records = read_query_pool(Path(args.query_pool))
    ordering = read_ordering(Path(args.query_ordering)) if args.query_ordering else None
    selected = select_records(records, ordering, args.budget)
    output = build_query_records(selected, template=template)
    write_jsonl(Path(args.output_jsonl), output)
    print(f"Wrote {len(output)} Model Leeching templated queries to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
