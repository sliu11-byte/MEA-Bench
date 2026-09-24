from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .common import read_jsonl, record_response, write_jsonl


DEFAULT_SEEDS = [
    "Give a detailed overview of an important specialized field and its key concepts.",
    "Explain several advanced concepts in a technical domain and how they relate to each other.",
    "Describe important entities, mechanisms, and relationships in a professional knowledge area.",
]

TEMPLATES = {
    "DEF": "What is {entity}? Give a precise definition and explain why it is important.",
    "CAT": "What category does {entity} belong to, and what are closely related concepts?",
    "FUN": "What is the function or role of {entity} in its domain?",
    "PART": "What are the main components, subtypes, or parts of {entity}?",
}


def extract_entities(text: str, max_entities: int = 50) -> list[str]:
    candidates: OrderedDict[str, None] = OrderedDict()
    patterns = [
        r"\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4}\b",
        r"\b[a-z][a-z0-9\-]{4,}(?:\s+[a-z][a-z0-9\-]{4,}){0,3}\b",
    ]
    stop = {
        "this",
        "that",
        "these",
        "those",
        "therefore",
        "because",
        "important",
        "domain",
        "concept",
        "concepts",
    }
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            entity = " ".join(match.group(0).strip(" .,;:()[]{}").split())
            if len(entity) < 4 or entity.lower() in stop:
                continue
            candidates.setdefault(entity, None)
            if len(candidates) >= max_entities:
                return list(candidates)
    return list(candidates)


def seed_query_records() -> list[dict[str, Any]]:
    return [
        {
            "query_id": f"qedks_seed_{idx:04d}",
            "query": prompt,
            "query_type": "seed",
            "source_entity": None,
            "priority": idx,
            "method": "qedks",
        }
        for idx, prompt in enumerate(DEFAULT_SEEDS)
    ]


def template_query_records(seed_transcript: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    text = "\n".join(record_response(record) for record in seed_transcript)
    entities = extract_entities(text, max_entities=max(1, budget))
    queries: list[dict[str, Any]] = []
    for entity in entities:
        for template_id, template in TEMPLATES.items():
            queries.append(
                {
                    "query_id": f"qedks_{template_id.lower()}_{len(queries):06d}",
                    "query": template.format(entity=entity),
                    "query_type": template_id,
                    "source_entity": entity,
                    "priority": len(queries),
                    "method": "qedks",
                }
            )
            if len(queries) >= budget:
                return queries
    return queries


def apply_ppl_schedule(records: list[dict[str, Any]], ppl_records: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not ppl_records:
        return records
    scores = {str(r.get("query_id")): float(r.get("ppl", 0.0)) for r in ppl_records if r.get("query_id")}
    return sorted(records, key=lambda r: (-scores.get(str(r.get("query_id")), -1.0), r.get("priority", 0)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build QEDKS seed/template query plan.")
    parser.add_argument("--seed-transcript", default=None, help="Teacher responses from seed prompts. If absent, emit seed prompts.")
    parser.add_argument("--student-ppl-jsonl", default=None, help="Optional query_id/ppl file for PPL-guided scheduling.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()

    if args.seed_transcript is None:
        records = seed_query_records()[: args.budget]
    else:
        records = template_query_records(read_jsonl(args.seed_transcript), args.budget)
    ppl = read_jsonl(args.student_ppl_jsonl) if args.student_ppl_jsonl else None
    records = apply_ppl_schedule(records, ppl)[: args.budget]
    write_jsonl(Path(args.output_jsonl), records)
    print(f"Wrote {len(records)} QEDKS query records to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


