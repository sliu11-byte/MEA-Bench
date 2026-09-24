from __future__ import annotations

import argparse
import re
from pathlib import Path

from .common import read_jsonl, write_jsonl


def sentence_chunks(text: str, max_chunks: int = 4) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    chunks = [part for part in parts if len(part.split()) >= 6]
    return chunks[:max_chunks]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create QEDKS Chain-of-Question follow-up prompts.")
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-followups-per-answer", type=int, default=4)
    parser.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()

    output = []
    for record in read_jsonl(args.teacher_jsonl):
        parent_id = record.get("query_id")
        answer = record.get("teacher_response") or record.get("raw_response") or ""
        for chunk in sentence_chunks(str(answer), args.max_followups_per_answer):
            output.append(
                {
                    "query_id": f"qedks_coq_{len(output):06d}",
                    "parent_query_id": parent_id,
                    "query": f"Please expand the following point with more concrete details: {chunk}",
                    "query_type": "CoQ",
                    "source_entity": record.get("source_entity"),
                    "priority": len(output),
                    "method": "qedks",
                }
            )
            if len(output) >= args.budget:
                write_jsonl(Path(args.output_jsonl), output)
                print(f"Wrote {len(output)} QEDKS CoQ follow-up queries to {args.output_jsonl}")
                return 0
    write_jsonl(Path(args.output_jsonl), output)
    print(f"Wrote {len(output)} QEDKS CoQ follow-up queries to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


