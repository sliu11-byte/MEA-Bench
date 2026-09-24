from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert QEDKS teacher transcript to prompt/response SFT JSONL.")
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    output = []
    for record in read_jsonl(args.teacher_jsonl):
        output.append(
            {
                "prompt_id": record["query_id"],
                "prompt": record["query"],
                "response": record["teacher_response"],
                "method": "qedks",
                "query_type": record.get("query_type"),
                "source_entity": record.get("source_entity"),
            }
        )
    write_jsonl(Path(args.output_jsonl), output)
    print(f"Wrote {len(output)} QEDKS SFT records to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


