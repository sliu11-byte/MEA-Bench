from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_jsonl, write_jsonl


def to_sft_records(records: list[dict]) -> list[dict]:
    output = []
    for record in records:
        output.append(
            {
                "prompt_id": record.get("prompt_id"),
                "prompt": record.get("source_prompt") or record.get("templated_query"),
                "response": record.get("response"),
                "method": "model_leeching",
                "template_id": record.get("template_id"),
                "metadata": {
                    "query_id": record.get("query_id"),
                    "supporting_sentence": record.get("supporting_sentence"),
                    "source_record": record.get("source_record"),
                },
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert cleaned Model Leeching records to SFT JSONL.")
    parser.add_argument("--clean-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()
    records = to_sft_records(read_jsonl(args.clean_jsonl))
    write_jsonl(Path(args.output_jsonl), records)
    print(f"Wrote {len(records)} Model Leeching SFT records to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
