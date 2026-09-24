from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from tqdm import tqdm

from .common import append_jsonl, read_jsonl, record_id, record_prompt
from .openai_compat import GenerationSettings, generate_text


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(record.get("prompt_id")) for record in read_jsonl(path) if record.get("prompt_id") is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect static zero-shot student negatives for SODA.")
    parser.add_argument("--teacher-jsonl", required=True, help="Teacher transcript JSONL; prompt ids are reused.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--request-model-name", required=True)
    parser.add_argument("--mode", choices=["completion", "chat"], default="completion")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("ATTACK_QUERY_CONCURRENCY", "8")))
    args = parser.parse_args()

    output = Path(args.output_jsonl)
    done = existing_ids(output)
    settings = GenerationSettings(args.temperature, args.top_p, args.max_tokens, args.seed)
    records = read_jsonl(args.teacher_jsonl)
    if args.limit is not None:
        records = records[: args.limit]

    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    pending = [(idx, source) for idx, source in enumerate(records) if record_id(source, idx) not in done]

    def collect(item: tuple[int, dict]) -> dict:
        idx, source = item
        pid = record_id(source, idx)
        prompt = record_prompt(source)
        response, tokens, latency, retry_count = generate_text(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.request_model_name,
            prompt=prompt,
            mode=args.mode,
            settings=settings,
        )
        return {
            "prompt_id": pid,
            "prompt": prompt,
            "student_response": response,
            "method": "soda",
            "negative_source": "student_zero_shot_static",
            "generation_config": settings.__dict__,
            "request_model_name": args.request_model_name,
            "endpoint_base_url": args.base_url,
            "latency_seconds": latency,
            "retry_count": retry_count,
            **tokens,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for result in tqdm(executor.map(collect, pending), total=len(pending), desc="SODA student negatives"):
            append_jsonl(output, result)
    print(f"Wrote SODA student negatives to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

