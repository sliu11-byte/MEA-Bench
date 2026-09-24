from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from tqdm import tqdm

from .common import append_jsonl, read_jsonl
from .openai_compat import GenerationSettings, generate_text


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(record.get("query_id")) for record in read_jsonl(path) if record.get("query_id") is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Query teacher endpoint for Model Leeching templated queries.")
    parser.add_argument("--query-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--request-model-name", required=True)
    parser.add_argument("--mode", choices=["completion", "chat"], default="chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("ATTACK_QUERY_CONCURRENCY", "8")))
    args = parser.parse_args()

    output = Path(args.output_jsonl)
    done = existing_ids(output)
    settings = GenerationSettings(args.temperature, args.top_p, args.max_tokens, args.seed)
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    pending = [record for record in read_jsonl(args.query_jsonl) if str(record["query_id"]) not in done]

    def collect(record: dict) -> dict:
        response, tokens, latency, retry_count = generate_text(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.request_model_name,
            prompt=record["query"],
            mode=args.mode,
            settings=settings,
        )
        return {
            **record,
            "teacher_response": response,
            "generation_config": settings.__dict__,
            "request_model_name": args.request_model_name,
            "endpoint_base_url": args.base_url,
            "latency_seconds": latency,
            "retry_count": retry_count,
            **tokens,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for result in tqdm(executor.map(collect, pending), total=len(pending), desc="Model Leeching teacher queries"):
            append_jsonl(output, result)
    print(f"Wrote Model Leeching teacher transcript to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
