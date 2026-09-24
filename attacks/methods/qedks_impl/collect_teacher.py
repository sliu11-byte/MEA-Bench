from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from tqdm import tqdm

from .common import append_jsonl, read_jsonl
from .openai_compat import GenerationSettings, generate_text


def existing_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {str(record["query_id"]): record for record in read_jsonl(path) if record.get("query_id") is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Query teacher endpoint for QEDKS query plan.")
    parser.add_argument("--query-plan-jsonl", required=True)
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
    settings = GenerationSettings(args.temperature, args.top_p, args.max_tokens, args.seed)
    plan = read_jsonl(args.query_plan_jsonl)
    completed = existing_records(output)
    plan_by_id = {str(record["query_id"]): record for record in plan}
    unknown = sorted(set(completed) - set(plan_by_id))
    if unknown:
        raise ValueError(f"existing transcript contains query IDs absent from current plan: {unknown[:3]}")
    for query_id, record in completed.items():
        planned = plan_by_id[query_id]
        if record.get("query") != planned.get("query"):
            raise ValueError(f"existing transcript query differs from current plan for {query_id}")
        if record.get("request_model_name") != args.request_model_name:
            raise ValueError(f"existing transcript request model differs for {query_id}")
        if record.get("generation_config") != settings.__dict__:
            raise ValueError(f"existing transcript generation settings differ for {query_id}")
        prior_mode = record.get("request_mode")
        if prior_mode is not None and prior_mode != args.mode:
            raise ValueError(f"existing transcript request mode differs for {query_id}")
    if completed:
        print(f"Resuming QEDKS teacher collection with {len(completed)} saved responses from {output}")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    pending = [record for record in plan if str(record["query_id"]) not in completed]

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
            "request_mode": args.mode,
            "endpoint_base_url": args.base_url,
            "latency_seconds": latency,
            "retry_count": retry_count,
            **tokens,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for result in tqdm(executor.map(collect, pending), total=len(pending), desc="QEDKS teacher queries"):
            append_jsonl(output, result)
    print(f"Wrote QEDKS teacher transcript to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


