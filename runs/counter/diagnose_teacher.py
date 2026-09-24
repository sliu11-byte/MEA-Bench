"""Run a small local-teacher generation probe without attack or student training."""

import argparse
import json
import logging
import time
from pathlib import Path

from defenses.oracle.serve_defended_teacher import CleanLocalBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--query-plan-jsonl", help="Optional existing QEDKS plan with query fields.")
    args = parser.parse_args()
    if args.num_queries <= 0 or args.max_tokens <= 0:
        parser.error("num-queries and max-tokens must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prompts = [
        "Explain how photosynthesis works, including its main stages.",
        "Describe how to design a controlled scientific experiment and interpret its results.",
        "Explain the differences between supervised learning and reinforcement learning.",
    ]
    if args.query_plan_jsonl:
        prompts = []
        with Path(args.query_plan_jsonl).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    prompts.append(json.loads(line)["query"])
                if len(prompts) >= args.num_queries:
                    break
        if not prompts:
            parser.error("query plan is empty")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": args.max_tokens}
    started = time.monotonic()
    backend = CleanLocalBackend(args.teacher_model, "cuda", settings)
    logging.info("Teacher load seconds=%.3f", time.monotonic() - started)
    request = {"temperature": 0.0, "top_p": 1.0, "max_tokens": args.max_tokens,
               "_oracle_route": "chat/completions"}
    with (output / "responses.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(min(args.num_queries, len(prompts)) if args.query_plan_jsonl else args.num_queries):
            prompt = prompts[index % len(prompts)]
            started = time.monotonic()
            row = backend.generate_one(prompt, query_id=f"probe_{index}", request=request)
            row["elapsed_seconds"] = time.monotonic() - started
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            logging.info("Probe %s completed in %.3f seconds", index + 1, row["elapsed_seconds"])
    logging.info("Probe finished: %s", output)


if __name__ == "__main__":
    main()
