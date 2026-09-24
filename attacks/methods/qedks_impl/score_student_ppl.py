from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .common import read_jsonl, write_jsonl


@torch.inference_mode()
def score_text(model, tokenizer, text: str, device: str) -> float:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    labels = encoded["input_ids"].clone()
    output = model(**encoded, labels=labels)
    return float(math.exp(min(20.0, output.loss.item())))


def main() -> int:
    parser = argparse.ArgumentParser(description="Score QEDKS candidate queries by student-model perplexity.")
    parser.add_argument("--query-plan-jsonl", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.student_model, torch_dtype="auto", trust_remote_code=True)
    model.to(args.device)
    model.eval()

    records = read_jsonl(args.query_plan_jsonl)
    if args.limit is not None:
        records = records[: args.limit]
    output = []
    for record in records:
        ppl = score_text(model, tokenizer, record["query"], args.device)
        output.append({"query_id": record["query_id"], "query": record["query"], "ppl": ppl})
    write_jsonl(Path(args.output_jsonl), output)
    print(f"Wrote {len(output)} QEDKS PPL scores to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


