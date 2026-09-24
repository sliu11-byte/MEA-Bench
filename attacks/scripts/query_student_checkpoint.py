from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attacks.core.student_model import (
    GenerationConfig,
    load_student_from_checkpoint,
    load_student_from_manifest,
)


def iter_queries(args: argparse.Namespace) -> Iterable[tuple[str | None, str]]:
    if args.query is not None:
        yield None, args.query
    if args.query_file is not None:
        path = Path(args.query_file)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                text = line.rstrip("\n")
                if text.strip():
                    yield str(line_no), text


def run_interactive(args: argparse.Namespace, student) -> None:
    print("Interactive student checkpoint query mode. Press Ctrl-D or enter an empty line to exit.")
    while True:
        try:
            query = input("query> ").strip()
        except EOFError:
            print()
            return
        if not query:
            return
        print(student.generate_one(query, generation_config=build_generation_config(args)))


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        use_chat_template=args.use_chat_template,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a Stage-1 attack student checkpoint and query it.")
    parser.add_argument("--checkpoint", help="Full HF checkpoint or LoRA adapter checkpoint directory.")
    parser.add_argument("--manifest", help="Attack run attack_manifest.json; uses its checkpoint_dir.")
    parser.add_argument("--base-model", help="Required for LoRA if adapter_config.json cannot infer it.")
    parser.add_argument("--student-name", default="student", help="Name stored in JSONL outputs.")
    parser.add_argument("--query", help="Single query string.")
    parser.add_argument("--query-file", help="UTF-8 text file with one query per line.")
    parser.add_argument("--output-jsonl", help="Optional JSONL output path for batch/single queries.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive prompt loop.")
    parser.add_argument("--dtype", default="auto", help="torch_dtype passed to transformers, e.g. auto, bfloat16, float16.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true", help="Use sampling. Default is greedy decoding.")
    parser.add_argument("--use-chat-template", action="store_true", help="Use tokenizer chat template when available.")
    parser.add_argument("--print-metadata", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint and not args.manifest:
        raise SystemExit("Provide --checkpoint or --manifest.")
    if args.checkpoint and args.manifest:
        raise SystemExit("Provide only one of --checkpoint or --manifest.")
    if not any([args.query, args.query_file, args.interactive, args.print_metadata]):
        raise SystemExit("Provide --query, --query-file, --interactive, or --print-metadata.")

    if args.manifest:
        student = load_student_from_manifest(
            args.manifest,
            base_model=args.base_model,
            student_name=None if args.student_name == "student" else args.student_name,
            dtype=args.dtype,
            device_map=args.device_map,
        )
    else:
        student = load_student_from_checkpoint(
            args.checkpoint,
            base_model=args.base_model,
            student_name=args.student_name,
            dtype=args.dtype,
            device_map=args.device_map,
        )

    metadata = student.info.to_dict() | {"use_chat_template": bool(args.use_chat_template)}
    if args.print_metadata:
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        if not any([args.query, args.query_file, args.interactive]):
            return 0

    generation_config = build_generation_config(args)
    output_handle = None
    if args.output_jsonl:
        output_path = Path(args.output_jsonl).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("a", encoding="utf-8")

    try:
        for query_id, query in iter_queries(args):
            response = student.generate_one(query, generation_config=generation_config)
            record = {
                **metadata,
                "generation_config": generation_config.to_dict(),
                "query_id": query_id,
                "query": query,
                "response": response,
            }
            if output_handle is not None:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
            else:
                print(response)
        if args.interactive:
            run_interactive(args, student)
    finally:
        if output_handle is not None:
            output_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
