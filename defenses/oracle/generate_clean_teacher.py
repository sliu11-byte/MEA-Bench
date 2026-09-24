"""Generate a resumable clean transcript before loading the attack student."""

import argparse
import json
import time
from pathlib import Path

from defenses.core.io_utils import ensure_dir, read_jsonl
from defenses.oracle.serve_defended_teacher import CleanLocalBackend, load_json_arg


def generate(args, backend_factory=CleanLocalBackend):
    output = ensure_dir(Path(args.output_dir))
    artifacts = ensure_dir(output / "artifacts")
    transcript = ensure_dir(output / "transcripts") / "defended_teacher.jsonl"
    queries = read_jsonl(args.query_pool_path)[:args.max_queries]
    if not queries:
        raise ValueError("clean query pool is empty")
    settings = {**load_json_arg(args.defense_config), "temperature": args.temperature,
                "top_p": args.top_p, "max_new_tokens": args.max_new_tokens}
    config = {"teacher_model": args.teacher_model, "mode": args.mode, "settings": settings}
    config_path = artifacts / "clean_generation_config.json"
    completed = read_jsonl(transcript) if transcript.exists() else []
    if completed and (not config_path.exists() or json.loads(config_path.read_text()) != config):
        raise ValueError("existing clean transcript was generated with a different configuration")
    if len(completed) > len(queries):
        raise ValueError("existing clean transcript is longer than the query pool")
    for row, query in zip(completed, queries):
        if row.get("query_id") != query["query_id"] or row.get("query") != query["query"]:
            raise ValueError("existing clean transcript does not match the ordered queries")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[clean] Transcript: {transcript}; completed={len(completed)}/{len(queries)}", flush=True)
    if len(completed) < len(queries):
        backend = backend_factory(args.teacher_model, args.device, settings)
        device_map = getattr(getattr(backend, "model", None), "hf_device_map", {})
        print(f"[clean] Teacher device map: {device_map}", flush=True)
        if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
            print("[clean] Teacher has CPU/disk offload; token generation may be very slow.", flush=True)
        request = {"temperature": args.temperature, "top_p": args.top_p,
                   "max_tokens": args.max_new_tokens,
                   "_oracle_route": "completions" if args.mode == "completion" else "chat/completions"}
        with transcript.open("a", encoding="utf-8") as handle:
            for index, query in enumerate(queries[len(completed):], start=len(completed) + 1):
                started = time.monotonic()
                print(f"[clean] Generating {index}/{len(queries)}: {query['query_id']}", flush=True)
                row = backend.generate_one(query["query"], query_id=query["query_id"], request=request)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[clean] Completed {index}/{len(queries)} in {time.monotonic()-started:.1f}s", flush=True)
    manifest = {"defense": "clean", "teacher_model": args.teacher_model,
                "transcript": str(transcript), "num_queries": len(queries), "config": config}
    (output / "defense_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--query_pool_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_queries", type=int, required=True)
    parser.add_argument("--defense_config", default="{}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--mode", choices=["chat", "completion"], default="chat")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
