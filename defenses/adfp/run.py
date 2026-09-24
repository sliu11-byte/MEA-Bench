"""CLI: run ADFP fingerprinted teacher generation."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from argparse import Namespace
from pathlib import Path

from defenses.adfp.core import generate_adfp
from defenses.core.constants import DEFENSE_TYPE
from defenses.core.cost import build_cost
from defenses.core.io_utils import ensure_dir, load_queries, read_jsonl
from defenses.core.manifest import DefenseManifest


def _load_json(raw: str | None) -> dict:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return json.loads(raw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ADFP output-fingerprint generation")
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--proxy_model", required=True)
    p.add_argument("--query_pool_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--adfp_config", default=None, help="JSON file or inline JSON")
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--run_id", default=None)
    p.add_argument("--secret_key", default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--window_size", type=int, default=None)
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


def _write_limited_query_pool(src: Path, dest: Path, max_queries: int | None) -> Path:
    if max_queries is None:
        return src
    records = load_queries(src, max_queries=max_queries)
    from defenses.core.io_utils import write_jsonl

    write_jsonl(records, dest)
    return dest


def main() -> None:
    args = parse_args()
    cfg = _load_json(args.adfp_config)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_dir = Path(args.output_dir)
    if args.run_id and out_dir.name != run_id:
        out_dir = out_dir / run_id
    ensure_dir(out_dir)
    transcript_dir = ensure_dir(out_dir / "transcripts")
    artifacts_dir = ensure_dir(out_dir / "artifacts")

    query_pool_path = Path(args.query_pool_path).expanduser().resolve()
    limited_query_pool = _write_limited_query_pool(
        query_pool_path,
        artifacts_dir / "query_pool_limited.jsonl",
        args.max_queries,
    )

    gen_args = Namespace(
        teacher_model=args.teacher_model,
        proxy_model=args.proxy_model,
        input=str(limited_query_pool),
        output=str(transcript_dir / "defended_teacher.jsonl"),
        manifest=str(out_dir / "defense_manifest.json"),
        artifact_dir=str(artifacts_dir),
        secret_key=args.secret_key or cfg.get("secret_key", "adfp-benchmark-key-v1"),
        gamma=float(args.gamma if args.gamma is not None else cfg.get("gamma", 0.5)),
        window_size=int(args.window_size if args.window_size is not None else cfg.get("window_size", 2)),
        strength=float(args.strength if args.strength is not None else cfg.get("strength_lambda", cfg.get("strength", 140.0))),
        temperature=float(args.temperature if args.temperature is not None else cfg.get("temperature", 1.0)),
        top_p=float(args.top_p if args.top_p is not None else cfg.get("top_p", 1.0)),
        max_new_tokens=int(args.max_new_tokens if args.max_new_tokens is not None else cfg.get("max_new_tokens", 256)),
        batch_size=int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 8)),
    )

    t0 = time.time()
    generate_adfp(gen_args)
    elapsed = time.time() - t0

    transcript_path = Path(gen_args.output)
    records = read_jsonl(transcript_path)

    fingerprint_config = artifacts_dir / "fingerprint_config.json"
    hash_config = artifacts_dir / "hash_config.json"
    config = {
        "teacher_model": args.teacher_model,
        "proxy_model": args.proxy_model,
        "secret_key": gen_args.secret_key,
        "gamma": gen_args.gamma,
        "window_size": gen_args.window_size,
        "strength_lambda": gen_args.strength,
        "temperature": gen_args.temperature,
        "top_p": gen_args.top_p,
        "max_new_tokens": gen_args.max_new_tokens,
        "batch_size": gen_args.batch_size,
        "hash_scheme": "sha256_context_seed_v1",
        "same_tokenizer_only": True,
    }
    artifacts = {
        "hash_config": str(hash_config.resolve()),
        "fingerprint_config": str(fingerprint_config.resolve()),
    }
    manifest = DefenseManifest(
        defense="adfp",
        type=DEFENSE_TYPE["adfp"],
        teacher_model=args.teacher_model,
        query_pool=str(query_pool_path),
        output_transcript=str(transcript_path.resolve()),
        num_queries=len(records),
        config=config,
        artifacts=artifacts,
        cost=build_cost(
            wall_seconds=elapsed,
            num_queries=len(records),
            models=[args.teacher_model],
            extra_models=[args.proxy_model],
            data_files=[str(query_pool_path)],
            artifacts=artifacts,
            stage="defense_install",
            method="adfp",
        ),
        run_id=out_dir.name,
    )
    manifest.save(out_dir / "defense_manifest.json")
    print(f"[ADFP] wrote {len(records)} records -> {transcript_path}")
    print(f"[ADFP] manifest -> {out_dir / 'defense_manifest.json'}")


if __name__ == "__main__":
    main()
