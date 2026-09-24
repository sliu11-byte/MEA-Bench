"""CLI: run Trace Rewriting defense generation."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from defenses.core.constants import DEFENSE_TYPE
from defenses.core.cost import build_cost
from defenses.core.io_utils import ensure_dir, load_queries, read_jsonl, write_jsonl
from defenses.core.manifest import DefenseManifest
from defenses.trace_rewriting.generator import TraceRewritingGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Trace Rewriting anti-distillation generation")
    p.add_argument("--teacher_model", default=None)
    p.add_argument("--teacher_backend", choices=["local_hf", "openai_compatible"], default="local_hf")
    p.add_argument("--teacher_base_url", default=None, help="OpenAI-compatible teacher base URL, e.g. http://localhost:8000/v1")
    p.add_argument("--teacher_request_model", default=None, help="Model name sent to the teacher endpoint; defaults to --teacher_model")
    p.add_argument("--teacher_api_key", default="EMPTY")
    p.add_argument("--rewriter_model", required=True)
    p.add_argument("--rewriter_backend", choices=["local_hf", "openai_compatible"], default="local_hf")
    p.add_argument("--rewriter_base_url", default=None, help="OpenAI-compatible rewriter base URL, e.g. http://localhost:8001/v1")
    p.add_argument("--rewriter_request_model", default=None, help="Model name sent to the rewriter endpoint; defaults to --rewriter_model")
    p.add_argument("--rewriter_api_key", default="EMPTY")
    p.add_argument("--query_pool_path", default=None, help="JSON/JSONL query pool; teacher is queried first")
    p.add_argument(
        "--clean_transcript_path",
        default=None,
        help="JSONL with query plus response/original_response; skips teacher generation",
    )
    p.add_argument("--output_dir", required=True, help="outputs/defenses/trace_rewriting/<run_id>")
    p.add_argument("--trace_rewriting_config", default=None, help="JSON file or inline JSON string")
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--run_id", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--temperature", type=float, default=None, help="Rewriter temperature")
    p.add_argument("--max_new_tokens", type=int, default=None, help="Rewriter max_new_tokens")
    return p.parse_args()


def _load_json(raw: str | None) -> dict:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return json.loads(raw)


def _pick(d: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if d.get(key) is not None:
            return d[key]
    return None


def _load_clean_transcript(path: str, max_queries: int | None = None) -> List[Dict[str, str]]:
    records = read_jsonl(path)
    if max_queries is not None:
        records = records[: max(0, int(max_queries))]
    out: List[Dict[str, str]] = []
    for i, rec in enumerate(records):
        query = _pick(rec, "query", "prompt", "question", "problem", "instruction", "text", "input")
        original = _pick(rec, "original_response", "response", "output", "answer", "completion")
        if query is None:
            raise ValueError(f"Clean transcript record {i} missing query/prompt text")
        if original is None:
            raise ValueError(f"Clean transcript record {i} missing response/original_response")
        qid = _pick(rec, "query_id", "id", "qid", "uid", "example_id", "idx")
        out.append(
            {
                "query_id": str(qid if qid is not None else i),
                "query": str(query),
                "original_response": str(original),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    if not args.query_pool_path and not args.clean_transcript_path:
        raise SystemExit("Pass either --query_pool_path or --clean_transcript_path")
    if args.query_pool_path and args.clean_transcript_path:
        raise SystemExit("Pass only one of --query_pool_path / --clean_transcript_path")
    if args.query_pool_path and not args.teacher_model:
        raise SystemExit("--teacher_model is required when using --query_pool_path")
    if args.teacher_backend == "openai_compatible" and args.query_pool_path and not args.teacher_base_url:
        raise SystemExit("--teacher_base_url is required when teacher_backend=openai_compatible")
    if args.rewriter_backend == "openai_compatible" and not args.rewriter_base_url:
        raise SystemExit("--rewriter_base_url is required when rewriter_backend=openai_compatible")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_dir = Path(args.output_dir)
    if args.run_id and out_dir.name != run_id:
        out_dir = out_dir / run_id
    ensure_dir(out_dir)
    transcripts_dir = ensure_dir(out_dir / "transcripts")
    artifacts_dir = ensure_dir(out_dir / "artifacts")

    cfg = _load_json(args.trace_rewriting_config)
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens

    if args.clean_transcript_path:
        queries = _load_clean_transcript(args.clean_transcript_path, max_queries=args.max_queries)
        load_teacher = False
        query_pool = ""
        input_clean_transcript = str(Path(args.clean_transcript_path).resolve())
    else:
        queries = load_queries(args.query_pool_path, max_queries=args.max_queries)
        load_teacher = True
        query_pool = str(Path(args.query_pool_path).resolve())
        input_clean_transcript = None

    gen = TraceRewritingGenerator(
        teacher_model=args.teacher_model,
        rewriter_model=args.rewriter_model,
        device=args.device,
        teacher_system_prompt=cfg.get("teacher_system_prompt"),
        rewriter_system_prompt=cfg.get("rewriter_system_prompt"),
        load_teacher=load_teacher,
        teacher_backend=args.teacher_backend,
        teacher_base_url=args.teacher_base_url,
        teacher_api_key=args.teacher_api_key,
        teacher_request_model=args.teacher_request_model,
        rewriter_backend=args.rewriter_backend,
        rewriter_base_url=args.rewriter_base_url,
        rewriter_api_key=args.rewriter_api_key,
        rewriter_request_model=args.rewriter_request_model,
    )

    t0 = time.time()
    records = gen.generate(queries, config=cfg)
    elapsed = time.time() - t0

    transcript_path = transcripts_dir / "defended_teacher.jsonl"
    write_jsonl(records, transcript_path)

    cfg_path = artifacts_dir / "trace_rewriting_config.json"
    resolved_cfg = records[0]["trace_rewriting_config"] if records else cfg
    manifest_cfg = {
        **resolved_cfg,
        "teacher_backend": args.teacher_backend,
        "rewriter_backend": args.rewriter_backend,
        "teacher_base_url": args.teacher_base_url,
        "rewriter_base_url": args.rewriter_base_url,
        "teacher_request_model": args.teacher_request_model,
        "rewriter_request_model": args.rewriter_request_model,
    }
    cfg_path.write_text(json.dumps(manifest_cfg, indent=2) + "\n", encoding="utf-8")

    empty_rewrites = sum(1 for r in records if not str(r.get("response") or "").strip())
    empty_originals = sum(1 for r in records if not str(r.get("original_response") or "").strip())
    report = {
        "num_records": len(records),
        "rewrite_success_count": len(records) - empty_rewrites,
        "empty_rewrite_count": empty_rewrites,
        "empty_original_response_count": empty_originals,
        "error_count": 0,
        "rewrite_success_rate": (len(records) - empty_rewrites) / len(records) if records else None,
        "rewrite_scope": resolved_cfg.get("rewrite_scope", "full_answer"),
        "teacher_backend": args.teacher_backend,
        "rewriter_backend": args.rewriter_backend,
    }
    report_path = artifacts_dir / "rewrite_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    artifacts = {
        "trace_rewriting_config": str(cfg_path.resolve()),
        "rewrite_report": str(report_path.resolve()),
    }

    data_files = []
    if args.query_pool_path:
        data_files.append(str(Path(args.query_pool_path).resolve()))
    if args.clean_transcript_path:
        data_files.append(str(Path(args.clean_transcript_path).resolve()))

    cost = build_cost(
        wall_seconds=elapsed,
        num_queries=len(records),
        models=[args.teacher_model] if args.teacher_model and load_teacher else [],
        extra_models=[args.rewriter_model],
        data_files=data_files,
        artifacts={k: str(v) for k, v in artifacts.items()},
        device=args.device,
        stage="defense_install",
        method="trace_rewriting",
        extra={
            "teacher_query_count": len(records) if load_teacher else 0,
            "rewriter_query_count": len(records),
        },
    )

    manifest = DefenseManifest(
        defense="trace_rewriting",
        type=DEFENSE_TYPE["trace_rewriting"],
        teacher_model=args.teacher_model or "unknown",
        rewriter_model=args.rewriter_model,
        query_pool=query_pool,
        input_clean_transcript=input_clean_transcript,
        output_transcript=str(transcript_path.resolve()),
        num_queries=len(records),
        config=manifest_cfg,
        artifacts=artifacts,
        cost=cost,
        run_id=out_dir.name,
    )
    manifest.save(out_dir / "defense_manifest.json")
    print(f"[TraceRewriting] wrote {len(records)} records -> {transcript_path}")
    print(f"[TraceRewriting] manifest -> {out_dir / 'defense_manifest.json'}")


if __name__ == "__main__":
    main()
