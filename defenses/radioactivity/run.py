"""CLI: run Radioactivity watermarked teacher generation."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from defenses.core.constants import DEFENSE_TYPE
from defenses.core.cost import build_cost
from defenses.core.io_utils import ensure_dir, load_queries, write_jsonl
from defenses.core.manifest import DefenseManifest
from defenses.radioactivity.generator import RadioactivityGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Radioactivity (Maryland/KGW) defense generation")
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--query_pool_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--watermark_config", default=None)
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--run_id", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--no_filter", action="store_true", help="Skip building filter.pkl")
    return p.parse_args()


def _load_json(raw: str | None) -> dict:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_dir = Path(args.output_dir)
    if args.run_id and out_dir.name != run_id:
        out_dir = out_dir / run_id
    ensure_dir(out_dir)
    transcripts_dir = ensure_dir(out_dir / "transcripts")
    artifacts_dir = ensure_dir(out_dir / "artifacts")

    wm_cfg = _load_json(args.watermark_config)
    gen_cfg = {}
    if args.temperature is not None:
        gen_cfg["temperature"] = args.temperature
    if args.max_new_tokens is not None:
        gen_cfg["max_new_tokens"] = args.max_new_tokens
    for k in ("temperature", "top_p", "max_new_tokens", "system_prompt"):
        if k in wm_cfg:
            gen_cfg[k] = wm_cfg[k]

    queries = load_queries(args.query_pool_path, max_queries=args.max_queries)
    gen = RadioactivityGenerator(
        teacher_model=args.teacher_model,
        device=args.device,
        system_prompt=wm_cfg.get("system_prompt"),
        watermark_config=wm_cfg,
    )

    t0 = time.time()
    records = gen.generate(queries, config=gen_cfg)
    artifact_paths = gen.save_artifacts(artifacts_dir, build_filter=not args.no_filter)
    elapsed = time.time() - t0

    transcript_path = transcripts_dir / "defended_teacher.jsonl"
    write_jsonl(records, transcript_path)

    artifacts = dict(artifact_paths or {})

    cost = build_cost(
        wall_seconds=elapsed,
        num_queries=len(records),
        models=[args.teacher_model],
        data_files=[str(Path(args.query_pool_path).resolve())],
        artifacts={k: str(v) for k, v in artifacts.items()},
        device=args.device,
        stage="defense_install",
        method="radioactivity",
    )

    manifest = DefenseManifest(
        defense="radioactivity",
        type=DEFENSE_TYPE["radioactivity"],
        teacher_model=args.teacher_model,
        query_pool=str(Path(args.query_pool_path).resolve()),
        output_transcript=str(transcript_path.resolve()),
        num_queries=len(records),
        config={**wm_cfg, **gen_cfg},
        artifacts=artifacts,
        cost=cost,
        run_id=out_dir.name,
    )
    manifest.save(out_dir / "defense_manifest.json")
    print(f"[Radioactivity] wrote {len(records)} records -> {transcript_path}")
    print(f"[Radioactivity] artifacts -> {artifacts_dir}")


if __name__ == "__main__":
    main()
