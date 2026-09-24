"""CLI: run ADS defense generation."""

from __future__ import annotations

import argparse
import gc
import json
import time
import uuid
from pathlib import Path

from defenses.ads.generator import ADSGenerator
from defenses.ads.save_grad import compute_proxy_student_grads
from defenses.core.constants import DEFENSE_TYPE
from defenses.core.cost import build_cost
from defenses.core.io_utils import ensure_dir, load_queries, read_jsonl, write_jsonl
from defenses.core.manifest import DefenseManifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Antidistillation Sampling (ADS) defense")
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--query_pool_path", required=True)
    p.add_argument("--output_dir", required=True, help="outputs/defenses/ads/<run_id>")
    p.add_argument("--ads_config", default=None, help="JSON file or inline JSON string")
    p.add_argument("--proxy_student", default=None)
    p.add_argument(
        "--grad_path",
        default=None,
        help="Pre-existing student_grads.pt (required when lam>0, unless --auto_grad is set)",
    )
    p.add_argument(
        "--auto_grad",
        action="store_true",
        help="If lam>0 and --grad_path is missing/absent, generate it locally before sampling "
        "(via --grad_backend) instead of assuming it pre-exists.",
    )
    p.add_argument(
        "--grad_backend",
        choices=["lite", "official"],
        default="lite",
        help="'lite' = defenses.ads.save_grad (dependency-light reimplementation, any trl version); "
        "'official' = vendored locuslab/antidistillation-sampling save_grad.py (third_party/ads/, "
        "needs trl<0.20,>=0.16 for DataCollatorForCompletionOnlyLM)",
    )
    p.add_argument(
        "--holdout_transcript",
        default=None,
        help="JSONL of {query, response} teacher traces to compute proxy-student grads from "
        "(disjoint from --query_pool_path). If omitted with --auto_grad, queries are auto-derived "
        "from the tail of the query pool and generated with the teacher (lam=0).",
    )
    p.add_argument("--auto_grad_holdout", type=int, default=50, help="# auto-derived holdout queries")
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument(
        "--grad_model_dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="Proxy dtype for --auto_grad. Default auto uses bfloat16 on CUDA.",
    )
    p.add_argument(
        "--grad_dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="CPU accumulation/storage dtype for --auto_grad gradients.",
    )
    p.add_argument("--run_id", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _load_config(raw: str | None) -> dict:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return json.loads(raw)


def _query_id(record: dict, fallback: str) -> str:
    return str(record.get("query_id", record.get("id", fallback)))


def _checkpoint_records(records: list[dict], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(records, tmp_path)
    tmp_path.replace(path)


def _remaining_queries(queries: list[dict], records: list[dict]) -> list[dict]:
    done = {
        str(record.get("query_id", record.get("id", "")))
        for record in records
        if record.get("query_id") is not None or record.get("id") is not None
    }
    return [
        query
        for index, query in enumerate(queries)
        if _query_id(query, str(index)) not in done
    ]


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_dir = Path(args.output_dir)
    # Treat output_dir as the run directory; if --run_id is given and differs, nest under it.
    if args.run_id and out_dir.name != run_id:
        out_dir = out_dir / run_id
    ensure_dir(out_dir)
    transcripts_dir = ensure_dir(out_dir / "transcripts")
    artifacts_dir = ensure_dir(out_dir / "artifacts")

    cfg = _load_config(args.ads_config)
    if args.proxy_student:
        cfg.setdefault("proxy_student", args.proxy_student)
    if args.grad_path:
        cfg.setdefault("grad_path", args.grad_path)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.max_new_tokens is not None:
        cfg.setdefault("max_new_tokens", int(args.max_new_tokens))
    if args.temperature is not None:
        cfg.setdefault("tau", float(args.temperature))
    if args.top_p is not None:
        cfg.setdefault("top_p", float(args.top_p))

    all_queries = load_queries(args.query_pool_path)
    queries = all_queries[: args.max_queries] if args.max_queries is not None else all_queries

    proxy_student = cfg.get("proxy_student") or args.proxy_student or args.teacher_model
    grad_path = cfg.get("grad_path") or args.grad_path
    lam = float(cfg.get("lam", 0.0))
    proxy_used = False

    if lam != 0 and (not grad_path or not Path(grad_path).exists()):
        if not args.auto_grad:
            raise SystemExit(
                f"lam={lam} but --grad_path is missing/absent ({grad_path!r}). "
                "Pass --auto_grad to generate it locally, or precompute it with "
                "`python -m defenses.ads.save_grad` / `--grad_backend official`."
            )
        print(f"[ADS] --auto_grad: computing proxy-student grads (backend={args.grad_backend}) ...")
        if args.holdout_transcript:
            holdout_records = read_jsonl(args.holdout_transcript)
            holdout_artifact_path = Path(args.holdout_transcript)
        else:
            n_hold = args.auto_grad_holdout
            holdout_queries = all_queries[len(queries) : len(queries) + n_hold]
            if len(holdout_queries) < n_hold:
                raise SystemExit(
                    f"Query pool too small for a disjoint {n_hold}-query holdout after the "
                    f"{len(queries)} queries being defended; pass --holdout_transcript explicitly."
                )
            print(f"[ADS] generating {len(holdout_queries)} holdout traces from teacher (lam=0) ...")
            holdout_gen = ADSGenerator(
                teacher_model=args.teacher_model,
                proxy_student=proxy_student,
                grad_path=None,
                device=args.device,
                system_prompt=cfg.get("system_prompt"),
            )
            holdout_records = holdout_gen.generate(
                holdout_queries,
                config={
                    "lam": 0.0,
                    "tau": cfg.get("tau", 0.7),
                    "top_p": cfg.get("top_p", 0.95),
                    "max_new_tokens": cfg.get("max_new_tokens", 256),
                    "batch_size": cfg.get("batch_size", 1),
                },
            )
            holdout_artifact_path = artifacts_dir / "holdout_teacher.jsonl"
            write_jsonl(holdout_records, holdout_artifact_path)
            del holdout_gen
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if args.grad_backend == "official":
            from defenses.ads.save_grad_official import run_official_save_grad

            grad_file = run_official_save_grad(
                proxy_student, holdout_records, exp_dir=artifacts_dir, seed=cfg.get("seed", 0)
            )
        else:
            grad_manifest = compute_proxy_student_grads(
                proxy_student,
                holdout_records,
                output_path=artifacts_dir / "student_grads.pt",
                max_length=cfg.get("grad_max_length", 512),
                device=args.device,
                model_dtype=args.grad_model_dtype,
                grad_dtype=args.grad_dtype,
            )
            grad_file = Path(grad_manifest["output_path"])
            print(
                f"[ADS] grad_norm={grad_manifest['grad_norm']:.3e} "
                f"(n={grad_manifest['num_samples_used']}) -> {grad_file}"
            )

        grad_path = str(grad_file)
        cfg["grad_path"] = grad_path
        cfg["proxy_student"] = proxy_student
        proxy_used = True
        print(f"[ADS] using auto-generated grad_path -> {grad_path}")

    if lam != 0:
        proxy_used = True

    gen = ADSGenerator(
        teacher_model=args.teacher_model,
        proxy_student=proxy_student,
        grad_path=grad_path,
        device=args.device,
        system_prompt=cfg.get("system_prompt"),
    )

    t0 = time.time()
    transcript_path = transcripts_dir / "defended_teacher.jsonl"
    records = read_jsonl(transcript_path) if transcript_path.exists() else []
    if records:
        print(f"[ADS] resume: loaded {len(records)} existing defended responses from {transcript_path}", flush=True)
    pending_queries = _remaining_queries(queries, records)
    batch_size = max(1, int(cfg.get("batch_size", 1)))
    total_batches = (len(pending_queries) + batch_size - 1) // batch_size if pending_queries else 0
    for batch_index, start in enumerate(range(0, len(pending_queries), batch_size), start=1):
        batch_queries = pending_queries[start : start + batch_size]
        print(
            f"[ADS] defended batch {batch_index}/{total_batches} start: "
            f"size={len(batch_queries)}, existing={len(records)}",
            flush=True,
        )
        batch_records = gen.generate(batch_queries, config=cfg)
        records.extend(batch_records)
        _checkpoint_records(records, transcript_path)
        print(
            f"[ADS] defended batch {batch_index}/{total_batches} done: total={len(records)}",
            flush=True,
        )
    elapsed = time.time() - t0

    _checkpoint_records(records, transcript_path)

    config_path = artifacts_dir / "ads_config.json"
    config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    artifacts = {
        "ads_config": str(config_path.resolve()),
    }
    if cfg.get("grad_path"):
        artifacts["grad_path"] = str(Path(cfg["grad_path"]).resolve())
    if args.holdout_transcript:
        artifacts["holdout_transcript"] = str(Path(args.holdout_transcript).resolve())
    elif (artifacts_dir / "holdout_teacher.jsonl").exists():
        artifacts["holdout_transcript"] = str((artifacts_dir / "holdout_teacher.jsonl").resolve())

    caches = None
    if cfg.get("grad_path"):
        caches = [str(Path(cfg["grad_path"]).resolve())]
    extra_models = [proxy_student] if proxy_used and proxy_student else None

    cost = build_cost(
        wall_seconds=elapsed,
        num_queries=len(records),
        models=[args.teacher_model],
        extra_models=extra_models,
        data_files=[str(Path(args.query_pool_path).resolve())],
        artifacts={k: str(v) for k, v in artifacts.items()},
        caches=caches,
        device=args.device,
        stage="defense_install",
        method="ads",
    )

    manifest = DefenseManifest(
        defense="ads",
        type=DEFENSE_TYPE["ads"],
        teacher_model=args.teacher_model,
        query_pool=str(Path(args.query_pool_path).resolve()),
        output_transcript=str(transcript_path.resolve()),
        num_queries=len(records),
        config=cfg,
        artifacts=artifacts,
        cost=cost,
        run_id=out_dir.name,
    )
    manifest.save(out_dir / "defense_manifest.json")
    print(f"[ADS] wrote {len(records)} records -> {transcript_path}")
    print(f"[ADS] manifest -> {out_dir / 'defense_manifest.json'}")


if __name__ == "__main__":
    main()
