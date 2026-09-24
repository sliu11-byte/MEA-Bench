"""CLI: run ADFP detector on attack student checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from attacks.core.student_model import infer_lora_base_model, load_json
from defenses.adfp.core import detect_adfp
from defenses.core.cost import build_cost
from defenses.core.io_utils import ensure_dir, read_jsonl, write_jsonl
from defenses.core.manifest import DetectorManifest, finalize_detector_row


def _resolve_attack_manifest(path: str | None) -> tuple[str | None, str | None, str | None]:
    if path is None:
        return None, None, None
    manifest = load_json(path)
    result = manifest.get("result") or {}
    checkpoint = result.get("checkpoint_dir") or manifest.get("checkpoint_dir")
    attack_run_id = result.get("run_id") or manifest.get("run_id")
    attack_name = result.get("attack") or manifest.get("attack")
    if not checkpoint:
        raise ValueError(f"attack manifest has no checkpoint_dir: {path}")
    return str(checkpoint), None if attack_run_id is None else str(attack_run_id), None if attack_name is None else str(attack_name)


def _fingerprint_config_from_artifacts(path: str | None, explicit: str | None) -> str:
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    if not path:
        raise ValueError("pass --fingerprint_config or --fingerprint_artifacts")
    artifacts = Path(path).expanduser().resolve()
    candidate = artifacts / "fingerprint_config.json"
    if not candidate.exists():
        raise FileNotFoundError(f"fingerprint_config.json not found under {artifacts}")
    return str(candidate)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ADFP checkpoint detector")
    p.add_argument("--fingerprint_artifacts", default=None, help="artifacts/ dir from ADFP generation")
    p.add_argument("--fingerprint_config", default=None)
    p.add_argument("--transcript", required=True, help="ADFP defended_teacher.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--student_checkpoint", default=None)
    p.add_argument("--student_base_model", default=None)
    p.add_argument("--attack_manifest", default=None)
    p.add_argument("--model_id", default=None)
    p.add_argument("--label", default="unknown")
    p.add_argument("--defense_run_id", default=None)
    p.add_argument("--attack_run_id", default=None)
    p.add_argument("--max_contexts", type=int, default=1000)
    p.add_argument("--min_context_tokens", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_from_manifest, manifest_attack_run_id, attack_name = _resolve_attack_manifest(args.attack_manifest)
    student_checkpoint = args.student_checkpoint or checkpoint_from_manifest
    if not student_checkpoint:
        raise SystemExit("pass --student_checkpoint or --attack_manifest")
    student_base_model = args.student_base_model or infer_lora_base_model(student_checkpoint)
    attack_run_id = args.attack_run_id or manifest_attack_run_id
    fingerprint_config = _fingerprint_config_from_artifacts(args.fingerprint_artifacts, args.fingerprint_config)
    out_dir = ensure_dir(args.output_dir)
    defense_run_id = args.defense_run_id or (Path(args.fingerprint_artifacts).parent.name if args.fingerprint_artifacts else None)

    t0 = time.time()
    core_args = Namespace(
        student_model=student_checkpoint,
        student_base_model=student_base_model,
        transcript=args.transcript,
        fingerprint_config=fingerprint_config,
        output_dir=str(out_dir),
        model_id=args.model_id or attack_name or student_checkpoint,
        label=args.label,
        defense_run_id=defense_run_id,
        attack_run_id=attack_run_id,
        max_contexts=args.max_contexts,
        min_context_tokens=args.min_context_tokens,
    )
    detect_adfp(core_args)
    elapsed = time.time() - t0

    rows_path = out_dir / "detector_rows.jsonl"
    rows = [
        finalize_detector_row(row, detector="adfp", defense_run_id=defense_run_id, attack_run_id=attack_run_id)
        for row in read_jsonl(rows_path)
    ]
    write_jsonl(rows, rows_path)

    report_path = out_dir / "detector_report.json"
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    report["students"] = rows
    report["direction"] = "higher"
    report["score_direction"] = "higher"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    detector_manifest = DetectorManifest(
        detector="adfp",
        defense_run_id=defense_run_id or "unknown",
        attack_run_id=attack_run_id,
        student_checkpoint=student_checkpoint,
        watermark_artifacts={
            "fingerprint_config": fingerprint_config,
            "fingerprint_artifacts": "" if args.fingerprint_artifacts is None else str(Path(args.fingerprint_artifacts).resolve()),
        },
        output_report=str(report_path.resolve()),
        cost=build_cost(
            wall_seconds=elapsed,
            tokens_scored=report.get("num_contexts"),
            models=[student_checkpoint],
            artifacts={"detector_report": str(report_path.resolve())},
            stage="defense_install",
            method="adfp",
        ),
        config={
            "max_contexts": args.max_contexts,
            "min_context_tokens": args.min_context_tokens,
            "same_tokenizer_only": True,
        },
        students=rows,
    )
    detector_manifest.save(out_dir / "detector_manifest.json")
    print(f"[ADFP detector] report -> {report_path}")


if __name__ == "__main__":
    main()
