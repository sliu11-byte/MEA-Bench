"""CLI: run GINSEW detector on one or more student checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from defenses.core.cost import build_cost
from defenses.core.io_utils import ensure_dir, write_jsonl
from defenses.core.manifest import DetectorManifest, finalize_detector_row
from defenses.ginsew.detector import GinsewDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GINSEW watermark detector")
    p.add_argument("--watermark_artifacts", required=True, help="artifacts/ dir from ginsew run")
    p.add_argument("--probe_queries", required=False, default=None, help="Required for --student_checkpoint/--attack_manifest")
    p.add_argument("--output_dir", required=True, help=".../detector/")
    p.add_argument("--student_checkpoint", action="append", default=[], help="Repeatable")
    p.add_argument("--attack_manifest", action="append", default=[], help="Repeatable")
    p.add_argument(
        "--student_outputs",
        action="append",
        default=[],
        help="Black-box path (spec input 'B'): repeatable student_outputs.jsonl "
        "(rows with token_ids/output_ids, or response text + --outputs_tokenizer), no checkpoint needed",
    )
    p.add_argument(
        "--outputs_tokenizer",
        default=None,
        help="HF tokenizer name/path to re-tokenize --student_outputs 'response' text "
        "when rows lack token_ids/output_ids",
    )
    p.add_argument("--outputs_label", action="append", default=[], help="Labels aligned with --student_outputs")
    p.add_argument("--attack_run_id", default=None)
    p.add_argument("--defense_run_id", default=None)
    p.add_argument(
        "--label",
        action="append",
        default=[],
        help="Matrix labels aligned with students: positive|negative "
        "(aliases watermarked/radioactive/defended→positive, clean→negative)",
    )
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--qmin", type=float, default=0.0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _safe_label(label: Optional[str], idx: int) -> str:
    if label:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        return safe or f"s{idx}"
    return f"s{idx}"


def _write_student_outputs(
    out_dir: Path,
    per_student: List[List[Dict[str, Any]]],
    labels: List[Optional[str]],
) -> Optional[str]:
    if not per_student:
        return None
    combined: List[Dict[str, Any]] = []
    multi = len(per_student) > 1
    for i, outs in enumerate(per_student):
        if not outs:
            continue
        if multi:
            per_path = out_dir / f"student_outputs_{_safe_label(labels[i] if i < len(labels) else None, i)}.jsonl"
            write_jsonl(outs, per_path)
        for row in outs:
            tagged = dict(row)
            if labels[i] is not None:
                tagged.setdefault("label", labels[i])
            combined.append(tagged)
    if not combined:
        return None
    combined_path = out_dir / "student_outputs.jsonl"
    write_jsonl(combined, combined_path)
    return str(combined_path.resolve())


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    detector = GinsewDetector(args.watermark_artifacts, qmin=args.qmin)

    specs = []
    for ckpt in args.student_checkpoint:
        specs.append({"checkpoint": ckpt, "manifest": None})
    for mf in args.attack_manifest:
        specs.append({"checkpoint": None, "manifest": mf})
    if not specs and not args.student_outputs:
        raise SystemExit(
            "Provide at least one --student_checkpoint / --attack_manifest (checkpoint path) "
            "or --student_outputs (black-box path)"
        )
    if specs and not args.probe_queries:
        raise SystemExit("--probe_queries is required when using --student_checkpoint/--attack_manifest")

    defense_run_id = args.defense_run_id or Path(args.watermark_artifacts).parent.name

    labels = list(args.label)
    while len(labels) < len(specs):
        labels.append(None)

    students: List[dict] = []
    collected_outputs: List[List[Dict[str, Any]]] = []
    tokens_scored = 0
    num_queries = 0
    models: List[str] = []
    t0 = time.time()
    for i, spec in enumerate(specs):
        import torch
        torch.manual_seed(args.seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + i)
        row = detector.detect_checkpoint(
            student_checkpoint=spec["checkpoint"],
            attack_manifest=spec["manifest"],
            probe_queries=args.probe_queries,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_queries=args.max_queries,
            label=labels[i],
        )
        outs = row.pop("student_outputs", None) or []
        row.pop("probe_pairs", None)
        collected_outputs.append(outs)
        tokens_scored += int(row.get("scored_tokens") or row.get("num_tokens") or 0)
        num_queries += int(row.get("num_probe_queries") or 0)
        if row.get("student_checkpoint"):
            models.append(str(row["student_checkpoint"]))
        row = finalize_detector_row(
            row, detector="ginsew", defense_run_id=defense_run_id, attack_run_id=args.attack_run_id
        )
        students.append(row)
        print(
            f"[GINSEW detector] {row['student']}: score={row['score']:.4f} "
            f"psnr={row['psnr']:.4f} z={row['z_score']:.4f} green_rate={row['green_rate']:.4f}"
        )

    if args.student_outputs:
        outputs_tokenizer = None
        if args.outputs_tokenizer:
            from transformers import AutoTokenizer

            outputs_tokenizer = AutoTokenizer.from_pretrained(args.outputs_tokenizer)
        outputs_labels = list(args.outputs_label)
        while len(outputs_labels) < len(args.student_outputs):
            outputs_labels.append(None)
        for i, outputs_path in enumerate(args.student_outputs):
            row = detector.detect_outputs(
                outputs_path, tokenizer=outputs_tokenizer, label=outputs_labels[i]
            )
            tokens_scored += int(row.get("num_tokens") or row.get("scored_tokens") or 0)
            row = finalize_detector_row(
                row, detector="ginsew", defense_run_id=defense_run_id, attack_run_id=args.attack_run_id
            )
            students.append(row)
            print(
                f"[GINSEW detector] (black-box) {outputs_path}: score={row['score']:.4f} "
                f"green_rate={row.get('green_rate')} z={row.get('z_score')}"
            )
    elapsed = time.time() - t0

    student_outputs_path = _write_student_outputs(out_dir, collected_outputs, labels)

    rows_path = out_dir / "detector_rows.jsonl"
    write_jsonl(students, rows_path)

    report = {
        "detector": "ginsew",
        "num_students": len(students),
        "students": students,
        "direction": "higher",
        "score_direction": "higher",
        "acceptance_hint": "positive (watermarked) student should score higher than negative (clean)",
        "student_outputs": student_outputs_path,
        "matrix_m4": {
            "compatible": True,
            "required_fields": ["id", "label", "score", "direction", "detector"],
            "label_values": ["positive", "negative"],
            "direction": "higher",
        },
    }
    report_path = out_dir / "detector_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    cost = build_cost(
        wall_seconds=elapsed,
        num_queries=num_queries or None,
        tokens_scored=tokens_scored or None,
        models=models or None,
        artifacts={"detector_report": str(report_path.resolve())},
        device=args.device,
        stage="defense_install",
        method="ginsew",
    )

    manifest = DetectorManifest(
        detector="ginsew",
        defense_run_id=defense_run_id,
        attack_run_id=args.attack_run_id,
        student_checkpoint=args.student_checkpoint[0] if args.student_checkpoint else None,
        watermark_artifacts={"dir": str(Path(args.watermark_artifacts).resolve())},
        output_report=str(report_path.resolve()),
        cost=cost,
        probe_queries=str(Path(args.probe_queries).resolve()) if args.probe_queries else None,
        num_probe_queries=args.max_queries,
        config={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "qmin": args.qmin,
        },
        students=students,
        student_outputs=student_outputs_path,
    )
    manifest.save(out_dir / "detector_manifest.json")
    print(f"[GINSEW detector] report -> {report_path}")


if __name__ == "__main__":
    main()
