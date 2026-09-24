"""CLI: run Knowledge-DuFFin detector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from attacks.core.student_model import infer_lora_base_model, load_json
from defenses.core.cost import build_cost
from defenses.core.io_utils import ensure_dir, read_jsonl, write_jsonl
from defenses.core.manifest import DetectorManifest, finalize_detector_row
from defenses.duffin.detector import (
    build_probe_prompt,
    compute_knowledge_duffin,
    load_local_model,
    query_local_model_batch,
)
from defenses.duffin.probes import load_default_probes


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Knowledge-DuFFin lineage detector")
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--student_checkpoint", default=None)
    p.add_argument("--student_base_model", default=None)
    p.add_argument("--attack_manifest", default=None)
    p.add_argument("--probe_set", default=None)
    p.add_argument("--probe_output", default=None)
    p.add_argument("--probe_source", default="mmlu_pro", choices=["mmlu_pro", "mmlu"])
    p.add_argument("--probe_split", default="test")
    p.add_argument("--probe_categories", default="biology")
    p.add_argument("--probe_seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_id", default=None)
    p.add_argument("--attack_run_id", default=None)
    p.add_argument("--label", default="unknown")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_probes", type=int, default=None)
    p.add_argument("--min_valid_rate", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--teacher_reference", default=None)
    p.add_argument("--reference_only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch_size must be positive")
    checkpoint_from_manifest, manifest_attack_run_id, attack_name = _resolve_attack_manifest(args.attack_manifest)
    student_checkpoint = args.student_checkpoint or checkpoint_from_manifest
    if not student_checkpoint and not args.reference_only:
        raise SystemExit("pass --student_checkpoint or --attack_manifest")
    student_base_model = args.student_base_model or (infer_lora_base_model(student_checkpoint) if student_checkpoint else None)
    attack_run_id = args.attack_run_id or manifest_attack_run_id
    model_id = args.model_id or attack_name or student_checkpoint

    start = time.time()
    out_dir = ensure_dir(args.output_dir)
    if args.probe_set:
        probes = read_jsonl(args.probe_set)
        probe_set_path = str(Path(args.probe_set).resolve())
        probe_source = "jsonl"
        if args.max_probes is not None:
            probes = probes[: max(0, int(args.max_probes))]
    else:
        probes = load_default_probes(
            source=args.probe_source,
            split=args.probe_split,
            categories=args.probe_categories,
            max_probes=args.max_probes,
            seed=args.probe_seed,
        )
        generated_probe_path = Path(args.probe_output) if args.probe_output else out_dir / "generated_probe_set.jsonl"
        generated_probe_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(probes, generated_probe_path)
        probe_set_path = str(generated_probe_path.resolve())
        probe_source = args.probe_source
    if not probes:
        raise SystemExit("empty DuFFin probe set")

    from defenses.duffin.reference import get_teacher_reference

    reference_path = args.teacher_reference or str(out_dir / "teacher_reference.json")
    teacher_rows, teacher_generations = get_teacher_reference(
        reference_path, probes, args.teacher_model, args.max_new_tokens, args.batch_size,
    )
    if args.reference_only:
        print(f"[DuFFin reference] complete -> {reference_path}")
        return
    student, student_tok = load_local_model(student_checkpoint, student_base_model)

    probe_rows: list[dict[str, Any]] = []
    for start_index in range(0, len(probes), args.batch_size):
        batch = probes[start_index : start_index + args.batch_size]
        prepared = [build_probe_prompt(probe) for probe in batch]
        results = query_local_model_batch(
            student, student_tok,
            [item[0] for item in prepared],
            allowed_choices=[item[1] for item in prepared],
            max_new_tokens=args.max_new_tokens,
        )
        for offset, (probe, prepared_item, result) in enumerate(zip(batch, prepared, results)):
            i = start_index + offset
            allowed = prepared_item[1]
            student_response, student_choice = result
            probe_rows.append({
                "probe_id": probe.get("probe_id", probe.get("id", str(i))),
                "category": probe.get("category", probe.get("domain", "unknown")),
                "allowed_choices": allowed,
                "teacher_response": teacher_rows[i]["teacher_response"],
                "teacher_choice": teacher_rows[i]["teacher_choice"],
                "student_response": student_response,
                "student_choice": student_choice,
                "match": teacher_rows[i]["teacher_choice"] is not None
                and teacher_rows[i]["teacher_choice"] == student_choice,
            })
        print(f"[DuFFin student] {len(probe_rows)}/{len(probes)}", flush=True)

    stats = compute_knowledge_duffin(probe_rows)
    teacher_valid = sum(row["teacher_choice"] is not None for row in probe_rows)
    student_valid = sum(row["student_choice"] is not None for row in probe_rows)
    teacher_valid_rate = teacher_valid / len(probe_rows)
    student_valid_rate = student_valid / len(probe_rows)
    validity_passed = min(teacher_valid_rate, student_valid_rate) >= args.min_valid_rate
    elapsed = time.time() - start

    result_row = finalize_detector_row(
        {
            "id": f"duffin::{model_id}",
            "model_id": model_id,
            "teacher_id": args.teacher_model,
            "student_checkpoint": student_checkpoint,
            "label": args.label,
            "score": stats["score"],
            "score_name": stats["score_name"],
            "score_direction": "higher",
            "detector": "duffin",
            "normalized_hamming_distance": stats["normalized_hamming_distance"],
            "matches": stats["matches"],
            "valid_probes": stats["valid_probes"],
            "total_probes": stats["total_probes"],
            "teacher_valid_rate": teacher_valid_rate,
            "student_valid_rate": student_valid_rate,
            "validity_passed": validity_passed,
        },
        detector="duffin",
        defense_run_id="duffin",
        attack_run_id=attack_run_id,
    )
    rows_path = out_dir / "detector_rows.jsonl"
    write_jsonl([result_row], rows_path)

    probe_rows_path = out_dir / "probe_rows.jsonl"
    write_jsonl(probe_rows, probe_rows_path)

    report = {
        "detector": "duffin",
        "variant": "knowledge_choice_similarity",
        "teacher_model": args.teacher_model,
        "teacher_reference": str(Path(reference_path).resolve()),
        "student_checkpoint": student_checkpoint,
        "student_base_model": student_base_model,
        "probe_set": probe_set_path,
        "probe_source": probe_source,
        "probe_split": args.probe_split,
        "probe_categories": args.probe_categories,
        "probe_rows": str(probe_rows_path.resolve()),
        "direction": "higher",
        "score_direction": "higher",
        "acceptance_hint": "positive student should have higher teacher-student choice similarity than unrelated negative students",
        "teacher_valid_probes": teacher_valid,
        "student_valid_probes": student_valid,
        "teacher_valid_rate": teacher_valid_rate,
        "student_valid_rate": student_valid_rate,
        "min_valid_rate": args.min_valid_rate,
        "validity_passed": validity_passed,
        **stats,
    }
    report_path = out_dir / "detector_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = DetectorManifest(
        detector="duffin",
        defense_run_id="duffin",
        attack_run_id=attack_run_id,
        student_checkpoint=student_checkpoint,
        watermark_artifacts={},
        output_report=str(report_path.resolve()),
        cost=build_cost(
            wall_seconds=elapsed,
            num_queries=teacher_generations + len(probes),
            models=([args.teacher_model] if teacher_generations else []) + [student_checkpoint],
            artifacts={"detector_report": str(report_path.resolve()), "probe_rows": str(probe_rows_path.resolve())},
            stage="detector",
            method="duffin",
            extra={
                "teacher_generations": teacher_generations,
                "student_generations": len(probes),
            },
        ),
        probe_queries=probe_set_path,
        num_probe_queries=len(probes),
        config={
            "variant": "knowledge_choice_similarity",
            "score": "knowledge_fingerprint_similarity",
            "distance": "normalized_hamming",
            "probe_source": probe_source,
            "probe_split": args.probe_split,
            "probe_categories": args.probe_categories,
            "probe_seed": args.probe_seed,
            "max_new_tokens": args.max_new_tokens,
            "max_choices": "A-P",
            "teacher_reference": str(Path(reference_path).resolve()),
            "min_valid_rate": args.min_valid_rate,
            "batch_size": args.batch_size,
        },
        students=[result_row],
    )
    manifest.save(out_dir / "detector_manifest.json")

    print("[DuFFin detector] complete")
    print(f"score={stats['score']:.6f} normalized_hamming={stats['normalized_hamming_distance']:.6f}")
    print(f"report -> {report_path}")
    if not validity_passed:
        raise SystemExit(
            f"DuFFin result is invalid: teacher_valid_rate={teacher_valid_rate:.3f}, "
            f"student_valid_rate={student_valid_rate:.3f}, required={args.min_valid_rate:.3f}"
        )


if __name__ == "__main__":
    main()
