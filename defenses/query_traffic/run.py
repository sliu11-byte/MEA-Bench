"""CLI wrapper for MMD / PRADA / SEAT query-traffic detectors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from defenses.core.cost import build_cost
from defenses.core.io_utils import ensure_dir, write_jsonl
from defenses.core.manifest import DetectorManifest, finalize_detector_row
from defenses.query_traffic.common import normalize_teacher_query_log, resolve_teacher_query_log

DEFAULT_SOURCE_REPO = Path(__file__).resolve().parent / "vendor"

SCRIPT_BY_DETECTOR = {
    "mmd": Path("MMD_detection/mmd_detector_gpu.py"),
    "prada": Path("PRADA/prada_detector.py"),
    "seat": Path("SEAT/seat_detector.py"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run query-traffic model-extraction detectors.")
    p.add_argument("--detector", choices=["mmd", "prada", "seat"], required=True)
    p.add_argument(
        "--teacher_query_log",
        default=None,
        help="JSONL of the queries actually sent to the teacher/oracle. Not the planned query pool.",
    )
    p.add_argument("--attack_manifest", default=None, help="Optional attack manifest used to infer teacher_query_log.")
    p.add_argument("--benign_query_log", required=True, help="Reference benign/normal query JSONL.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--source_repo", default=os.environ.get("MODEL_EXTRACTION_DETECTION_REPO", str(DEFAULT_SOURCE_REPO)))
    p.add_argument("--label", default="positive")
    p.add_argument("--attack_run_id", default=None)
    p.add_argument("--max_teacher_queries", type=int, default=None)
    p.add_argument("--max_benign", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--normal_train_ratio", type=float, default=0.8)
    p.add_argument("--null_samples", type=int, default=1000)
    p.add_argument("--threshold_percentile", type=float, default=95.0)
    p.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--query_prefix", default="")
    p.add_argument("--device", default=None)
    p.add_argument("--compute_device", default=None, help="PRADA/SEAT compute device; MMD uses --mmd_device.")
    p.add_argument("--mmd_device", default=None)
    p.add_argument("--mmd_dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--encode_batch_size", type=int, default=64)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_last_batch", action="store_true")
    p.add_argument("--benign_eval_batches", type=int, default=0)
    p.add_argument("--attacker_eval_batches", type=int, default=0)
    p.add_argument("--mixed_attacker_ratios", default="", help="Default empty: do not synthesize mixed windows.")
    p.add_argument("--mixed_batches_per_ratio", type=int, default=50)
    p.add_argument("--reference_repeats", type=int, default=20, help="MMD only.")
    p.add_argument("--multi_kernel", action="store_true", help="MMD only.")
    p.add_argument("--distance_metric", choices=["l2", "cosine"], default="l2", help="PRADA only.")
    p.add_argument("--tail", choices=["upper", "lower", "two-sided"], default="upper", help="PRADA only.")
    p.add_argument("--max_shapiro_samples", type=int, default=5000, help="PRADA only.")
    p.add_argument("--score_mode", choices=["ratio", "count"], default="ratio", help="SEAT only.")
    p.add_argument("--detection_tail", choices=["upper", "lower", "two-sided"], default="two-sided", help="SEAT only.")
    p.add_argument("--similarity_threshold", type=float, default=None, help="SEAT only.")
    p.add_argument("--similarity_threshold_percentile", type=float, default=99.0, help="SEAT only.")
    p.add_argument("--pair_sample_size", type=int, default=200000, help="SEAT only.")
    return p.parse_args()


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        proc = subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"query traffic detector failed with exit code {proc.returncode}; see {log_path}")


def _read_metadata(method_dir: Path) -> dict[str, Any]:
    path = method_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_batch_scores(method_dir: Path) -> list[dict[str, Any]]:
    path = method_dir / "batch_scores.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _score_from_metadata(metadata: dict[str, Any]) -> tuple[float, str]:
    by_split = metadata.get("metrics_by_split") or {}
    attacker = by_split.get("attacker") or {}
    detection_rate = attacker.get("detection_rate")
    if detection_rate is not None:
        return float(detection_rate), "attacker_detection_rate"
    metrics = metadata.get("metrics") or {}
    if metrics.get("tpr") is not None:
        return float(metrics["tpr"]), "tpr"
    return float("nan"), "attacker_detection_rate"


def _build_external_command(args: argparse.Namespace, method_dir: Path, attacker_path: Path, cache_dir: Path) -> list[str]:
    source_repo = Path(args.source_repo)
    script = source_repo / SCRIPT_BY_DETECTOR[args.detector]
    if not script.exists():
        raise FileNotFoundError(script)
    command = [
        sys.executable,
        str(script),
        "--normal-path",
        str(Path(args.benign_query_log).resolve()),
        "--attacker-path",
        str(attacker_path.resolve()),
        "--text-field",
        "query",
        "--embedding-model",
        args.embedding_model,
        "--query-prefix",
        args.query_prefix,
        "--encode-batch-size",
        str(args.encode_batch_size),
        "--batch-size",
        str(args.batch_size),
        "--null-samples",
        str(args.null_samples),
        "--threshold-percentile",
        str(args.threshold_percentile),
        "--normal-train-ratio",
        str(args.normal_train_ratio),
        "--mixed-attacker-ratios",
        args.mixed_attacker_ratios,
        "--mixed-batches-per-ratio",
        str(args.mixed_batches_per_ratio),
        "--benign-eval-batches",
        str(args.benign_eval_batches),
        "--attacker-eval-batches",
        str(args.attacker_eval_batches),
        "--seed",
        str(args.seed),
        "--cache-dir",
        str(cache_dir.resolve()),
        "--output-dir",
        str(method_dir.resolve()),
    ]
    if args.max_benign is not None:
        command.extend(["--max-normal", str(args.max_benign)])
    if args.max_teacher_queries is not None:
        command.extend(["--max-attacker", str(args.max_teacher_queries)])
    if args.keep_last_batch:
        command.append("--keep-last-batch")
    if args.device is not None:
        command.extend(["--device", args.device])

    if args.detector == "mmd":
        command.extend(
            [
                "--reference-repeats",
                str(args.reference_repeats),
                "--mmd-dtype",
                args.mmd_dtype,
            ]
        )
        if args.mmd_device is not None:
            command.extend(["--mmd-device", args.mmd_device])
        if args.multi_kernel:
            command.append("--multi-kernel")
    elif args.detector == "prada":
        command.extend(
            [
                "--distance-metric",
                args.distance_metric,
                "--tail",
                args.tail,
                "--max-shapiro-samples",
                str(args.max_shapiro_samples),
            ]
        )
        if args.compute_device is not None:
            command.extend(["--compute-device", args.compute_device])
    elif args.detector == "seat":
        command.extend(
            [
                "--score-mode",
                args.score_mode,
                "--detection-tail",
                args.detection_tail,
                "--similarity-threshold-percentile",
                str(args.similarity_threshold_percentile),
                "--pair-sample-size",
                str(args.pair_sample_size),
            ]
        )
        if args.similarity_threshold is not None:
            command.extend(["--similarity-threshold", str(args.similarity_threshold)])
        if args.compute_device is not None:
            command.extend(["--compute-device", args.compute_device])

    return command


def main() -> None:
    args = parse_args()
    t0 = time.time()
    out_dir = ensure_dir(args.output_dir)
    artifacts_dir = ensure_dir(out_dir / "artifacts")
    method_dir = ensure_dir(out_dir / args.detector)
    cache_dir = Path(args.cache_dir) if args.cache_dir else artifacts_dir / "cache"

    source_teacher_log = resolve_teacher_query_log(
        teacher_query_log=args.teacher_query_log,
        attack_manifest=args.attack_manifest,
    )
    normalized = normalize_teacher_query_log(
        source_teacher_log,
        artifacts_dir / "teacher_received_queries.jsonl",
        limit=args.max_teacher_queries,
    )
    attacker_path = Path(normalized["path"])

    command = _build_external_command(args, method_dir, attacker_path, cache_dir)
    _run_command(command, cwd=Path(args.source_repo), log_path=artifacts_dir / f"{args.detector}_run.log")

    metadata = _read_metadata(method_dir)
    batch_scores = _read_batch_scores(method_dir)
    score, score_name = _score_from_metadata(metadata)
    report_path = out_dir / "detector_report.json"
    report = {
        "detector": args.detector,
        "kind": "query_traffic_detector",
        "score": score,
        "score_name": score_name,
        "score_direction": "higher",
        "source_teacher_query_log": str(Path(source_teacher_log).resolve()),
        "teacher_received_queries": normalized,
        "benign_query_log": str(Path(args.benign_query_log).resolve()),
        "external_metadata": metadata,
        "external_output_dir": str(method_dir.resolve()),
        "num_batch_scores": len(batch_scores),
        "command": command,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    row = finalize_detector_row(
        {
            "id": f"{args.detector}::{Path(source_teacher_log).stem}",
            "model_id": Path(source_teacher_log).stem,
            "label": args.label,
            "score": score,
            "score_name": score_name,
            "score_direction": "higher",
            "detector": args.detector,
            "teacher_query_log": str(Path(source_teacher_log).resolve()),
            "benign_query_log": str(Path(args.benign_query_log).resolve()),
            "attacker_queries": normalized["num_queries"],
        },
        detector=args.detector,
        defense_run_id=args.detector,
        attack_run_id=args.attack_run_id,
    )
    rows_path = out_dir / "detector_rows.jsonl"
    write_jsonl([row], rows_path)

    manifest = DetectorManifest(
        detector=args.detector,
        defense_run_id=args.detector,
        attack_run_id=args.attack_run_id,
        student_checkpoint=None,
        watermark_artifacts={},
        output_report=str(report_path.resolve()),
        cost=build_cost(
            wall_seconds=time.time() - t0,
            num_queries=normalized["num_queries"],
            models=[args.embedding_model],
            artifacts={
                "detector_report": str(report_path.resolve()),
                "detector_rows": str(rows_path.resolve()),
                "teacher_received_queries": normalized["path"],
                "external_metadata": str((method_dir / "metadata.json").resolve()),
                "batch_scores": str((method_dir / "batch_scores.csv").resolve()),
            },
            data_files=[str(Path(args.benign_query_log).resolve()), str(Path(source_teacher_log).resolve())],
            caches=[str(cache_dir.resolve())],
            stage="detector",
            method=args.detector,
            extra={"detector_kind": "query_traffic", "source_repo": str(Path(args.source_repo).resolve())},
        ),
        probe_queries=normalized["path"],
        num_probe_queries=normalized["num_queries"],
        config={
            "detector_kind": "query_traffic",
            "batch_size": args.batch_size,
            "normal_train_ratio": args.normal_train_ratio,
            "threshold_percentile": args.threshold_percentile,
            "null_samples": args.null_samples,
            "embedding_model": args.embedding_model,
            "uses_teacher_received_queries": True,
        },
        students=[row],
    )
    manifest.save(out_dir / "detector_manifest.json")

    print(f"[{args.detector}] score={score} ({score_name})")
    print(f"[{args.detector}] teacher_received_queries -> {normalized['path']}")
    print(f"[{args.detector}] report -> {report_path}")


if __name__ == "__main__":
    main()
