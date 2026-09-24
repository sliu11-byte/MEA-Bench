"""Evaluate local Llama SeqKD B=1000 counter checkpoints without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from evaluation.attack_eval.evaluate_attack_four_metrics import prepare_tasks
from evaluation.attack_eval.evaluate_attack_outputs import iter_jsonl, load_prompts, load_teacher, write_json
from evaluation.core.choice_scoring import SCORING_METHOD
from evaluation.defense_eval.evaluate import check_protocol, evaluate_model
from evaluation.metrics.m2_fidelity import bertscore_f1

TEACHER = "meta-llama/Llama-3.3-70B-Instruct"
BASE = "meta-llama/Llama-3.1-8B-Instruct"
DEFENSES = ("adfp", "ginsew", "radioactivity")
COUNTERS = ("dipper", "translation")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_path(value, storage):
    if not isinstance(value, str) or not value:
        raise ValueError("Missing local artifact path in manifest")
    raw = Path(value)
    if raw.exists():
        return raw.resolve()
    if "outputs" in raw.parts:
        candidate = storage.joinpath(*raw.parts[raw.parts.index("outputs"):])
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Local artifact missing: {value}")


def checkpoint_from_manifest(path, storage):
    manifest = read_json(path)
    result = manifest.get("result", {})
    config = manifest.get("run_config", {})
    if result.get("attack") != "seqkd" or result.get("budget") != 1000 or result.get("status") != "completed":
        raise ValueError(f"Not a completed SeqKD B=1000 run: {path}")
    if config.get("teacher_model") != TEACHER or config.get("student_model") != BASE:
        raise ValueError(f"Wrong teacher/student: {path}")
    checkpoint = local_path(result.get("checkpoint_dir") or manifest.get("checkpoint_dir"), storage)
    if not (checkpoint / "config.json").exists():
        raise ValueError(f"Expected full model checkpoint: {checkpoint}")
    index = checkpoint / "model.safetensors.index.json"
    weights = set(read_json(index)["weight_map"].values()) if index.exists() else {"model.safetensors"}
    for name in weights:
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise ValueError(f"Missing checkpoint weights: {checkpoint / name}")
    if not (checkpoint / "tokenizer_config.json").exists():
        raise ValueError(f"Missing checkpoint tokenizer: {checkpoint}")
    return checkpoint


def discover_local(storage):
    models, reports = {}, {}
    for counter in COUNTERS:
        for defense in DEFENSES:
            directory = storage / "outputs/countermeasures/seqkd_b1000" / counter / defense
            report = read_json(directory / "comparison_report.json")
            if (report.get("attack"), report.get("budget"), report.get("defense")) != ("seqkd", 1000, defense):
                raise ValueError(f"Wrong comparison report: {directory}")
            reports[counter, defense] = report
            for group in ("clean", "defense_only"):
                baseline = read_json(local_path(report["baseline_manifests"][group], storage))
                if baseline.get("status") != "ok":
                    raise ValueError(f"Incomplete baseline: {directory}/{group}")
                manifest = local_path(baseline["attack_manifest_path"], storage)
                checkpoint = checkpoint_from_manifest(manifest, storage)
                name = "clean" if group == "clean" else f"{defense}_defense_only"
                if name in models and models[name] != checkpoint:
                    raise ValueError(f"Baselines differ between counter methods: {name}")
                models[name] = checkpoint
            detection = report["reports"]["counter"]
            student = detection.get("students", [detection])[0]
            suspect = local_path(detection.get("student_checkpoint") or student["student_checkpoint"], storage)
            if suspect.name == "attack_manifest.json":
                manifest = suspect
            else:
                candidates = []
                for path in directory.glob("attack/seqkd/*/attack_manifest.json"):
                    result = read_json(path).get("result", {})
                    if result.get("status") != "completed":
                        continue
                    value = result.get("checkpoint_dir") or read_json(path).get("checkpoint_dir")
                    if value and local_path(value, storage) == suspect:
                        candidates.append(path)
                if len(candidates) != 1:
                    raise ValueError(f"Cannot uniquely identify counter run: {directory}")
                manifest = candidates[0]
            models[f"{defense}_{counter}"] = checkpoint_from_manifest(manifest, storage)
    for defense in DEFENSES:
        dipper = reports["dipper", defense]
        translation = reports["translation", defense]
        for group in ("clean", "defense_only"):
            if dipper["reports"].get(group) != translation["reports"].get(group):
                raise ValueError(
                    f"{defense} {group} detector baseline differs between DIPPER and translation; "
                    "run runs/evaluation/rebuild_counter_seqkd_b1000_baselines.sh first"
                )
    if len(models) != 10 or len(set(models.values())) != 10:
        raise ValueError("Expected ten distinct local checkpoints")
    return models, reports


def detector_stats(report):
    student = report.get("students", [report])[0]
    return {key: student.get(key, report.get(key)) for key in
            ("score", "score_name", "score_direction", "gtp", "green_rate", "psnr", "p_value")}


def write_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reuse_valid_generations(source, destination):
    """Copy only protocol-stable generations; old multiple-choice rows are intentionally excluded."""
    copied = []
    for relative in (Path("m1_predictions/gsm8k.jsonl"), Path("heldout_outputs.jsonl")):
        source_path = source / relative
        destination_path = destination / relative
        if not source_path.is_file() or source_path.stat().st_size == 0 or destination_path.exists():
            continue
        # Parse before copying so an interrupted final JSONL line fails before model loading.
        list(iter_jsonl(source_path))
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied.append(str(relative))
    return copied


def validate_teacher_manifest(manifest):
    expected = {"teacher_model": TEACHER, "mode": "chat", "temperature": 0.0,
                "top_p": 1.0, "max_tokens": 1536}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Teacher reference protocol mismatch: {key} expected {value}")
    if manifest.get("request_model", TEACHER) != TEACHER:
        raise ValueError(f"Teacher reference request model mismatch: {manifest.get('request_model')}")
    seed = manifest.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"Teacher reference seed must be an integer, got {seed!r}")
    if manifest.get("prompt_count") != manifest.get("completed_count"):
        raise ValueError("Teacher reference manifest is incomplete")
    return {**expected, "request_model": manifest.get("request_model", TEACHER), "seed": seed,
            "schema_version": manifest.get("schema_version"), "completed_count": manifest.get("completed_count")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--teacher-manifest")
    parser.add_argument("--reuse-generation-root",
                        help="Old Counter output root; reuse only GSM8K and held-out JSONL files.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    storage = Path(args.storage_root).resolve()
    models, reports = discover_local(storage)
    teacher_path = Path(args.teacher_jsonl)
    manifest = read_json(args.teacher_manifest or teacher_path.with_suffix(".manifest.json"))
    teacher_generation = validate_teacher_manifest(manifest)
    prompts = load_prompts(Path(args.prompts_jsonl))
    teacher = load_teacher(teacher_path)
    if sum(1 for _ in iter_jsonl(teacher_path)) != len(teacher):
        raise ValueError("Duplicate teacher reference IDs")
    if not prompts or len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("Empty/duplicate held-out prompts")
    for prompt in prompts:
        row = teacher.get(prompt["id"])
        if not row or row["prompt"] != prompt["prompt"] or not row["teacher_text"].strip():
            raise ValueError(f"Missing/mismatched/empty teacher answer: {prompt['id']}")
    for name, checkpoint in models.items():
        print(f"[Counter input] {name}: {checkpoint}", flush=True)
    print(f"[Counter reference] validated {len(prompts)} prompts; teacher will not be loaded", flush=True)
    if args.preflight:
        return
    if args.limit:
        prompts = prompts[:args.limit]
    output = Path(args.output_root).resolve()
    reuse_root = Path(args.reuse_generation_root).resolve() if args.reuse_generation_root else None
    if reuse_root and reuse_root == output:
        raise ValueError("Reuse source and new output root must differ")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "input_detection_reports.json", {f"{counter}/{defense}": report
                                                        for (counter, defense), report in reports.items()})
    tasks, metadata = prepare_tasks(Path("evaluation/configs/m1_rollout.yaml"), args.limit, output)
    protocol = {"teacher": TEACHER, "base": BASE, "checkpoints": {k: str(v) for k, v in models.items()},
                "tasks": metadata, "prompts": prompts, "teacher_reference": [teacher[p["id"]] for p in prompts],
                "teacher_generation": teacher_generation, "student_rendering": "chat", "seed": 42,
                "m1_multiple_choice_scoring": SCORING_METHOD,
                "math_max_tokens": 512, "heldout_max_tokens": 1536,
                "bert_model": "roberta-large", "rescale_with_baseline": True, "smoke": bool(args.limit),
                "reuse_generation_root": str(reuse_root) if reuse_root else None,
                "reused_artifacts": ["m1_predictions/gsm8k.jsonl", "heldout_outputs.jsonl"]
                if reuse_root else []}
    check_protocol(output / "protocol.json", protocol)
    (output / "complete.json").unlink(missing_ok=True)
    scores, summary = {}, []
    for name, checkpoint in models.items():
        print(f"[Counter evaluation] {name}", flush=True)
        directory = output / "models" / name
        if reuse_root:
            copied = reuse_valid_generations(reuse_root / "models" / name, directory)
            if copied:
                print(f"[Counter reuse] {name}: {', '.join(copied)}", flush=True)
        acc, rows = evaluate_model(directory, str(checkpoint), True, tasks, prompts, True)
        score = bertscore_f1([rows[p["id"]]["text"] for p in prompts],
                             [teacher[p["id"]]["teacher_text"] for p in prompts], "en", "roberta-large", 16, "cuda:0")
        write_json(directory / "m2_bertscore.json", {"bertscore": score, "n": len(prompts), "reference": str(teacher_path)})
        scores[name] = {"acc": acc, "bertscore": score}
        summary.append({"model": name, **scores[name], "smoke": bool(args.limit), "checkpoint": str(checkpoint)})
        write_json(output / "models_summary.json", summary)
        write_table(output / "models_summary.csv", summary)
    comparisons = []
    for (counter, defense), report in reports.items():
        for group, name in (("clean", "clean"), ("defense_only", f"{defense}_defense_only"),
                            ("counter", f"{defense}_{counter}")):
            comparisons.append({"attack": "seqkd", "budget": 1000, "defense": defense, "countermeasure": counter,
                                "condition": group, "model": name, **scores[name],
                                **detector_stats(report["reports"][group]), "smoke": bool(args.limit)})
    write_json(output / "summary.json", comparisons)
    write_table(output / "summary.csv", comparisons)
    write_json(output / "complete.json", {"complete": True, "models": 10, "comparison_rows": 18, "smoke": bool(args.limit)})
    print(f"[Counter evaluation] complete -> {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
