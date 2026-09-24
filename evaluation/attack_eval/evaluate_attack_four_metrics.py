"""Local attack evaluation: six-task ACC, BERTScore, Rep-4, allocated GPU-hours."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import shutil
import subprocess
from collections import Counter
from evaluation.core.batch_generation import batch_size, generate_batch, iter_generated
from evaluation.core.choice_scoring import SCORING_METHOD, choice_spec, score_examples
from pathlib import Path

from evaluation.attack_eval.evaluate_attack_outputs import (
    discover_attacks, iter_jsonl, load_prompts, load_teacher, resolve_attack_root, write_json,
)
from evaluation.core.config import DatasetSpec
from evaluation.metrics.m2_fidelity import bertscore_f1
from evaluation.metrics.m3_quality import repetition_report

SPLITS = {
    "arc_challenge": "test", "hellaswag": "validation", "mmlu": "test",
    "truthfulqa": "validation", "winogrande": "validation", "gsm8k": "test",
}


def parse_job_ids(values):
    return [job_id for value in values for job_id in re.split(r"[\s,]+", value.strip()) if job_id]


def allocation_cost(job_ids, run_id):
    jobs = []
    for job_id in job_ids:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--noheader", "--parsable2",
             "--format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES"],
            check=True, capture_output=True, text=True,
        )
        rows = [line.split("|", 4) for line in result.stdout.splitlines() if line.strip()]
        matches = [row for row in rows if len(row) == 5 and row[0] == job_id]
        if len(matches) != 1:
            raise ValueError(f"Could not identify top-level sacct row for job {job_id}")
        _, name, state, elapsed, tres_text = matches[0]
        if not state.startswith("COMPLETED"):
            raise ValueError(f"Allocation job {job_id} is not completed: {state}")
        tres = dict(item.split("=", 1) for item in tres_text.split(",") if "=" in item)
        if "gres/gpu" not in tres:
            raise ValueError(f"Allocation job {job_id} has no generic gres/gpu allocation: {tres_text}")
        allocated_gpus = int(tres["gres/gpu"])
        elapsed_seconds = int(elapsed)
        gpu_types = sorted(key.removeprefix("gres/gpu:") for key in tres if key.startswith("gres/gpu:"))
        jobs.append({"job_id": job_id, "job_name": name, "state": state,
                     "elapsed_seconds": elapsed_seconds, "allocated_gpus": allocated_gpus,
                     "gpu_types": gpu_types,
                     "allocated_gpu_hours": elapsed_seconds * allocated_gpus / 3600})
    return {"run_id": run_id, "job_ids": job_ids, "jobs": jobs,
            "allocated_gpu_hours": sum(job["allocated_gpu_hours"] for job in jobs),
            "teacher_scope": "sum of explicitly supplied staged attack allocations"}


def reuse_gsm8k(source_root, destination, attack):
    source = source_root / attack.attack / attack.run_id / "m1_predictions" / "gsm8k.jsonl"
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Reusable GSM8K predictions are missing: {source}")
    if source.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return source


def make_example_ids_unique(examples):
    """Disambiguate only duplicated source IDs while preserving all other resume keys."""
    counts = Counter(example.example_id for example in examples)
    for example in examples:
        if counts[example.example_id] > 1:
            example.example_id = f"{example.example_id}::row-{example.example_index}"
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset example IDs remain duplicated after row-index disambiguation")
    return examples


def generate(student, prompt, max_tokens):
    return generate_batch(student.model, student.tokenizer, [prompt], max_tokens, True)[0]


def resume_rows(path):
    rows = list(iter_jsonl(path)) if path.exists() else []
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate ids in {path}")
    return {row["id"]: row for row in rows}


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_protocol_before_generation(path, protocol, directory):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        if any(directory.rglob("*.jsonl")):
            raise ValueError(f"Protocol changed; use a new output directory: {directory}")
        print(f"Replacing pre-generation protocol after input path resolution: {path}", flush=True)
    write_json(path, protocol)


def prepare_tasks(config_path, limit, output):
    import yaml
    from evaluation.tasks.adapters.tasks import create_adapter
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = {}
    metadata = {}
    for key, split in SPLITS.items():
        spec = DatasetSpec(key=key, **config["datasets"][key])
        adapter = create_adapter(spec)
        data = adapter.get_split(split)
        count = min(limit, len(data)) if limit else len(data)
        tasks[key] = make_example_ids_unique([adapter.adapt(dict(data[i]), split, i) for i in range(count)])
        metadata[key] = {"source": spec.identifier, "config": spec.config_name, "split": split,
                         "fingerprint": getattr(data, "_fingerprint", None), "n": count,
                         "prompt_template_hash": adapter.prompt_template_hash, "fewshot": 0}
        if key != "gsm8k":
            scoring = choice_spec(tasks[key][0])
            metadata[key].update({"metric": scoring["metric"],
                                  "choice_scoring": scoring["method"],
                                  "length_normalize": scoring["length_normalize"]})
        else:
            metadata[key].update({"metric": "exact_match", "generation_max_tokens": 512})
    write_json(output / "m1_dataset_protocol.json", metadata)
    return tasks, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-root", required=True,
                        help="Local directory containing completed attack runs.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--eval-prompts-jsonl")
    parser.add_argument("--teacher-jsonl")
    parser.add_argument("--cost-records", default=str(Path(__file__).with_name("b100_allocation_costs.json")))
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--skip-m6", action="store_true",
                        help="Run M1/M2/M3 and record M6 as unavailable when allocation records are not yet known.")
    parser.add_argument("--expected-teacher-model")
    parser.add_argument("--expected-student-model")
    parser.add_argument("--m1-config", default="evaluation/configs/m1_rollout.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Smoke only: first N samples per task and held-out.")
    parser.add_argument("--attack", help="Evaluate one method only.")
    parser.add_argument("--exclude-attack", action="append", default=[], help="Exclude one method; repeat as needed.")
    parser.add_argument("--m1-only", action="store_true", help="Evaluate M1 only; skip held-out, M2, M3, and M6.")
    parser.add_argument("--reuse-gsm8k-root", help="Old evaluation root containing matching GSM8K predictions.")
    parser.add_argument("--allocation-job-ids", nargs="+", default=[],
                        help="Completed staged attack job IDs to sum for M6; requires exactly one selected attack.")
    parser.add_argument("--bert-model", default="roberta-large")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if not args.m1_only and (not args.eval_prompts_jsonl or not args.teacher_jsonl):
        parser.error("Full evaluation requires --eval-prompts-jsonl and --teacher-jsonl")
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = resolve_attack_root(args, output)
    attacks = discover_attacks(root)
    if args.attack:
        attacks = [attack for attack in attacks if attack.attack == args.attack]
    if args.exclude_attack:
        attacks = [attack for attack in attacks if attack.attack not in set(args.exclude_attack)]
    if not attacks or len({a.attack for a in attacks}) != len(attacks):
        raise ValueError("Expected one run per method in the attack snapshot")
    job_ids = parse_job_ids(args.allocation_job_ids)
    if job_ids and (args.m1_only or len(attacks) != 1):
        raise ValueError("--allocation-job-ids requires a full evaluation of exactly one attack")
    if args.m1_only and not args.reuse_gsm8k_root:
        raise ValueError("--m1-only requires --reuse-gsm8k-root")
    costs = {} if args.m1_only or args.skip_m6 or job_ids else json.loads(
        Path(args.cost_records).read_text(encoding="utf-8")
    )
    for attack in attacks:
        if attack.budget != args.budget or not attack.checkpoint or attack.status != "completed":
            raise ValueError(f"Invalid budget{args.budget} completed checkpoint: {attack}")
        if args.expected_teacher_model and attack.teacher_model != args.expected_teacher_model:
            raise ValueError(f"Unexpected teacher for {attack.attack}: {attack.teacher_model}")
        if args.expected_student_model and attack.student_model != args.expected_student_model:
            raise ValueError(f"Unexpected student for {attack.attack}: {attack.student_model}")
        if not args.m1_only and not args.skip_m6 and not job_ids and (
                attack.attack not in costs or costs[attack.attack]["run_id"] != attack.run_id):
            raise ValueError(f"Historical cost run mismatch: {attack.attack}")
    if args.m1_only:
        reuse_root = Path(args.reuse_gsm8k_root).resolve()
        missing = [str(reuse_root / attack.attack / attack.run_id / "m1_predictions" / "gsm8k.jsonl")
                   for attack in attacks
                   if not (reuse_root / attack.attack / attack.run_id / "m1_predictions" / "gsm8k.jsonl").is_file()]
        if missing:
            raise FileNotFoundError("Missing reusable GSM8K predictions:\n" + "\n".join(missing))
    prompts, teacher = [], {}
    if not args.m1_only:
        prompts = load_prompts(Path(args.eval_prompts_jsonl))
        teacher = load_teacher(Path(args.teacher_jsonl))
        if len({p["id"] for p in prompts}) != len(prompts):
            raise ValueError("Duplicate held-out prompt ids")
        for item in prompts:
            if item["id"] not in teacher or teacher[item["id"]]["prompt"] != item["prompt"]:
                raise ValueError(f"Missing or mismatched teacher prompt: {item['id']}")
        if args.limit:
            prompts = prompts[:args.limit]
    task_protocol_root = output / "_task_protocols" / args.attack if args.attack else output
    tasks, metadata = prepare_tasks(Path(args.m1_config), args.limit, task_protocol_root)
    from attacks.core.student_model import load_student_from_checkpoint
    import torch
    from transformers import set_seed
    summary = []
    summary_stem = f"summary_{args.attack}" if args.attack else "summary"
    for attack in attacks:
        print(f"Starting {attack.attack} / {attack.run_id}", flush=True)
        directory = output / attack.attack / attack.run_id
        directory.mkdir(parents=True, exist_ok=True)
        protocol = {"run_id": attack.run_id, "checkpoint": attack.checkpoint, "seed": 42,
                    "budget": args.budget, "m1": metadata, "m1_only": args.m1_only,
                    "m1_multiple_choice_scoring": SCORING_METHOD,
                    "max_tokens_math": 512}
        if args.m1_only:
            protocol["gsm8k_reuse_root"] = str(Path(args.reuse_gsm8k_root).resolve())
        else:
            protocol.update({"heldout": prompts, "teacher": [teacher[p["id"]] for p in prompts],
                             "max_tokens_heldout": 1536, "bert_model": args.bert_model,
                             "rep_tokenizer": "unicode_cjk"})
        protocol_path = directory / "protocol.json"
        write_protocol_before_generation(protocol_path, protocol, directory)
        set_seed(42)
        student = load_student_from_checkpoint(attack.checkpoint, dtype="bfloat16", device_map="auto")
        per_task = {}
        for key, examples in tasks.items():
            path = directory / ("m1_predictions" if key == "gsm8k" else "m1_choice_scores_v2") / f"{key}.jsonl"
            if args.m1_only and key == "gsm8k":
                source = reuse_gsm8k(Path(args.reuse_gsm8k_root).resolve(), path, attack)
                print(f"{attack.attack} gsm8k: reused {source}", flush=True)
            rows = resume_rows(path)
            size = batch_size("math" if key == "gsm8k" else "mc")
            pending = [example for example in examples if example.example_id not in rows]
            print(f"{attack.attack} {key}: pending={len(pending)} batch={size}", flush=True)
            if key == "gsm8k":
                iterator = iter_generated(student.model, student.tokenizer, pending,
                                          lambda e: e.rendered_prompt, 512, True, size)
                for example, text in iterator:
                    parsed = example.parser.parse(text)
                    row = {"id": example.example_id, "prompt": example.rendered_prompt, "response": text,
                           "gold": example.normalized_gold_answer, "prediction": parsed.normalized_prediction,
                           "generation_batch_size": size, "parse_status": parsed.status,
                           "correct": parsed.status == "success" and parsed.normalized_prediction == example.normalized_gold_answer}
                    append(path, row)
                    rows[example.example_id] = row
                    if len(rows) % 100 == 0:
                        print(f"{attack.attack} {key}: {len(rows)}/{len(examples)}", flush=True)
            else:
                for example, result in score_examples(student.model, student.tokenizer, pending, True, size):
                    row = {"id": example.example_id, **result, "scoring_batch_size": size}
                    append(path, row)
                    rows[example.example_id] = row
                    if len(rows) % 100 == 0:
                        print(f"{attack.attack} {key}: {len(rows)}/{len(examples)}", flush=True)
            selected = [rows[e.example_id] for e in examples]
            per_task[key] = {"acc": sum(r["correct"] for r in selected) / len(selected), "n": len(selected),
                             "metric": "exact_match" if key == "gsm8k" else choice_spec(examples[0])["metric"],
                             "parse_failures": sum(r.get("parse_status") != "success" for r in selected)
                             if key == "gsm8k" else 0}
        acc = sum(task["acc"] for task in per_task.values()) / len(per_task)
        write_json(directory / "m1_acc.json", {"acc": acc, "per_task": per_task,
                                                "aggregation": "six-task macro average",
                                                "multiple_choice_protocol": SCORING_METHOD})
        if args.m1_only:
            del student
            gc.collect()
            torch.cuda.empty_cache()
            summary.append({"attack": attack.attack, "run_id": attack.run_id, "acc": acc,
                            "smoke": bool(args.limit), "report_dir": str(directory)})
            write_json(output / f"{summary_stem}.json", summary)
            with (output / f"{summary_stem}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            continue
        path = directory / "heldout_student.jsonl"
        rows = resume_rows(path)
        size = batch_size("heldout")
        pending = [item for item in prompts if item["id"] not in rows]
        print(f"{attack.attack} held-out: pending={len(pending)} batch={size}", flush=True)
        for item, text in iter_generated(student.model, student.tokenizer, pending,
                                         lambda p: p["prompt"], 1536, True, size):
            if item["id"] not in rows:
                row = {**item, "student_text": text, "generation_batch_size": size}
                append(path, row)
                rows[item["id"]] = row
            if len(rows) % 100 == 0:
                print(f"{attack.attack} held-out: {len(rows)}/{len(prompts)}", flush=True)
        del student
        gc.collect()
        torch.cuda.empty_cache()
        texts = [rows[p["id"]]["student_text"] for p in prompts]
        references = [teacher[p["id"]]["teacher_text"] for p in prompts]
        score = bertscore_f1(texts, references, "en", args.bert_model, 16, "cuda:0")
        write_json(directory / "m2_bertscore.json", {"bertscore_f1_rescaled": score, "n": len(texts), "encoder": args.bert_model})
        rep = repetition_report(texts, ns=[4], tokenizer="unicode_cjk", short_policy="exclude")
        write_json(directory / "m3_rep4.json", rep)
        if args.skip_m6:
            cost = {"run_id": attack.run_id, "status": "not_computed",
                    "reason": "Slurm job allocation records were not supplied"}
            gpu_hours = None
        else:
            cost = allocation_cost(job_ids, attack.run_id) if job_ids else costs[attack.attack]
            gpu_hours = cost.get("allocated_gpu_hours")
            if gpu_hours is None:
                gpu_hours = cost["elapsed_seconds"] * cost["allocated_gpus"] / 3600
        write_json(directory / "m6_cost.json", {**cost, "allocated_gpu_hours": gpu_hours,
                   "scope": "sum of supplied whole-job GPU allocations, not measured GPU busy time"})
        summary.append({"attack": attack.attack, "run_id": attack.run_id, "acc": acc,
                        "bertscore": score, "rep4": rep["metrics"]["rep_4"]["macro_average"],
                        "allocated_gpu_hours": gpu_hours, "smoke": bool(args.limit),
                        "report_dir": str(directory)})
        write_json(output / f"{summary_stem}.json", summary)
        with (output / f"{summary_stem}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Finished. Summary: {output / f'{summary_stem}.csv'}", flush=True)


if __name__ == "__main__":
    main()
