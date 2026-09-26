from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
from evaluation.core.batch_generation import batch_size, generate_batch, iter_generated
from evaluation.core.choice_scoring import SCORING_METHOD, choice_spec, score_examples
from pathlib import Path

from evaluation.attack_eval.evaluate_attack_four_metrics import append, prepare_tasks, resume_rows
from evaluation.attack_eval.evaluate_attack_outputs import load_prompts, write_json
from evaluation.metrics.m2_fidelity import bertscore_f1

TEACHER = "Qwen/Qwen2.5-72B-Instruct"
BASE = "Qwen/Qwen2.5-7B"
DEFENSES = ("clean", "ads", "doge", "trace_rewriting", "adfp", "ginsew", "radioactivity")
CLEAN = {
    "seqkd": "clean/20260820T020802Z_seqkd_b1000/training/seqkd/budget_1000/seqkd/checkpoint-final",
    "soda": "clean/checkpoint-final",
    "qedks": "clean/20260912T215757Z_qedks_b1000/checkpoints/lora_sft",
}


def check_protocol(path, protocol):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError(f"Protocol changed; use a new output root: {path}")
    write_json(path, protocol)


def generate(model, tokenizer, prompt, max_tokens, chat):
    return generate_batch(model, tokenizer, [prompt], max_tokens, chat)[0]


def evaluate_model(directory, source, chat, tasks, prompts, heldout, checkpoint=None, reuse_gsm8k_directory=None):
    """Load only when generation is incomplete; persist every completed sample."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    import torch
    directory.mkdir(parents=True, exist_ok=True)
    task_paths = {key: directory / ("m1_predictions" if key == "gsm8k" else "m1_choice_scores_v2") /
                  f"{key}.jsonl" for key in tasks}
    if "gsm8k" in tasks and reuse_gsm8k_directory and not task_paths["gsm8k"].exists():
        source_path = Path(reuse_gsm8k_directory) / "m1_predictions" / "gsm8k.jsonl"
        source_rows = resume_rows(source_path)
        expected = {example.example_id for example in tasks["gsm8k"]}
        if not expected.issubset(source_rows):
            missing = sorted(expected - set(source_rows))[:5]
            raise ValueError(f"Reusable GSM8K output is incomplete: {source_path}; missing={missing}")
        task_paths["gsm8k"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, task_paths["gsm8k"])
        print(f"Reused GSM8K predictions: {source_path} -> {task_paths['gsm8k']}", flush=True)
    task_rows = {key: resume_rows(task_paths[key]) for key in tasks}
    heldout_path = directory / "heldout_outputs.jsonl"
    rows = resume_rows(heldout_path) if heldout else {}
    incomplete = any(e.example_id not in task_rows[k] for k, examples in tasks.items() for e in examples)
    incomplete |= heldout and any(p["id"] not in rows for p in prompts)
    if incomplete:
        set_seed(42)
        tokenizer_source = str(checkpoint) if checkpoint and (checkpoint / "tokenizer_config.json").exists() else source
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        model = AutoModelForCausalLM.from_pretrained(source, dtype="bfloat16", device_map="auto")
        if checkpoint:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(checkpoint))
        model.eval()
        for key, examples in tasks.items():
            path = task_paths[key]
            size = batch_size("math" if key == "gsm8k" else "mc", teacher=source == TEACHER)
            pending = [example for example in examples if example.example_id not in task_rows[key]]
            print(f"{directory.name} {key}: pending={len(pending)} batch={size}", flush=True)
            if key == "gsm8k":
                iterator = iter_generated(model, tokenizer, pending, lambda e: e.rendered_prompt, 512, chat, size)
                for example, text in iterator:
                    parsed = example.parser.parse(text)
                    row = {"id": example.example_id, "prompt": example.rendered_prompt, "response": text,
                           "gold": example.normalized_gold_answer, "prediction": parsed.normalized_prediction,
                           "generation_batch_size": size, "parse_status": parsed.status, "correct": parsed.status == "success" and
                           parsed.normalized_prediction == example.normalized_gold_answer}
                    append(path, row)
                    task_rows[key][example.example_id] = row
                    if len(task_rows[key]) % 100 == 0:
                        print(f"{directory.name} {key}: {len(task_rows[key])}/{len(examples)}", flush=True)
            else:
                for example, result in score_examples(model, tokenizer, pending, chat, size):
                    row = {"id": example.example_id, **result, "scoring_batch_size": size}
                    append(path, row)
                    task_rows[key][example.example_id] = row
                    if len(task_rows[key]) % 100 == 0:
                        print(f"{directory.name} {key}: {len(task_rows[key])}/{len(examples)}", flush=True)
        if heldout:
            size = batch_size("heldout", teacher=source == TEACHER)
            pending = [item for item in prompts if item["id"] not in rows]
            print(f"{directory.name} heldout: pending={len(pending)} batch={size}", flush=True)
            for item, text in iter_generated(model, tokenizer, pending, lambda p: p["prompt"], 1536, chat, size):
                if item["id"] not in rows:
                    row = {**item, "text": text, "model_id": source, "generation_batch_size": size}
                    append(heldout_path, row)
                    rows[item["id"]] = row
                if len(rows) % 100 == 0:
                    print(f"{directory.name} heldout: {len(rows)}/{len(prompts)}", flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    per_task = {}
    for key, examples in tasks.items():
        selected = [task_rows[key][e.example_id] for e in examples]
        metric = "exact_match" if key == "gsm8k" else choice_spec(examples[0])["metric"]
        per_task[key] = {"acc": sum(r["correct"] for r in selected) / len(selected), "n": len(selected),
                         "metric": metric,
                         "parse_failures": sum(r.get("parse_status") != "success" for r in selected)
                         if key == "gsm8k" else 0}
    report = {"acc": sum(r["acc"] for r in per_task.values()) / len(per_task), "per_task": per_task,
              "aggregation": "six-task macro average", "multiple_choice_protocol": SCORING_METHOD}
    write_json(directory / "m1_acc.json", report)
    return report["acc"], rows


def _checkpoint_complete(path):
    return (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    )


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def local_corrected_soda_checkpoint(defense):
    storage = Path(os.environ.get(
        "STORAGE_ROOT", f"/path/to/storage/{os.environ.get('USER', '')}/A-Benchmark-for-Model-distillation-survey"
    )).expanduser()
    root = Path(os.environ.get(
        "SODA_FULL_OUTPUT_ROOT", storage / "outputs" / "defenses" / "soda_b1000_seqkd_warmup"
    )).expanduser().resolve()
    if defense == "clean":
        manifest = root / "clean" / "attack_manifest.json"
        if not manifest.is_file():
            return None
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        checkpoint_value = (payload.get("result") or {}).get("checkpoint_dir")
    else:
        manifest = root / defense / "defense_run_manifest.json"
        if not manifest.is_file():
            return None
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        checkpoint_value = (payload.get("generator_result") or {}).get("student_checkpoint_path")
    if not checkpoint_value:
        raise ValueError(f"Corrected SODA manifest has no checkpoint: {manifest}")
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    if not _checkpoint_complete(checkpoint):
        raise ValueError(f"Corrected SODA checkpoint is incomplete: {checkpoint}")
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != BASE:
        raise ValueError(f"Unexpected corrected SODA adapter base: {checkpoint}")
    return checkpoint, {
        "source": "local_corrected_soda",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "checkpoint": str(checkpoint),
    }


def checkpoint_for(attack, defense, checkpoint_root, direct_checkpoint=None):
    if direct_checkpoint is not None:
        checkpoint = Path(direct_checkpoint).expanduser().resolve()
        if not _checkpoint_complete(checkpoint):
            raise ValueError(f"Incomplete local adapter checkpoint: {checkpoint}")
        config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        if config.get("base_model_name_or_path") != BASE:
            raise ValueError(f"Unexpected adapter base model: {config}")
        return checkpoint, {"source": "manifest", "checkpoint": str(checkpoint)}
    if attack == "soda":
        local = local_corrected_soda_checkpoint(defense)
        if local is not None:
            return local
    folder = CLEAN[attack] if defense == "clean" else f"{defense}/checkpoint-final"
    artifact_root = checkpoint_root / f"mea-{attack}-defenses"
    checkpoint = artifact_root / folder
    if not (checkpoint / "adapter_config.json").exists():
        raise ValueError(f"Missing local adapter: {checkpoint}")
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != BASE:
        raise ValueError(f"Unexpected adapter base model: {config}")
    return checkpoint, {"source": "local", "root": str(artifact_root), "folder": folder}


def main():
    parser = argparse.ArgumentParser(description="Defense M1 ACC and M2 BERTScore; reuse existing detection reports.")
    parser.add_argument("--stage", choices=("reference", "students"), required=True)
    parser.add_argument("--attack", choices=tuple(CLEAN))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--checkpoint-root", type=Path,
                        help="Local directory containing mea-<attack>-defenses folders.")
    parser.add_argument("--defense", choices=DEFENSES,
                        help="Evaluate one manifest-resolved defense instead of the legacy full bundle.")
    parser.add_argument("--checkpoint", type=Path,
                        help="Checkpoint recorded by the selected defense manifest.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--m1-only", action="store_true",
                        help="Recompute M1 without regenerating held-out answers or BERTScore.")
    parser.add_argument("--reuse-gsm8k-root",
                        help="Previous defense-evaluation root whose matching GSM8K rows are reused.")
    parser.add_argument("--reuse-soda-gsm8k-root",
                        help="Optional corrected-SODA evaluation root used instead for SODA students.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.stage == "students" and not args.attack:
        parser.error("Student evaluation requires --attack")
    if args.stage == "students" and args.defense and not args.checkpoint:
        parser.error("Single-defense evaluation requires --checkpoint")
    if args.stage == "students" and not args.defense and not args.checkpoint_root:
        parser.error("Legacy bundle evaluation requires --checkpoint-root")
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompts = [] if args.m1_only else load_prompts(Path(args.prompts_jsonl))
    if prompts and len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("Duplicate heldout prompt ids")
    if args.limit and prompts:
        prompts = prompts[:args.limit]
    protocol_dir = output / ("reference" if args.stage == "reference" else args.attack)
    tasks, metadata = prepare_tasks(Path("evaluation/configs/m1_rollout.yaml"), args.limit, protocol_dir)
    protocol = {"teacher": TEACHER, "base": BASE, "tasks": metadata,
                "seed": 42, "teacher_rendering": "chat", "student_rendering": "raw_text",
                "m1_multiple_choice_scoring": SCORING_METHOD,
                "math_max_tokens": 512, "m1_only": args.m1_only,
                "gsm8k_reuse_root": str(Path(args.reuse_gsm8k_root).resolve()) if args.reuse_gsm8k_root else None,
                "soda_gsm8k_reuse_root": str(Path(args.reuse_soda_gsm8k_root).resolve())
                if args.reuse_soda_gsm8k_root else None}
    if not args.m1_only:
        protocol.update({"prompts": prompts, "heldout_max_tokens": 1536,
                         "bert_model": "roberta-large", "rescale_with_baseline": True})
    if args.stage == "reference":
        check_protocol(output / "reference" / "protocol.json", protocol)
        reuse_root = Path(args.reuse_gsm8k_root).resolve() / "reference" if args.reuse_gsm8k_root else None
        evaluate_model(output / "reference" / "teacher", TEACHER, True, tasks, prompts, not args.m1_only,
                       reuse_gsm8k_directory=reuse_root / "teacher" if reuse_root else None)
        evaluate_model(output / "reference" / "base", BASE, False, tasks, prompts, False,
                       reuse_gsm8k_directory=reuse_root / "base" if reuse_root else None)
        write_json(output / "reference" / "complete.json", {"complete": True, "smoke": bool(args.limit),
                                                               "m1_only": args.m1_only})
        return
    reference = output / "reference"
    if not (reference / "complete.json").exists():
        raise ValueError("Run the shared reference stage first")
    if json.loads((reference / "protocol.json").read_text(encoding="utf-8")) != protocol:
        raise ValueError("Reference/student protocols differ")
    check_protocol(protocol_dir / "protocol.json", protocol)
    teacher = {} if args.m1_only else resume_rows(reference / "teacher" / "heldout_outputs.jsonl")
    if not args.m1_only:
        for prompt in prompts:
            row = teacher.get(prompt["id"])
            if not row or row["prompt"] != prompt["prompt"] or row.get("model_id") != TEACHER:
                raise ValueError(f"Invalid Qwen teacher reference: {prompt['id']}")
    teacher_acc = json.loads((reference / "teacher" / "m1_acc.json").read_text())["acc"]
    base_acc = json.loads((reference / "base" / "m1_acc.json").read_text())["acc"]
    summary = []
    selected_defenses = (args.defense,) if args.defense else DEFENSES
    for defense in selected_defenses:
        directory = protocol_dir / defense
        checkpoint, source = checkpoint_for(
            args.attack,
            defense,
            None if args.checkpoint_root is None else args.checkpoint_root.expanduser().resolve(),
            args.checkpoint if args.defense else None,
        )
        check_protocol(directory / "source.json", source)
        print(f"Evaluating {args.attack}/{defense}", flush=True)
        reuse_value = (args.reuse_soda_gsm8k_root
                       if args.attack == "soda" and args.reuse_soda_gsm8k_root else args.reuse_gsm8k_root)
        reuse_directory = Path(reuse_value).resolve() / args.attack / defense if reuse_value else None
        acc, rows = evaluate_model(directory, BASE, False, tasks, prompts, not args.m1_only, checkpoint,
                                   reuse_gsm8k_directory=reuse_directory)
        score = None
        if not args.m1_only:
            score = bertscore_f1([rows[p["id"]]["text"] for p in prompts],
                                 [teacher[p["id"]]["text"] for p in prompts], "en", "roberta-large", 16, "cuda:0")
            write_json(directory / "m2_bertscore.json", {"bertscore": score, "n": len(prompts), "reference": TEACHER})
        clean = summary[0] if summary and summary[0]["defense"] == "clean" else None
        row = {"attack": args.attack, "defense": defense, "acc": acc,
               "teacher_acc": teacher_acc, "base_acc": base_acc,
               "gain_over_base": acc - base_acc,
               "acc_drop_from_clean": clean["acc"] - acc if clean else 0.0,
               "smoke": bool(args.limit), "m1_protocol": SCORING_METHOD}
        if not args.m1_only:
            row.update({"bertscore": score,
                        "bertscore_drop_from_clean": clean["bertscore"] - score if clean else 0.0})
        summary.append(row)
        write_json(protocol_dir / "summary.json", summary)
        with (protocol_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print(f"Finished: {protocol_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
