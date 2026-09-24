
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics.m1_capability_reports import openllm6_report, read_json_object
from evaluation.metrics.m2_fidelity import text_report
from evaluation.metrics.m3_quality import repetition_report
from evaluation.metrics.m6_cost import cost_report

OPENLLM6_TASKS = "arc_challenge,hellaswag,mmlu,truthfulqa_mc2,winogrande,gsm8k"
PROMPT_FIELDS = ("prompt", "query", "prompt_text", "source_prompt")
TEACHER_TEXT_FIELDS = ("teacher_text", "teacher_response", "response", "text")
STUDENT_TEXT_FIELDS = ("student_text", "response", "text")


@dataclass
class AttackRun:
    attack: str
    run_id: str
    manifest_path: str
    run_dir: str
    status: str | None
    checkpoint: str | None
    budget: int | None
    teacher_model: str | None
    student_model: str | None
    stage1_results: str | None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def localize_uploaded_path(value: Any, run_dir: Path, run_id: str | None) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    if run_id and run_id in raw.parts:
        parts = list(raw.parts)
        candidate = run_dir.joinpath(*parts[parts.index(run_id) + 1 :])
        if candidate.exists():
            return candidate.resolve()
    if raw.exists():
        return raw.resolve()
    return None


def find_checkpoint(run_dir: Path, manifest: dict[str, Any], run_id: str | None) -> Path | None:
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    training = artifacts.get("training") if isinstance(artifacts.get("training"), dict) else {}
    warmup = artifacts.get("generator_warmup") if isinstance(artifacts.get("generator_warmup"), dict) else {}
    values = [
        result.get("checkpoint_dir"),
        manifest.get("checkpoint_dir"),
        artifacts.get("checkpoint_dir"),
        artifacts.get("planned_checkpoint_dir"),
        training.get("checkpoint"),
        warmup.get("checkpoint"),
    ]
    for value in values:
        candidate = localize_uploaded_path(value, run_dir, run_id)
        if candidate and (candidate / "adapter_config.json").exists():
            return candidate
    checkpoints = sorted(
        {p.parent.resolve() for p in run_dir.rglob("adapter_config.json")},
        key=lambda p: (0 if "checkpoint-final" in str(p) else 1, len(p.parts), str(p)),
    )
    return checkpoints[0] if checkpoints else None


def find_stage1_results(run_dir: Path, manifest: dict[str, Any], run_id: str | None) -> Path | None:
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    training = artifacts.get("training") if isinstance(artifacts.get("training"), dict) else {}
    for value in (training.get("result_json_path"), artifacts.get("result_json_path")):
        candidate = localize_uploaded_path(value, run_dir, run_id)
        if candidate and candidate.exists():
            return candidate
    candidates = sorted(run_dir.rglob("stage1_results_v1.json"))
    return candidates[0].resolve() if candidates else None


def discover_attacks(root: Path) -> list[AttackRun]:
    attacks: list[AttackRun] = []
    for manifest_path in sorted(root.rglob("attack_manifest.json")):
        manifest = read_json(manifest_path)
        result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
        run_config = manifest.get("run_config") if isinstance(manifest.get("run_config"), dict) else {}
        attack = str(result.get("attack") or run_config.get("attack") or manifest_path.parent.parent.name)
        run_id = str(result.get("run_id") or manifest_path.parent.name)
        checkpoint = find_checkpoint(manifest_path.parent, manifest, run_id)
        stage1_results = find_stage1_results(manifest_path.parent, manifest, run_id)
        attacks.append(
            AttackRun(
                attack=attack,
                run_id=run_id,
                manifest_path=str(manifest_path.resolve()),
                run_dir=str(manifest_path.parent.resolve()),
                status=result.get("status"),
                checkpoint=str(checkpoint) if checkpoint else None,
                budget=result.get("budget") if isinstance(result.get("budget"), int) else None,
                teacher_model=run_config.get("teacher_model"),
                student_model=run_config.get("student_model"),
                stage1_results=str(stage1_results) if stage1_results else None,
            )
        )
    return attacks


def m6_row_from_attack(attack: AttackRun, args: argparse.Namespace) -> dict[str, Any]:
    query_count = attack.budget
    wall_clock_hours = None
    gpu_hours = None
    checkpoint_size = None
    if attack.stage1_results:
        payload = read_json(Path(attack.stage1_results))
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        m6 = metrics.get("M6") if isinstance(metrics.get("M6"), dict) else {}
        query_count = m6.get("teacher_successful_queries") or m6.get("teacher_query_attempts") or query_count
        wall_seconds = m6.get("wall_clock_seconds")
        if isinstance(wall_seconds, (int, float)) and math.isfinite(float(wall_seconds)):
            wall_clock_hours = float(wall_seconds) / 3600.0
        if isinstance(m6.get("gpu_hours"), (int, float)):
            gpu_hours = float(m6["gpu_hours"])
        checkpoint_size = m6.get("checkpoint_size_bytes")
    else:
        manifest = read_json(Path(attack.manifest_path))
        result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        wall_seconds = metrics.get("wall_clock_seconds")
        if isinstance(wall_seconds, (int, float)) and math.isfinite(float(wall_seconds)):
            wall_clock_hours = float(wall_seconds) / 3600.0
        if isinstance(metrics.get("teacher_successful_queries"), int):
            query_count = metrics["teacher_successful_queries"]
    if attack.checkpoint:
        try:
            checkpoint_size = sum(p.stat().st_size for p in Path(attack.checkpoint).rglob("*") if p.is_file())
        except OSError:
            pass
    return {
        "id": f"{attack.attack}:{attack.run_id}:attack",
        "method": attack.attack,
        "stage": "attack",
        "currency": args.currency,
        "price_as_of": args.price_as_of,
        "api_billing": "unknown",
        "query_count": int(query_count) if isinstance(query_count, int) else None,
        "sample_count": int(query_count) if isinstance(query_count, int) else None,
        "gpu_hours": gpu_hours,
        "gpu_price_per_hour": args.gpu_price_per_hour,
        "wall_clock_hours": wall_clock_hours,
        "data_cost": 0.0,
        "human_cost": 0.0,
        "other_cost": 0.0,
        "budget_cap": attack.budget,
        "linked_effect": {"family": "artifact", "metric": "checkpoint_size_bytes", "value": checkpoint_size} if checkpoint_size is not None else None,
    }


def first_text(row: dict[str, Any], fields: tuple[str, ...], source: Path) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str):
            return value
    raise ValueError(f"{source} row missing one of {fields}")


def get_id(row: dict[str, Any], index: int) -> str:
    for field in ("id", "query_id", "prompt_id"):
        if row.get(field) is not None:
            return str(row[field])
    return str(index)


def load_prompts(path: Path) -> list[dict[str, str]]:
    rows = [{"id": get_id(row, i), "prompt": first_text(row, PROMPT_FIELDS, path)} for i, row in enumerate(iter_jsonl(path), 1)]
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def load_teacher(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    for i, row in enumerate(iter_jsonl(path), 1):
        item_id = get_id(row, i)
        prompt = ""
        for field in PROMPT_FIELDS:
            if isinstance(row.get(field), str):
                prompt = row[field]
                break
        rows[item_id] = {"id": item_id, "prompt": prompt, "teacher_text": first_text(row, TEACHER_TEXT_FIELDS, path)}
    if not rows:
        raise ValueError(f"No teacher outputs found in {path}")
    return rows


def generate_student(attack: AttackRun, prompts: list[dict[str, str]], out: Path, args: argparse.Namespace) -> None:
    if not attack.checkpoint:
        raise ValueError(f"{attack.attack} has no checkpoint")
    from attacks.core.student_model import GenerationConfig, load_student_from_checkpoint

    student = load_student_from_checkpoint(
        attack.checkpoint,
        base_model=args.base_model,
        student_name=attack.attack,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        use_chat_template=args.use_chat_template,
    )
    write_jsonl(
        out,
        (
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "student_text": student.generate_one(item["prompt"], generation_config=config),
                "attack": attack.attack,
                "run_id": attack.run_id,
                "checkpoint": attack.checkpoint,
                "generation_config": config.to_dict(),
            }
            for item in prompts
        ),
    )


def build_pairs(student_path: Path, teacher: dict[str, dict[str, str]], out: Path) -> None:
    pairs = []
    for i, row in enumerate(iter_jsonl(student_path), 1):
        item_id = get_id(row, i)
        if item_id not in teacher:
            continue
        pairs.append(
            {
                "id": item_id,
                "prompt": row.get("prompt") or teacher[item_id].get("prompt") or "",
                "teacher_text": teacher[item_id]["teacher_text"],
                "student_text": first_text(row, STUDENT_TEXT_FIELDS, student_path),
            }
        )
    if not pairs:
        raise ValueError(f"No matching teacher/student pairs for {student_path}")
    write_jsonl(out, pairs)


def compute_m2_m3(attack: AttackRun, student_path: Path, teacher: dict[str, dict[str, str]], output_root: Path, args: argparse.Namespace) -> dict[str, str]:
    pair_path = output_root / "pairs" / f"{attack.attack}.jsonl"
    build_pairs(student_path, teacher, pair_path)
    m2_path = output_root / "m2_text" / f"{attack.attack}.json"
    m2_args = SimpleNamespace(
        inputs=[str(pair_path)],
        base_inputs=None,
        bertscore=args.bertscore,
        mauve=args.mauve,
        lang=args.bertscore_lang,
        bert_model=args.bert_model,
        batch_size=args.bertscore_batch_size,
        device=args.bertscore_device,
        mauve_device=args.mauve_device,
        max_text_length=args.mauve_max_text_length,
    )
    write_json(m2_path, text_report(m2_args))
    texts = [first_text(row, STUDENT_TEXT_FIELDS, student_path) for row in iter_jsonl(student_path)]
    m3_path = output_root / "m3_repetition" / f"{attack.attack}.json"
    write_json(m3_path, repetition_report(texts, ns=args.rep_n, tokenizer=args.rep_tokenizer, short_policy=args.rep_short_policy))
    return {"pairs": str(pair_path), "m2_text": str(m2_path), "m3_repetition": str(m3_path)}


def run_lm_eval(attack: AttackRun, output_root: Path, args: argparse.Namespace) -> str | None:
    if not attack.checkpoint:
        return None
    out_dir = output_root / "lm_eval" / attack.attack
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.lm_eval_bin,
        "--model", "hf",
        "--model_args", f"pretrained={args.base_model},peft={attack.checkpoint},trust_remote_code=True",
        "--tasks", args.lm_eval_tasks,
        "--batch_size", str(args.lm_eval_batch_size),
        "--output_path", str(out_dir),
        "--log_samples",
    ]
    (out_dir / "lm_eval_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    if not args.dry_run:
        subprocess.run(command, check=True)
    return str(out_dir)


def find_result_json(path: Path) -> Path | None:
    candidates = sorted(path.rglob("results*.json")) if path.is_dir() else ([path] if path.exists() else [])
    return candidates[-1] if candidates else None


def maybe_m1(attack: AttackRun, lm_eval_dir: str | None, output_root: Path) -> str | None:
    if not lm_eval_dir:
        return None
    result_path = find_result_json(Path(lm_eval_dir))
    if not result_path:
        return None
    out = output_root / "m1_openllm6" / f"{attack.attack}.json"
    write_json(out, openllm6_report(read_json_object(result_path)))
    return str(out)


def summarize(attacks: list[AttackRun], artifacts: dict[str, dict[str, str | None]], output_root: Path) -> None:
    rows = []
    for attack in attacks:
        row = asdict(attack)
        row.update(artifacts.get(attack.attack, {}))
        rows.append(row)
    write_json(output_root / "summary.json", rows)
    fields = sorted({key for row in rows for key in row})
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate completed attack outputs with M1/M2/M3/M6 helpers.")
    parser.add_argument("--attack-root", required=True, help="Local attack output tree.")
    parser.add_argument("--output-root", default="results/attack_eval/latest")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--eval-prompts-jsonl")
    parser.add_argument("--teacher-jsonl")
    parser.add_argument("--generate-student", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--mauve", action="store_true")
    parser.add_argument("--bertscore-lang", default="en")
    parser.add_argument("--bert-model")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-device")
    parser.add_argument("--mauve-device", type=int, default=-1)
    parser.add_argument("--mauve-max-text-length", type=int, default=512)
    parser.add_argument("--rep-n", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--rep-tokenizer", choices=("unicode_cjk", "whitespace"), default="unicode_cjk")
    parser.add_argument("--rep-short-policy", choices=("zero", "exclude"), default="zero")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--price-as-of", default=date.today().isoformat())
    parser.add_argument("--gpu-price-per-hour", type=float)
    parser.add_argument("--run-lm-eval", action="store_true")
    parser.add_argument("--lm-eval-bin", default="lm_eval")
    parser.add_argument("--lm-eval-tasks", default=OPENLLM6_TASKS)
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_attack_root(args: argparse.Namespace, output_root: Path) -> Path:
    return Path(args.attack_root).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    attack_root = resolve_attack_root(args, output_root)
    attacks = discover_attacks(attack_root)
    if not attacks:
        raise SystemExit(f"No attack_manifest.json files found under {attack_root}")
    write_json(output_root / "discovered_attacks.json", [asdict(attack) for attack in attacks])

    m6_rows = [m6_row_from_attack(attack, args) for attack in attacks]
    m6_costs = output_root / "m6_costs.jsonl"
    write_jsonl(m6_costs, m6_rows)
    write_json(output_root / "m6_report.json", cost_report(m6_rows))

    prompts = load_prompts(Path(args.eval_prompts_jsonl).expanduser().resolve()) if args.eval_prompts_jsonl else None
    teacher = load_teacher(Path(args.teacher_jsonl).expanduser().resolve()) if args.teacher_jsonl else None
    artifacts: dict[str, dict[str, str | None]] = {}
    student_dir = output_root / "student_generations"

    for attack in attacks:
        attack_artifacts: dict[str, str | None] = {"m6_costs": str(m6_costs), "m6_report": str(output_root / "m6_report.json")}
        student_path = student_dir / f"{attack.attack}.jsonl"
        if args.generate_student:
            if prompts is None:
                raise SystemExit("--generate-student requires --eval-prompts-jsonl")
            if args.dry_run:
                student_path.parent.mkdir(parents=True, exist_ok=True)
                student_path.write_text("", encoding="utf-8")
            else:
                generate_student(attack, prompts, student_path, args)
            attack_artifacts["student_generations"] = str(student_path)
        elif student_path.exists() and student_path.stat().st_size > 0:
            attack_artifacts["student_generations"] = str(student_path)

        if teacher is not None and student_path.exists() and student_path.stat().st_size > 0:
            attack_artifacts.update(compute_m2_m3(attack, student_path, teacher, output_root, args))

        lm_eval_dir = None
        if args.run_lm_eval:
            if not args.dry_run and shutil.which(args.lm_eval_bin) is None:
                raise SystemExit(f"Could not find lm-eval binary: {args.lm_eval_bin}")
            lm_eval_dir = run_lm_eval(attack, output_root, args)
            attack_artifacts["lm_eval_dir"] = lm_eval_dir
        attack_artifacts["m1_openllm6"] = maybe_m1(attack, lm_eval_dir, output_root)
        artifacts[attack.attack] = attack_artifacts

    summarize(attacks, artifacts, output_root)
    print(json.dumps({"output_root": str(output_root), "attack_root": str(attack_root), "attacks": len(attacks)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
