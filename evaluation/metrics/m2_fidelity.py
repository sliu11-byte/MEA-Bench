#!/usr/bin/env python3
"""M2 fidelity metrics for teacher-student model comparison.

Commands:
  agreement  Compare teacher/student lm-eval --log_samples predictions.
  text       Compute paired ROUGE-L and optional BERTScore/MAUVE.
  cross-ppl  Let teacher and student models score each other's outputs.

The text and cross-ppl commands expect JSONL rows with this schema:
  {"id": "1", "prompt": "...", "teacher_text": "...", "student_text": "..."}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SAMPLE_TIMESTAMP = re.compile(r"_\d{4}-\d{2}-\d{2}T.*$")
ROUGE_TOKEN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[^\s]"
)
LENGTH_NORMALIZED_TASKS = ("arc_challenge", "hellaswag")
GENERATION_TASKS = ("gsm8k",)


def iter_jsonl(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                yield row


def write_report(report: dict[str, Any], output: str | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def task_from_sample_path(path: Path) -> str:
    name = path.stem
    if not name.startswith("samples_"):
        raise ValueError(f"Not an lm-eval sample file: {path}")
    task = name[len("samples_") :]
    return SAMPLE_TIMESTAMP.sub("", task)


def latest_sample_files(path_value: str) -> list[Path]:
    path = Path(path_value)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Path does not exist: {path}")

    latest: dict[str, Path] = {}
    for sample_path in path.rglob("samples_*.jsonl"):
        task = task_from_sample_path(sample_path)
        current = latest.get(task)
        if current is None or sample_path.stat().st_mtime_ns > current.stat().st_mtime_ns:
            latest[task] = sample_path
    if not latest:
        raise ValueError(f"No samples_*.jsonl files found under {path}")
    return [latest[task] for task in sorted(latest)]


def first_number(value: Any) -> float:
    while isinstance(value, list) and value:
        value = value[0]
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected a numeric log-likelihood, got {value!r}")
    return float(value)


def continuation_byte_length(argument: Any) -> int:
    if not isinstance(argument, list) or len(argument) < 2:
        raise ValueError("Length-normalized tasks need [context, continuation] arguments")
    continuation = argument[1]
    if not isinstance(continuation, str):
        raise ValueError("Length-normalized task continuation must be text")
    return max(1, len(continuation.encode("utf-8")))


def prediction_from_sample(row: dict[str, Any], task: str) -> int:
    responses = row.get("filtered_resps", row.get("resps"))
    if not isinstance(responses, list) or len(responses) < 2:
        raise ValueError("Sample does not contain multiple-choice filtered_resps")

    scores = [first_number(response) for response in responses]
    if task.startswith(LENGTH_NORMALIZED_TASKS):
        arguments = row.get("arguments")
        if not isinstance(arguments, list) or len(arguments) != len(scores):
            raise ValueError("Response and argument counts differ")
        scores = [
            score / continuation_byte_length(argument)
            for score, argument in zip(scores, arguments, strict=True)
        ]
    return max(range(len(scores)), key=scores.__getitem__)


def prompt_signature(row: dict[str, Any]) -> str:
    prompt_hash = row.get("prompt_hash")
    if isinstance(prompt_hash, str) and prompt_hash:
        return f"lm-eval:{prompt_hash}"
    arguments = row.get("arguments")
    if not isinstance(arguments, list):
        raise ValueError("Sample needs prompt_hash or arguments for prompt alignment")
    canonical = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"arguments-sha256:{hashlib.sha256(canonical).hexdigest()}"


def sample_key(task: str, row: dict[str, Any]) -> tuple[str, str, str, str]:
    doc_id = row.get("doc_id")
    if doc_id is None:
        raise ValueError(f"{task} sample is missing doc_id")
    return (
        task,
        str(doc_id),
        str(row.get("filter", "none")),
        prompt_signature(row),
    )


def load_lm_eval_predictions(
    path_value: str,
) -> tuple[dict[tuple[str, str, str, str], int], int]:
    predictions: dict[tuple[str, str, str, str], int] = {}
    skipped = 0
    for path in latest_sample_files(path_value):
        task = task_from_sample_path(path)
        if task.startswith(GENERATION_TASKS):
            skipped += sum(1 for _ in iter_jsonl([path]))
            continue
        for row in iter_jsonl([path]):
            key = sample_key(task, row)
            if key in predictions:
                raise ValueError(f"Duplicate lm-eval sample: {key}")
            predictions[key] = prediction_from_sample(row, task)
    return predictions, skipped


def agreement_report(
    teacher_path: str,
    student_path: str,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    teacher, teacher_skipped = load_lm_eval_predictions(teacher_path)
    student, student_skipped = load_lm_eval_predictions(student_path)
    teacher_only = teacher.keys() - student.keys()
    student_only = student.keys() - teacher.keys()
    if not allow_partial and (teacher_only or student_only):
        raise ValueError(
            "Teacher and student sample sets differ: "
            f"teacher_only={len(teacher_only)}, student_only={len(student_only)}"
        )
    common = sorted(teacher.keys() & student.keys())
    if not common:
        raise ValueError("Teacher and student have no matching samples")

    per_task_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for key in common:
        task = key[0]
        per_task_counts[task][1] += 1
        if teacher[key] == student[key]:
            per_task_counts[task][0] += 1

    agreements = sum(teacher[key] == student[key] for key in common)
    per_task = [
        {"task": task, "agreement_rate": matched / total, "matched": matched, "n": total}
        for task, (matched, total) in sorted(per_task_counts.items())
    ]
    return {
        "metric": "agreement_rate",
        "agreement_rate": agreements / len(common),
        "teacher_normalized_agreement_rate": agreements / len(common),
        "teacher_self_denominator": 1.0,
        "matched": agreements,
        "n": len(common),
        "teacher_only": len(teacher_only),
        "student_only": len(student_only),
        "generation_rows_skipped": {
            "teacher": teacher_skipped,
            "student": student_skipped,
        },
        "per_task": per_task,
        "protocol": {
            "arc_challenge": "argmax(loglikelihood / continuation UTF-8 byte length)",
            "hellaswag": "argmax(loglikelihood / continuation UTF-8 byte length)",
            "other_mcq": "argmax(loglikelihood)",
            "sample_alignment": (
                "intersection allowed for diagnostics"
                if allow_partial else "exact teacher/student sample-key equality required"
            ),
            "sample_key": "task + doc_id + filter + prompt_hash/arguments SHA-256",
        },
    }


def text_pairs(
    paths: Iterable[str],
    *,
    require_prompt: bool = False,
) -> list[dict[str, str]]:
    pairs = []
    seen_ids = set()
    for index, row in enumerate(iter_jsonl(paths), start=1):
        pair_id = str(row.get("id", index))
        if pair_id in seen_ids:
            raise ValueError(f"Duplicate text-pair id: {pair_id}")
        seen_ids.add(pair_id)
        for field in ("teacher_text", "student_text"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"Pair {pair_id} is missing text field {field!r}")
        prompt = row.get("prompt")
        if require_prompt and (
            not isinstance(prompt, str) or not prompt.strip()
        ):
            raise ValueError(
                f"Pair {pair_id} needs a non-empty string prompt for cross-PPL"
            )
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"Pair {pair_id} prompt must be text when provided")
        pairs.append(
            {
                "id": pair_id,
                "prompt": prompt if isinstance(prompt, str) else "",
                "teacher_text": row["teacher_text"],
                "student_text": row["student_text"],
            }
        )
    if not pairs:
        raise ValueError("No text pairs found")
    return pairs


def rouge_tokens(text: str) -> list[str]:
    return [token.lower() for token in ROUGE_TOKEN.findall(text)]


def lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    candidate_tokens = rouge_tokens(candidate)
    reference_tokens = rouge_tokens(reference)
    if not candidate_tokens and not reference_tokens:
        return 1.0
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = lcs_length(candidate_tokens, reference_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def bertscore_f1(
    candidates: list[str],
    references: list[str],
    lang: str,
    model_type: str | None,
    batch_size: int,
    device: str | None,
) -> float:
    if len(candidates) != len(references) or not candidates:
        raise ValueError("BERTScore needs non-empty paired candidate/reference lists")
    try:
        from bert_score import score
    except ImportError as exc:
        raise RuntimeError("BERTScore requires: pip install bert-score==0.3.13") from exc
    patch_roberta_special_tokens_for_transformers5()

    valid = [i for i, (candidate, reference) in enumerate(zip(candidates, references, strict=True))
             if candidate.strip() and reference.strip()]
    total = sum(1.0 for candidate, reference in zip(candidates, references, strict=True)
                if not candidate.strip() and not reference.strip())
    if not valid:
        return total / len(candidates)
    kwargs: dict[str, Any] = {
        "cands": [candidates[i] for i in valid],
        "refs": [references[i] for i in valid],
        "lang": lang,
        "batch_size": batch_size,
        "rescale_with_baseline": True,
        "verbose": False,
    }
    if model_type:
        kwargs["model_type"] = model_type
    if device:
        kwargs["device"] = device
    _, _, f1 = score(**kwargs)
    # Empty-vs-nonempty is defined as zero; empty-vs-empty as one. This avoids
    # relying on tokenizer behavior that changed in Transformers 5.
    return (total + float(f1.sum().item())) / len(candidates)


def patch_roberta_special_tokens_for_transformers5() -> None:
    """Keep bert-score 0.3.x compatible with Transformers 5 RoBERTa tokenizers."""
    try:
        import transformers
    except ImportError:
        return

    def build_roberta_inputs(self: Any, token_ids_0: list[int], token_ids_1: list[int] | None = None) -> list[int]:
        cls_id = self.cls_token_id
        sep_id = self.sep_token_id
        if cls_id is None or sep_id is None:
            raise ValueError("RoBERTa tokenizer needs cls_token_id and sep_token_id for BERTScore")
        if token_ids_1 is None:
            return [cls_id] + list(token_ids_0) + [sep_id]
        return [cls_id] + list(token_ids_0) + [sep_id, sep_id] + list(token_ids_1) + [sep_id]

    for class_name in ("RobertaTokenizer", "RobertaTokenizerFast"):
        tokenizer_class = getattr(transformers, class_name, None)
        if tokenizer_class is not None and not hasattr(tokenizer_class, "build_inputs_with_special_tokens"):
            setattr(tokenizer_class, "build_inputs_with_special_tokens", build_roberta_inputs)


def mauve_score(teacher_texts: list[str], student_texts: list[str], device_id: int, max_text_length: int) -> float:
    try:
        import mauve
    except ImportError as exc:
        raise RuntimeError("MAUVE requires: pip install mauve-text==0.4.0") from exc

    result = mauve.compute_mauve(
        p_text=teacher_texts,
        q_text=student_texts,
        device_id=device_id,
        max_text_length=max_text_length,
        verbose=False,
    )
    return float(result.mauve)


def require_aligned_text_pairs(
    reference_pairs: list[dict[str, str]],
    comparison_pairs: list[dict[str, str]],
    comparison_name: str,
) -> None:
    reference = {pair["id"]: pair for pair in reference_pairs}
    comparison = {pair["id"]: pair for pair in comparison_pairs}
    if set(reference) != set(comparison):
        raise ValueError(f"Teacher/student and {comparison_name} pair IDs differ")
    for pair_id, pair in reference.items():
        other = comparison[pair_id]
        for field in ("prompt", "teacher_text"):
            if pair[field] != other[field]:
                raise ValueError(
                    f"Pair {pair_id}: {field} differs between student and {comparison_name} inputs"
                )


def text_metrics(pairs: list[dict[str, str]], args: argparse.Namespace) -> dict[str, float]:
    teachers = [pair["teacher_text"] for pair in pairs]
    students = [pair["student_text"] for pair in pairs]
    rouge_scores = [
        rouge_l_f1(candidate=student, reference=teacher)
        for teacher, student in zip(teachers, students, strict=True)
    ]
    metrics = {"rouge_l_f1": sum(rouge_scores) / len(rouge_scores)}
    if args.bertscore:
        metrics["bertscore_f1_rescaled"] = bertscore_f1(
            students,
            teachers,
            lang=args.lang,
            model_type=args.bert_model,
            batch_size=args.batch_size,
            device=args.device,
        )
    if args.mauve:
        metrics["mauve"] = mauve_score(
            teachers,
            students,
            device_id=args.mauve_device,
            max_text_length=args.max_text_length,
        )
    return metrics


def text_report(args: argparse.Namespace) -> dict[str, Any]:
    pairs = text_pairs(args.inputs)
    metrics = text_metrics(pairs, args)
    report: dict[str, Any] = {
        "metrics": metrics,
        "teacher_normalized": {
            metric: value for metric, value in metrics.items()
        },
        "teacher_self_denominator": {
            metric: 1.0 for metric in metrics
        },
        "n": len(pairs),
        "protocol": {
            "reference": "teacher_text",
            "candidate": "student_text",
            "rouge_tokenizer": "lowercase Unicode tokens; CJK characters are separate tokens",
            "teacher_normalization": "Similarity metrics use their exact teacher-vs-teacher optimum of 1.0 as denominator.",
        },
    }
    if args.base_inputs:
        base_pairs = text_pairs(args.base_inputs)
        require_aligned_text_pairs(pairs, base_pairs, "bare-base")
        base_metrics = text_metrics(base_pairs, args)
        report["bare_base_floor"] = {
            "metrics": base_metrics,
            "n": len(base_pairs),
        }
        report["gain_over_bare_base"] = {
            metric: metrics[metric] - base_metrics[metric]
            for metric in metrics
        }
    return report


def import_model_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("cross-ppl requires PyTorch and Transformers") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise ValueError("cross-PPL requires a fast tokenizer backend")
    special_tokens = getattr(tokenizer, "special_tokens_map", {})
    special_text = repr(
        sorted((str(key), str(value)) for key, value in special_tokens.items())
    )
    payload = backend.to_str() + "\n" + special_text
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def joint_completion_tokenization(
    tokenizer: Any,
    prompt: str,
    completion: str,
    pair_id: str,
) -> tuple[list[int], list[bool]]:
    if not completion:
        raise ValueError(f"Pair {pair_id} has an empty completion")
    encoded = tokenizer(
        prompt + completion,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if not isinstance(input_ids, list) or not isinstance(offsets, list):
        raise ValueError("cross-PPL tokenizer must provide offset_mapping")
    if len(input_ids) != len(offsets):
        raise ValueError(f"Pair {pair_id} tokenizer offsets are misaligned")
    boundary = len(prompt)
    completion_mask = []
    for offset in offsets:
        if (
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or not all(isinstance(value, int) for value in offset)
        ):
            raise ValueError(f"Pair {pair_id} tokenizer offsets are invalid")
        start, end = offset
        if start < boundary < end:
            raise ValueError(
                f"Pair {pair_id} prompt/completion boundary splits a tokenizer token"
            )
        completion_mask.append(start >= boundary and end > start)
    if not any(completion_mask):
        raise ValueError(f"Pair {pair_id} has no scoreable completion tokens")
    return [int(token_id) for token_id in input_ids], completion_mask


def symmetric_cross_ppl(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[float | None, dict[str, Any]]:
    if left["perplexity"] is None or right["perplexity"] is None:
        return None, {
            "available": False,
            "reason": "At least one perplexity overflowed the finite JSON range.",
        }
    same_tokenizer = left["tokenizer_fingerprint"] == right["tokenizer_fingerprint"]
    if not same_tokenizer:
        return None, {
            "available": False,
            "reason": "The two PPL directions use different tokenizer vocabularies.",
        }
    return math.sqrt(left["perplexity"] * right["perplexity"]), {
        "available": True,
        "reason": None,
    }


def score_completions(
    model_name: str,
    model_revision: str,
    pairs: list[dict[str, str]],
    completion_field: str,
    device: str,
    max_length: int,
    trust_remote_code: bool,
) -> dict[str, Any]:
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("cross-PPL model revision must be pinned explicitly")
    torch, auto_model, auto_tokenizer = import_model_dependencies()
    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = auto_tokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
        revision=model_revision,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("cross-PPL requires a fast tokenizer with offset_mapping")
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    model = auto_model.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
        revision=model_revision,
    ).to(resolved_device)
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for pair in pairs:
            joint_ids, completion_mask = joint_completion_tokenization(
                tokenizer,
                pair["prompt"],
                pair[completion_field],
                pair["id"],
            )
            bos_token_id = getattr(tokenizer, "bos_token_id", None)
            bos_ids = [int(bos_token_id)] if bos_token_id is not None else []
            input_ids = bos_ids + joint_ids
            if len(input_ids) > max_length:
                raise ValueError(
                    f"Pair {pair['id']} has {len(input_ids)} tokens, above --max-length {max_length}"
                )

            inputs = torch.tensor([input_ids], dtype=torch.long, device=resolved_device)
            label_ids = [-100] * len(input_ids)
            for index, should_score in enumerate(completion_mask):
                if should_score:
                    label_ids[len(bos_ids) + index] = joint_ids[index]
            scored_tokens = sum(label != -100 for label in label_ids[1:])
            if scored_tokens == 0:
                raise ValueError(
                    f"Pair {pair['id']} has no causally scoreable completion tokens"
                )
            labels = torch.tensor(
                [label_ids], dtype=torch.long, device=resolved_device
            )
            outputs = model(input_ids=inputs, labels=labels)
            total_nll += float(outputs.loss.item()) * scored_tokens
            total_tokens += scored_tokens

    if total_tokens == 0:
        raise ValueError(f"No completion tokens were scored by {model_name}")
    mean_nll = total_nll / total_tokens
    if not math.isfinite(mean_nll):
        raise ValueError(f"Model {model_name} returned a non-finite mean NLL")
    perplexity = (
        None
        if mean_nll > math.log(sys.float_info.max)
        else math.exp(mean_nll)
    )
    del model
    if resolved_device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "model": model_name,
        "revision": model_revision,
        "completion_field": completion_field,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "tokens": total_tokens,
        "skipped_empty": 0,
        "tokenizer_fingerprint": tokenizer_hash,
        "device": resolved_device,
        "protocol": {
            "tokenization": "Tokenize prompt+completion once with a fast tokenizer.",
            "boundary": "Reject a prompt/completion split that cuts through one tokenizer token.",
            "special_tokens": "Only an available BOS token is prepended; no automatic EOS is added.",
            "empty_policy": "Reject empty or unscorable completions instead of dropping them.",
        },
    }


def cross_ppl_report(args: argparse.Namespace) -> dict[str, Any]:
    pairs = text_pairs(args.inputs, require_prompt=True)
    student_under_teacher = score_completions(
        args.teacher_model,
        args.teacher_revision,
        pairs,
        "student_text",
        args.device,
        args.max_length,
        args.trust_remote_code,
    )
    teacher_under_student = score_completions(
        args.student_model,
        args.student_revision,
        pairs,
        "teacher_text",
        args.device,
        args.max_length,
        args.trust_remote_code,
    )
    symmetric, symmetric_status = symmetric_cross_ppl(
        student_under_teacher, teacher_under_student
    )
    report: dict[str, Any] = {
        "metric": "cross_perplexity",
        "student_text_under_teacher": student_under_teacher,
        "teacher_text_under_student": teacher_under_student,
        "symmetric_geometric_mean": symmetric,
        "symmetric_status": symmetric_status,
        "n": len(pairs),
        "teacher_normalization": {
            "status": "unavailable",
            "reason": "Cross-PPL is lower-is-better and may use different tokenizers; the Matrix similarity-ratio rule is not mathematically defined for this metric.",
        },
        "protocol": "Prompt tokens are context only; loss is computed on completion tokens.",
    }
    base_options = (args.base_inputs, args.base_model, args.base_revision)
    if any(value is not None for value in base_options) and not all(
        value is not None for value in base_options
    ):
        raise ValueError(
            "--base-inputs, --base-model, and --base-revision must be provided together"
        )
    if args.base_inputs:
        base_pairs = text_pairs(args.base_inputs, require_prompt=True)
        require_aligned_text_pairs(pairs, base_pairs, "bare-base")
        base_under_teacher = score_completions(
            args.teacher_model,
            args.teacher_revision,
            base_pairs,
            "student_text",
            args.device,
            args.max_length,
            args.trust_remote_code,
        )
        teacher_under_base = score_completions(
            args.base_model,
            args.base_revision,
            base_pairs,
            "teacher_text",
            args.device,
            args.max_length,
            args.trust_remote_code,
        )
        base_symmetric, base_symmetric_status = symmetric_cross_ppl(
            base_under_teacher, teacher_under_base
        )
        report["bare_base_floor"] = {
            "base_text_under_teacher": base_under_teacher,
            "teacher_text_under_base": teacher_under_base,
            "symmetric_geometric_mean": base_symmetric,
            "symmetric_status": base_symmetric_status,
            "n": len(base_pairs),
        }
    return report


def command_agreement(args: argparse.Namespace) -> int:
    report = agreement_report(
        args.teacher_dir,
        args.student_dir,
        allow_partial=args.allow_partial,
    )
    if args.base_dir:
        base = agreement_report(
            args.teacher_dir,
            args.base_dir,
            allow_partial=args.allow_partial,
        )
        report["bare_base_floor"] = base
        report["gain_over_bare_base"] = report["agreement_rate"] - base["agreement_rate"]
    write_report(report, args.output)
    return 0


def command_text(args: argparse.Namespace) -> int:
    write_report(text_report(args), args.output)
    return 0


def command_cross_ppl(args: argparse.Namespace) -> int:
    write_report(cross_ppl_report(args), args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Matrix M2 teacher-student fidelity metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    agreement = subparsers.add_parser("agreement", help="Compare lm-eval MCQ predictions.")
    agreement.add_argument("--teacher-dir", required=True, help="Teacher lm-eval result directory or sample JSONL.")
    agreement.add_argument("--student-dir", required=True, help="Student lm-eval result directory or sample JSONL.")
    agreement.add_argument("--base-dir", help="Optional bare-base lm-eval result directory for the Stage-1 floor.")
    agreement.add_argument(
        "--allow-partial",
        action="store_true",
        help="Diagnostic only: compare the sample-key intersection instead of requiring exact coverage.",
    )
    agreement.add_argument("--output", help="Write JSON report to this path.")
    agreement.set_defaults(func=command_agreement)

    text = subparsers.add_parser("text", help="Compute teacher-student text similarity.")
    text.add_argument("inputs", nargs="+", help="Paired teacher/student JSONL files.")
    text.add_argument(
        "--base-inputs",
        nargs="+",
        help="Optional aligned teacher/bare-base pairs; student_text holds the bare-base output.",
    )
    text.add_argument("--bertscore", action="store_true", help="Also compute rescaled BERTScore-F1.")
    text.add_argument("--mauve", action="store_true", help="Also compute MAUVE distribution similarity.")
    text.add_argument("--lang", default="en", help="BERTScore language code. Default: en.")
    text.add_argument("--bert-model", help="Optional BERTScore encoder name.")
    text.add_argument("--batch-size", type=int, default=16, help="BERTScore batch size.")
    text.add_argument("--device", help="BERTScore device, such as cuda:0 or cpu.")
    text.add_argument("--mauve-device", type=int, default=-1, help="MAUVE GPU id; -1 uses CPU.")
    text.add_argument("--max-text-length", type=int, default=512, help="MAUVE maximum text length.")
    text.add_argument("--output", help="Write JSON report to this path.")
    text.set_defaults(func=command_text)

    cross_ppl = subparsers.add_parser("cross-ppl", help="Compute bidirectional cross-perplexity.")
    cross_ppl.add_argument("inputs", nargs="+", help="Paired teacher/student JSONL files.")
    cross_ppl.add_argument("--teacher-model", required=True, help="Teacher Hugging Face model path/name.")
    cross_ppl.add_argument("--teacher-revision", required=True)
    cross_ppl.add_argument("--student-model", required=True, help="Student Hugging Face model path/name.")
    cross_ppl.add_argument("--student-revision", required=True)
    cross_ppl.add_argument(
        "--base-inputs",
        nargs="+",
        help="Optional aligned teacher/bare-base pairs; student_text holds the bare-base output.",
    )
    cross_ppl.add_argument("--base-model", help="Bare-base model used with --base-inputs.")
    cross_ppl.add_argument("--base-revision", help="Pinned bare-base model revision.")
    cross_ppl.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    cross_ppl.add_argument("--max-length", type=int, default=4096, help="Reject samples above this token length.")
    cross_ppl.add_argument("--trust-remote-code", action="store_true")
    cross_ppl.add_argument("--output", help="Write JSON report to this path.")
    cross_ppl.set_defaults(func=command_cross_ppl)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
