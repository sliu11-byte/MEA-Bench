from __future__ import annotations

"""Shared held-out M1/M2/M3/M6 evaluation and result schema."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

from .stage1_data import normalize_prompt
from .stage1_transcript import stable_hash


RESULT_SCHEMA_VERSION = "stage1_results_v1"


class EvaluationLeakageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClassificationItem:
    prompt_id: str
    prompt_text: str
    choices: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class FidelityItem:
    prompt_id: str
    prompt_text: str


def unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def assert_held_out(eval_prompts: Sequence[str], training_prompts: Sequence[str]) -> None:
    training_keys = {normalize_prompt(prompt) for prompt in training_prompts}
    collisions = [prompt for prompt in eval_prompts if normalize_prompt(prompt) in training_keys]
    if collisions:
        raise EvaluationLeakageError(f"held-out evaluation overlaps training data ({len(collisions)} prompts).")


def _extract_choice(output: str, choices: Sequence[str]) -> str | None:
    normalized = normalize_prompt(output)
    for choice in choices:
        if re.search(rf"(?<!\w){re.escape(normalize_prompt(choice))}(?!\w)", normalized):
            return choice
    return None


def evaluate_m1_classification(
    items: Sequence[ClassificationItem],
    predictors: Mapping[str, Callable[[str], str]],
    *,
    training_prompts: Sequence[str],
) -> dict[str, Any]:
    assert_held_out([item.prompt_text for item in items], training_prompts)
    systems: dict[str, Any] = {}
    predictions: dict[str, list[str | None]] = {}
    for system_name, predictor in predictors.items():
        raw_outputs = [str(predictor(item.prompt_text)) for item in items]
        parsed = [_extract_choice(output, item.choices) for output, item in zip(raw_outputs, items)]
        predictions[system_name] = parsed
        correct = sum(prediction == item.label for prediction, item in zip(parsed, items))
        systems[system_name] = {
            "status": "ok",
            "accuracy": correct / len(items) if items else 0.0,
            "correct": correct,
            "count": len(items),
            "predictions": parsed,
            "raw_outputs": raw_outputs,
        }
    teacher_predictions = predictions.get("teacher")
    if teacher_predictions is not None:
        for system_name, parsed in predictions.items():
            systems[system_name]["teacher_agreement"] = (
                sum(
                    left is not None and right is not None and left == right
                    for left, right in zip(parsed, teacher_predictions)
                ) / len(items)
                if items else 0.0
            )
    return {
        "status": "ok",
        "protocol": "held_out_multiple_choice_exact_parse",
        "held_out": True,
        "prompt_ids": [item.prompt_id for item in items],
        "systems": systems,
        "openllm": unavailable("full OpenLLM evaluation was not run in the local engineering smoke"),
    }


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    row = [0] * (len(right) + 1)
    for left_token in left:
        previous = 0
        for index, right_token in enumerate(right, start=1):
            old = row[index]
            row[index] = previous + 1 if left_token == right_token else max(row[index], row[index - 1])
            previous = old
    return row[-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    reference_tokens = normalize_prompt(reference).split()
    candidate_tokens = normalize_prompt(candidate).split()
    if not reference_tokens or not candidate_tokens:
        return 1.0 if reference_tokens == candidate_tokens else 0.0
    lcs = _lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_m2_fidelity(
    items: Sequence[FidelityItem],
    teacher_outputs: Mapping[str, str],
    student_outputs: Mapping[str, str],
    *,
    training_prompts: Sequence[str],
    run_bertscore: bool = True,
) -> dict[str, Any]:
    assert_held_out([item.prompt_text for item in items], training_prompts)
    expected_ids = [item.prompt_id for item in items]
    if set(teacher_outputs) != set(expected_ids) or set(student_outputs) != set(expected_ids):
        raise ValueError("teacher and student outputs must exactly cover held-out fidelity prompt IDs.")
    references = [teacher_outputs[prompt_id] for prompt_id in expected_ids]
    candidates = [student_outputs[prompt_id] for prompt_id in expected_ids]
    agreement = sum(normalize_prompt(left) == normalize_prompt(right) for left, right in zip(references, candidates))
    rouge_scores = [rouge_l_f1(reference, candidate) for reference, candidate in zip(references, candidates)]
    bertscore: dict[str, Any]
    if run_bertscore:
        try:
            from bert_score import score as bert_score

            _, _, f1 = bert_score(candidates, references, lang="en", verbose=False)
            bertscore = {"status": "ok", "f1": float(f1.mean().item())}
        except Exception as error:
            bertscore = unavailable(f"BERTScore dependency/model unavailable: {type(error).__name__}: {error}")
    else:
        bertscore = unavailable("BERTScore disabled for this run")
    return {
        "status": "ok",
        "protocol": "held_out_text_fidelity",
        "held_out": True,
        "prompt_ids": expected_ids,
        "agreement_rate": agreement / len(items) if items else 0.0,
        "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "bertscore": bertscore,
        "mauve": unavailable("MAUVE dependency/model unavailable in the local engineering smoke"),
    }


def _rep_n(text: str, n: int) -> float:
    tokens = normalize_prompt(text).split()
    ngrams = [tuple(tokens[index:index + n]) for index in range(max(0, len(tokens) - n + 1))]
    return 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def evaluate_m3_quality(outputs: Sequence[str], *, degeneration_threshold: float = 0.2) -> dict[str, Any]:
    token_lengths = [len(normalize_prompt(output).split()) for output in outputs]
    repetitions = {n: [_rep_n(output, n) for output in outputs] for n in (2, 3, 4)}
    degenerate = [max(repetitions[n][index] for n in (2, 3, 4)) > degeneration_threshold for index in range(len(outputs))]
    count = len(outputs)
    return {
        "status": "ok",
        "rep_2": sum(repetitions[2]) / count if count else 0.0,
        "rep_3": sum(repetitions[3]) / count if count else 0.0,
        "rep_4": sum(repetitions[4]) / count if count else 0.0,
        "average_output_length": sum(token_lengths) / count if count else 0.0,
        "empty_output_ratio": sum(length == 0 for length in token_lengths) / count if count else 0.0,
        "repetitive_degeneration_ratio": sum(degenerate) / count if count else 0.0,
        "neutral_ppl_oracle": unavailable("neutral PPL oracle is not configured"),
    }


def evaluate_m6_cost(
    *,
    teacher_successful_queries: int,
    teacher_query_attempts: int,
    student_generation_tokens: int,
    wall_clock_seconds: float,
    cpu_time_seconds: float | None = None,
    checkpoint_path: str | Path,
    optimizer_steps: int,
    lord_candidate_generation_count: int = 0,
    lord_period_count: int = 0,
    lord_sub_stage_count: int = 0,
    peak_memory_bytes: int | None = None,
    gpu_hours: float = 0.0,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_size = sum(path.stat().st_size for path in checkpoint_path.rglob("*") if path.is_file())
    return {
        "status": "ok",
        "teacher_successful_queries": int(teacher_successful_queries),
        "teacher_query_attempts": int(teacher_query_attempts),
        "student_generation_tokens": int(student_generation_tokens),
        "wall_clock_seconds": float(wall_clock_seconds),
        "cpu_hours": float(cpu_time_seconds if cpu_time_seconds is not None else wall_clock_seconds) / 3600.0,
        "gpu_hours": float(gpu_hours),
        "peak_memory_bytes": peak_memory_bytes if peak_memory_bytes is not None else unavailable("peak memory was not sampled"),
        "checkpoint_size_bytes": checkpoint_size,
        "optimizer_steps": int(optimizer_steps),
        "lord_candidate_generation_count": int(lord_candidate_generation_count),
        "lord_period_count": int(lord_period_count),
        "lord_sub_stage_count": int(lord_sub_stage_count),
    }


def build_unified_result(
    *,
    attack: str,
    transcript_hash: str,
    ordering_hash: str,
    checkpoint: str | Path,
    config_source: str | Path,
    training_decode_config: Mapping[str, Any],
    evaluation_decode_config: Mapping[str, Any],
    m1: Mapping[str, Any],
    m2: Mapping[str, Any],
    m3: Mapping[str, Any],
    m6: Mapping[str, Any],
) -> dict[str, Any]:
    if attack not in {"seqkd", "lord"}:
        raise ValueError("attack must be 'seqkd' or 'lord'.")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "attack": attack,
        "transcript_hash": transcript_hash,
        "ordering_hash": ordering_hash,
        "checkpoint": str(checkpoint),
        "config_source": str(config_source),
        "training_decode_config": dict(training_decode_config),
        "evaluation_decode_config": dict(evaluation_decode_config),
        "evaluation_decode_config_hash": stable_hash(dict(evaluation_decode_config)),
        "held_out_evaluation": bool(m1.get("held_out") and m2.get("held_out")),
        "metrics": {"M1": dict(m1), "M2": dict(m2), "M3": dict(m3), "M6": dict(m6)},
    }


def save_unified_result(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return path


__all__ = [
    "ClassificationItem", "EvaluationLeakageError", "FidelityItem", "RESULT_SCHEMA_VERSION",
    "assert_held_out", "build_unified_result", "evaluate_m1_classification", "evaluate_m2_fidelity",
    "evaluate_m3_quality", "evaluate_m6_cost", "rouge_l_f1", "save_unified_result", "unavailable",
]
