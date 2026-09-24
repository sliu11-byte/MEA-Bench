#!/usr/bin/env python3
"""M3 output-quality metrics.

Commands:
  repetition  Compute macro rep-n for n=2/3/4 from JSONL text rows.
  ppl         Score continuation-only loss with one fixed Hugging Face causal-LM oracle.
  judge       Aggregate precomputed forward/reverse pairwise judgments.

Repetition input rows use this schema:
  {"id": "1", "text": "model output"}

PPL input rows separate context from the scored continuation:
  {"id": "1", "context": "C4 prefix", "continuation": "model continuation"}

Judge input rows use this schema:
  {
    "id": "1",
    "candidate_text": "student output",
    "reference_text": "comparison output",
    "forward": {"candidate_position": "A", "winner": "A"},
    "reverse": {"candidate_position": "B", "winner": "B"}
  }

The judge command only aggregates supplied decisions. It never calls an API.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


TOKENIZERS = ("unicode_cjk", "whitespace")
SHORT_POLICIES = ("zero", "exclude")
WIN_OUTCOMES = ("candidate", "reference", "tie")

# CJK characters are separate tokens. Other Unicode letters/digits form words,
# and remaining non-whitespace characters are individual punctuation tokens.
UNICODE_CJK_TOKEN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]"
    r"|[^\W_]+(?:['’\-][^\W_]+)*"
    r"|[^\s]",
    re.UNICODE,
)


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
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def text_rows(paths: Iterable[str], text_field: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(iter_jsonl(paths), start=1):
        row_id = str(row.get("id", index))
        if row_id in seen_ids:
            raise ValueError(f"Duplicate text id: {row_id}")
        seen_ids.add(row_id)
        text = row.get(text_field)
        if not isinstance(text, str):
            raise ValueError(f"Text row {row_id} is missing string field {text_field!r}")
        rows.append({"id": row_id, "text": text})
    if not rows:
        raise ValueError("No text rows found")
    return rows


def continuation_rows(
    paths: Iterable[str],
    context_field: str,
    continuation_field: str,
) -> list[dict[str, str]]:
    if not context_field or not continuation_field or context_field == continuation_field:
        raise ValueError("context and continuation field names must be non-empty and distinct")
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(iter_jsonl(paths), start=1):
        row_id = str(row.get("id", index))
        if row_id in seen_ids:
            raise ValueError(f"Duplicate PPL id: {row_id}")
        seen_ids.add(row_id)
        for field in (context_field, continuation_field):
            if not isinstance(row.get(field), str):
                raise ValueError(f"PPL row {row_id} is missing string field {field!r}")
        rows.append(
            {
                "id": row_id,
                "context": row[context_field],
                "continuation": row[continuation_field],
            }
        )
    if not rows:
        raise ValueError("No PPL continuation rows found")
    return rows


def tokenize(text: str, tokenizer: str = "unicode_cjk") -> list[str]:
    if tokenizer == "unicode_cjk":
        return [token.casefold() for token in UNICODE_CJK_TOKEN.findall(str(text))]
    if tokenizer == "whitespace":
        return str(text).casefold().split()
    raise ValueError(f"Unknown tokenizer {tokenizer!r}; choose from {TOKENIZERS}")


def rep_n_from_tokens(tokens: Sequence[str], n: int) -> float | None:
    """Return 1 - unique n-grams / total n-grams, or None when too short."""
    if n <= 0:
        raise ValueError("n must be positive")
    total = len(tokens) - n + 1
    if total <= 0:
        return None
    ngrams = [tuple(tokens[index : index + n]) for index in range(total)]
    return 1.0 - len(set(ngrams)) / total


def repetition_report(
    texts: Sequence[str],
    *,
    ns: Sequence[int] = (2, 3, 4),
    tokenizer: str,
    short_policy: str,
) -> dict[str, Any]:
    if short_policy not in SHORT_POLICIES:
        raise ValueError(f"short_policy must be one of {SHORT_POLICIES}")
    normalized_ns = tuple(dict.fromkeys(int(n) for n in ns))
    if not normalized_ns or any(n <= 0 for n in normalized_ns):
        raise ValueError("ns must contain at least one positive integer")

    tokenized = [tokenize(text, tokenizer) for text in texts]
    sample_count = len(texts)
    metrics: dict[str, Any] = {}
    for n in normalized_ns:
        raw_scores = [rep_n_from_tokens(tokens, n) for tokens in tokenized]
        short_count = sum(score is None for score in raw_scores)
        if short_policy == "zero":
            included_scores = [0.0 if score is None else score for score in raw_scores]
        else:
            included_scores = [score for score in raw_scores if score is not None]
        metrics[f"rep_{n}"] = {
            "macro_average": (
                sum(included_scores) / len(included_scores) if included_scores else None
            ),
            "included_count": len(included_scores),
            "short_count": short_count,
            "short_rate": short_count / sample_count if sample_count else 0.0,
        }

    lengths = [len(tokens) for tokens in tokenized]
    empty_count = sum(length == 0 for length in lengths)
    return {
        "metric": "rep_n",
        "sample_count": sample_count,
        "empty_count": empty_count,
        "empty_rate": empty_count / sample_count if sample_count else 0.0,
        "token_length": {
            "mean": sum(lengths) / sample_count if sample_count else 0.0,
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
        "metrics": metrics,
        "protocol": {
            "formula": "1 - unique_n_grams / total_n_grams, computed per text then macro-averaged",
            "n_values": list(normalized_ns),
            "tokenizer": tokenizer,
            "short_policy": short_policy,
            "short_definition": "token_count < n",
            "empty_outputs_are_short": True,
        },
    }


def summarize_ppl_against_human_interval(
    model_ppl: float,
    human_low: float,
    human_high: float,
) -> dict[str, Any]:
    values = (float(model_ppl), float(human_low), float(human_high))
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("model_ppl and human interval bounds must be finite and positive")
    model_ppl, human_low, human_high = values
    if human_low > human_high:
        raise ValueError("human_low must be <= human_high")

    if model_ppl < human_low:
        position = "below"
        distance = human_low - model_ppl
        boundary = human_low
    elif model_ppl > human_high:
        position = "above"
        distance = model_ppl - human_high
        boundary = human_high
    else:
        position = "inside"
        distance = 0.0
        boundary = model_ppl
    return {
        "model_ppl": model_ppl,
        "human_interval": {"low": human_low, "high": human_high},
        "inside_human_interval": position == "inside",
        "position": position,
        "distance_to_human_interval": distance,
        "relative_distance_to_nearest_boundary": distance / boundary if distance else 0.0,
        "protocol": "Distance is zero inside the closed human interval; otherwise it is the gap to the nearest boundary.",
    }


def import_hf_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PPL scoring requires optional dependencies: torch and transformers"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _finite_context_limit(model: Any, tokenizer: Any, requested: int) -> int:
    if requested <= 0:
        raise ValueError("max_length must be positive")
    candidates = [requested]
    for value in (
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)
    return min(candidates)


def _safe_perplexity(mean_nll: float) -> float | None:
    if not math.isfinite(mean_nll):
        raise ValueError("Oracle returned a non-finite loss")
    if mean_nll > math.log(sys.float_info.max):
        return None
    return math.exp(mean_nll)


def joint_tokenization(
    tokenizer: Any,
    context: str,
    continuation: str,
    row_id: str,
) -> tuple[list[int], list[bool]]:
    if not continuation:
        raise ValueError(f"Text {row_id} has an empty continuation")
    encoded = tokenizer(
        context + continuation,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if not isinstance(input_ids, list) or not isinstance(offsets, list):
        raise ValueError("Oracle tokenizer must provide offset_mapping")
    if len(input_ids) != len(offsets):
        raise ValueError(f"Oracle tokenizer returned misaligned offsets for {row_id}")
    boundary = len(context)
    score_mask = []
    for offset in offsets:
        if (
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or not all(isinstance(value, int) for value in offset)
        ):
            raise ValueError(f"Oracle tokenizer returned invalid offsets for {row_id}")
        start, end = offset
        if start < boundary < end:
            raise ValueError(
                f"Text {row_id} continuation boundary splits an oracle token; "
                "prepare the C4 split on oracle token boundaries"
            )
        score_mask.append(start >= boundary and end > start)
    if not any(score_mask):
        raise ValueError(f"Text {row_id} has no scoreable continuation tokens")
    return [int(token_id) for token_id in input_ids], score_mask


def score_ppl_hf(
    rows: Sequence[dict[str, str]],
    *,
    oracle_model: str,
    oracle_revision: str,
    device: str = "auto",
    max_length: int = 4096,
    bos_policy: str = "auto",
    trust_remote_code: bool = False,
    include_per_text: bool = False,
    expected_segments: int | None = None,
    expected_continuation_tokens: int | None = None,
) -> dict[str, Any]:
    """Score continuation tokens conditioned on context. No truncation is performed."""
    if bos_policy not in {"auto", "require", "none"}:
        raise ValueError("bos_policy must be auto, require, or none")
    if not oracle_model:
        raise ValueError("oracle_model must be non-empty")
    if not oracle_revision:
        raise ValueError("oracle_revision must be pinned explicitly")
    if expected_segments is not None:
        if expected_segments <= 0:
            raise ValueError("expected_segments must be positive")
        if len(rows) != expected_segments:
            raise ValueError(
                f"Expected {expected_segments} PPL segments, found {len(rows)}"
            )
    if expected_continuation_tokens is not None and expected_continuation_tokens <= 0:
        raise ValueError("expected_continuation_tokens must be positive")

    torch, auto_model, auto_tokenizer = import_hf_dependencies()
    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if oracle_revision:
        load_kwargs["revision"] = oracle_revision
    tokenizer = auto_tokenizer.from_pretrained(
        oracle_model,
        use_fast=True,
        **load_kwargs,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("PPL scoring requires a fast tokenizer with offset_mapping")
    model = auto_model.from_pretrained(
        oracle_model,
        torch_dtype="auto",
        **load_kwargs,
    ).to(resolved_device)
    model.eval()

    effective_limit = _finite_context_limit(model, tokenizer, max_length)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_policy == "require" and bos_token_id is None:
        raise ValueError("bos_policy=require but the oracle tokenizer has no BOS token")
    use_bos = bos_token_id is not None and bos_policy in {"auto", "require"}

    total_nll = 0.0
    total_tokens = 0
    per_text: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row in rows:
            joint_ids, continuation_mask = joint_tokenization(
                tokenizer,
                row["context"],
                row["continuation"],
                row["id"],
            )
            continuation_token_count = sum(continuation_mask)
            if (
                expected_continuation_tokens is not None
                and continuation_token_count != expected_continuation_tokens
            ):
                raise ValueError(
                    f"Text {row['id']} has {continuation_token_count} continuation tokens; "
                    f"expected {expected_continuation_tokens}"
                )

            bos_ids = [int(bos_token_id)] if use_bos else []
            input_ids = bos_ids + joint_ids
            if len(input_ids) > effective_limit:
                raise ValueError(
                    f"Text {row['id']} has {len(input_ids)} oracle tokens, above the effective limit "
                    f"{effective_limit}; truncation is forbidden"
                )

            inputs = torch.tensor([input_ids], dtype=torch.long, device=resolved_device)
            label_ids = [-100] * len(input_ids)
            for index, should_score in enumerate(continuation_mask):
                if should_score:
                    label_ids[len(bos_ids) + index] = joint_ids[index]
            scored_tokens = sum(label != -100 for label in label_ids[1:])
            if scored_tokens == 0:
                raise ValueError(
                    f"Text {row['id']} has no causally scoreable continuation tokens; "
                    "use a tokenizer BOS token or a non-empty context"
                )
            labels = torch.tensor(
                [label_ids], dtype=torch.long, device=resolved_device
            )

            outputs = model(input_ids=inputs, labels=labels)
            mean_nll = float(outputs.loss.item())
            total_nll += mean_nll * scored_tokens
            total_tokens += scored_tokens
            per_text.append(
                {
                    "id": row["id"],
                    "mean_nll": mean_nll,
                    "perplexity": _safe_perplexity(mean_nll),
                    "scored_tokens": scored_tokens,
                    "joint_tokens": len(joint_ids),
                    "continuation_tokens": continuation_token_count,
                }
            )

    if total_tokens == 0:
        raise ValueError("No text tokens were scored")
    corpus_mean_nll = total_nll / total_tokens
    finite_per_text_ppl = [
        row["perplexity"] for row in per_text if row["perplexity"] is not None
    ]
    report: dict[str, Any] = {
        "metric": "oracle_perplexity",
        "corpus_mean_nll": corpus_mean_nll,
        "corpus_perplexity": _safe_perplexity(corpus_mean_nll),
        "macro_mean_perplexity": (
            sum(finite_per_text_ppl) / len(finite_per_text_ppl)
            if finite_per_text_ppl else None
        ),
        "sample_count": len(rows),
        "scored_text_count": len(per_text),
        "scored_token_count": total_tokens,
        "skipped_empty_count": 0,
        "skipped_unscorable_count": 0,
        "oracle": {
            "model": oracle_model,
            "revision": oracle_revision,
            "device": resolved_device,
        },
        "protocol": {
            "input": "separate context and continuation fields",
            "loss_scope": "Tokenize context+continuation once; score tokens starting at or after the continuation boundary.",
            "boundary_policy": "Reject context/continuation splits that cut through an oracle token.",
            "empty_policy": "Reject the report instead of dropping empty or unscorable continuations.",
            "add_special_tokens": False,
            "bos_policy": bos_policy,
            "bos_used_as_unscored_context": use_bos,
            "aggregation": "corpus_perplexity is token-weighted; macro_mean_perplexity averages per-text PPL",
            "requested_max_length": max_length,
            "effective_max_length": effective_limit,
            "truncation": "forbidden; overlength input raises ValueError",
            "expected_segments": expected_segments,
            "expected_continuation_tokens": expected_continuation_tokens,
        },
    }
    if include_per_text:
        report["per_text"] = per_text

    del model
    if resolved_device.startswith("cuda"):
        torch.cuda.empty_cache()
    return report


def _normalize_position(value: Any, *, row_id: str, direction: str) -> str:
    position = str(value).strip().upper()
    if position not in {"A", "B"}:
        raise ValueError(f"Judge row {row_id} {direction}.candidate_position must be A or B")
    return position


def _semantic_judge_outcome(
    decision: Any,
    *,
    row_id: str,
    direction: str,
) -> tuple[str, str, str]:
    if not isinstance(decision, dict):
        raise ValueError(f"Judge row {row_id} is missing object field {direction!r}")
    if "candidate_position" not in decision or "winner" not in decision:
        raise ValueError(
            f"Judge row {row_id} {direction} must contain candidate_position and winner"
        )
    position = _normalize_position(
        decision["candidate_position"], row_id=row_id, direction=direction
    )
    winner_value = str(decision["winner"]).strip()
    if winner_value.casefold() == "tie":
        return "tie", position, "tie"
    winner = winner_value.upper()
    if winner not in {"A", "B"}:
        raise ValueError(f"Judge row {row_id} {direction}.winner must be A, B, or tie")
    outcome = "candidate" if winner == position else "reference"
    return outcome, position, winner


def _row_output_length(
    row: dict[str, Any],
    prefix: str,
    *,
    row_id: str,
    tokenizer: str,
) -> tuple[int, str]:
    length_field = f"{prefix}_length"
    text_field = f"{prefix}_text"
    if length_field in row:
        value = row[length_field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Judge row {row_id} field {length_field!r} must be a non-negative integer")
        return value, "explicit_length"
    text = row.get(text_field)
    if not isinstance(text, str):
        raise ValueError(
            f"Judge row {row_id} needs {text_field!r} or non-negative {length_field!r}"
        )
    return len(tokenize(text, tokenizer)), "tokenized_text"


def _length_group(
    candidate_length: int,
    reference_length: int,
    lower: float,
    upper: float,
) -> tuple[str, float | None]:
    if lower <= 0 or upper < lower:
        raise ValueError("length ratio bounds must satisfy 0 < lower <= upper")
    if reference_length == 0:
        if candidate_length == 0:
            return "similar", 1.0
        return "longer", None
    ratio = candidate_length / reference_length
    if ratio < lower:
        return "shorter", ratio
    if ratio > upper:
        return "longer", ratio
    return "similar", ratio


def _outcome_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    outcomes = Counter(record["outcome"] for record in records)
    consistent = sum(record["position_consistent"] for record in records)
    non_ties = outcomes["candidate"] + outcomes["reference"]
    return {
        "count": count,
        "candidate_wins": outcomes["candidate"],
        "reference_wins": outcomes["reference"],
        "ties": outcomes["tie"],
        "candidate_win_rate": outcomes["candidate"] / count if count else None,
        "candidate_tie_adjusted_win_rate": (
            (outcomes["candidate"] + 0.5 * outcomes["tie"]) / count
            if count else None
        ),
        "reference_win_rate": outcomes["reference"] / count if count else None,
        "tie_rate": outcomes["tie"] / count if count else None,
        "candidate_non_tie_win_rate": (
            outcomes["candidate"] / non_ties if non_ties else None
        ),
        "position_consistent_count": consistent,
        "position_consistency_rate": consistent / count if count else None,
    }


def aggregate_pairwise_judgments(
    rows: Sequence[dict[str, Any]],
    *,
    judge_model: str,
    judge_revision: str,
    rubric_id: str,
    length_tokenizer: str,
    length_ratio_lower: float,
    length_ratio_upper: float,
    include_decisions: bool = False,
) -> dict[str, Any]:
    for field, value in (
        ("judge_model", judge_model),
        ("judge_revision", judge_revision),
        ("rubric_id", rubric_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be non-empty text")
    decisions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    length_sources: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Judge row {index} must be an object")
        row_id = str(row.get("id", index))
        if row_id in seen_ids:
            raise ValueError(f"Duplicate judge id: {row_id}")
        seen_ids.add(row_id)

        forward_outcome, forward_position, forward_winner = _semantic_judge_outcome(
            row.get("forward"), row_id=row_id, direction="forward"
        )
        reverse_outcome, reverse_position, reverse_winner = _semantic_judge_outcome(
            row.get("reverse"), row_id=row_id, direction="reverse"
        )
        if forward_position == reverse_position:
            raise ValueError(
                f"Judge row {row_id} must swap candidate_position between forward and reverse"
            )

        position_consistent = forward_outcome == reverse_outcome
        outcome = forward_outcome if position_consistent else "tie"
        candidate_length, candidate_source = _row_output_length(
            row, "candidate", row_id=row_id, tokenizer=length_tokenizer
        )
        reference_length, reference_source = _row_output_length(
            row, "reference", row_id=row_id, tokenizer=length_tokenizer
        )
        length_sources[candidate_source] += 1
        length_sources[reference_source] += 1
        length_group, length_ratio = _length_group(
            candidate_length,
            reference_length,
            length_ratio_lower,
            length_ratio_upper,
        )
        decisions.append(
            {
                "id": row_id,
                "outcome": outcome,
                "forward_outcome": forward_outcome,
                "reverse_outcome": reverse_outcome,
                "forward_candidate_position": forward_position,
                "reverse_candidate_position": reverse_position,
                "forward_winner": forward_winner,
                "reverse_winner": reverse_winner,
                "position_consistent": position_consistent,
                "candidate_length": candidate_length,
                "reference_length": reference_length,
                "length_ratio": length_ratio,
                "length_group": length_group,
            }
        )

    if not decisions:
        raise ValueError("No judge rows found")
    by_length_group = {
        group: _outcome_summary(
            [decision for decision in decisions if decision["length_group"] == group]
        )
        for group in ("shorter", "similar", "longer")
    }
    nonempty_length_scores = [
        summary["candidate_tie_adjusted_win_rate"]
        for summary in by_length_group.values()
        if summary["count"] > 0
    ]
    report: dict[str, Any] = {
        "metric": "pairwise_judge_win_rate",
        "overall": _outcome_summary(decisions),
        "position_disagreement_tie_count": sum(
            not decision["position_consistent"] for decision in decisions
        ),
        "explicit_tie_count": sum(
            decision["position_consistent"] and decision["outcome"] == "tie"
            for decision in decisions
        ),
        "by_length_group": by_length_group,
        "length_controlled_candidate_win_rate": (
            sum(nonempty_length_scores) / len(nonempty_length_scores)
        ),
        "protocol": {
            "api_calls": False,
            "judge_model": judge_model,
            "judge_revision": judge_revision,
            "rubric_id": rubric_id,
            "required_decisions": ["forward", "reverse"],
            "decision_fields": ["candidate_position", "winner"],
            "winner_values": ["A", "B", "tie"],
            "position_swap": "forward and reverse candidate_position must be opposite",
            "inconsistent_position_swap": "final outcome is tie",
            "tie_adjusted_win_rate": "(candidate wins + 0.5 * ties) / pair count",
            "length_metric": "candidate_length / reference_length",
            "length_tokenizer": length_tokenizer,
            "length_ratio_lower": length_ratio_lower,
            "length_ratio_upper": length_ratio_upper,
            "length_groups": {
                "shorter": "ratio < lower",
                "similar": "lower <= ratio <= upper",
                "longer": "ratio > upper",
            },
            "length_controlled_win_rate": "Unweighted macro-average of tie-adjusted candidate win rates across non-empty shorter/similar/longer strata.",
            "length_sources": dict(sorted(length_sources.items())),
            "explicit_length_precedence": "candidate_length/reference_length override text token counts",
        },
    }
    if include_decisions:
        report["decisions"] = decisions
    return report


def command_repetition(args: argparse.Namespace) -> int:
    rows = text_rows(args.inputs, args.text_field)
    report = repetition_report(
        [row["text"] for row in rows],
        ns=args.n,
        tokenizer=args.tokenizer,
        short_policy=args.short_policy,
    )
    report["input"] = {"text_field": args.text_field}
    write_report(report, args.output)
    return 0


def command_ppl(args: argparse.Namespace) -> int:
    if (args.human_ppl_low is None) != (args.human_ppl_high is None):
        raise ValueError("--human-ppl-low and --human-ppl-high must be provided together")
    if args.human_ppl_low is not None and not args.human_interval_source:
        raise ValueError("--human-interval-source is required with a human PPL interval")
    rows = continuation_rows(
        args.inputs,
        args.context_field,
        args.continuation_field,
    )
    report = score_ppl_hf(
        rows,
        oracle_model=args.oracle_model,
        oracle_revision=args.oracle_revision,
        device=args.device,
        max_length=args.max_length,
        bos_policy=args.bos_policy,
        trust_remote_code=args.trust_remote_code,
        include_per_text=args.include_per_text,
        expected_segments=args.expected_segments,
        expected_continuation_tokens=args.expected_continuation_tokens,
    )
    report["input"] = {
        "context_field": args.context_field,
        "continuation_field": args.continuation_field,
    }
    if args.human_ppl_low is not None:
        field = f"{args.human_comparison_aggregation}_perplexity"
        model_ppl = report[field]
        if model_ppl is None:
            raise ValueError(f"Cannot compare human interval because {field} overflowed")
        report["human_comparison"] = summarize_ppl_against_human_interval(
            model_ppl,
            args.human_ppl_low,
            args.human_ppl_high,
        )
        report["human_comparison"]["model_ppl_source"] = field
        report["human_comparison"]["source"] = args.human_interval_source
    else:
        report["human_comparison"] = {
            "status": "unavailable",
            "reason": "No explicit human PPL interval was supplied",
        }
    write_report(report, args.output)
    return 0


def command_judge(args: argparse.Namespace) -> int:
    rows = list(iter_jsonl(args.inputs))
    report = aggregate_pairwise_judgments(
        rows,
        judge_model=args.judge_model,
        judge_revision=args.judge_revision,
        rubric_id=args.rubric_id,
        length_tokenizer=args.length_tokenizer,
        length_ratio_lower=args.length_ratio_lower,
        length_ratio_upper=args.length_ratio_upper,
        include_decisions=args.include_decisions,
    )
    write_report(report, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Matrix M3 output-quality metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    repetition = subparsers.add_parser("repetition", help="Compute macro rep-n.")
    repetition.add_argument("inputs", nargs="+", help="JSONL files containing output text.")
    repetition.add_argument("--text-field", default="text", help="Text field. Default: text.")
    repetition.add_argument("--n", nargs="+", type=int, default=[2, 3, 4], help="n-gram sizes.")
    repetition.add_argument("--tokenizer", choices=TOKENIZERS, required=True)
    repetition.add_argument(
        "--short-policy",
        choices=SHORT_POLICIES,
        required=True,
        help="Treat token_count<n as zero or exclude it from that n's macro average.",
    )
    repetition.add_argument("--output", help="Write JSON report to this path.")
    repetition.set_defaults(func=command_repetition)

    ppl = subparsers.add_parser(
        "ppl", help="Compute continuation-only perplexity with a fixed HF oracle."
    )
    ppl.add_argument("inputs", nargs="+", help="JSONL files containing context + continuation.")
    ppl.add_argument("--context-field", required=True)
    ppl.add_argument("--continuation-field", required=True)
    ppl.add_argument("--oracle-model", required=True)
    ppl.add_argument("--oracle-revision", required=True, help="Pinned Hugging Face revision.")
    ppl.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    ppl.add_argument("--max-length", type=int, default=4096)
    ppl.add_argument("--bos-policy", choices=("auto", "require", "none"), default="auto")
    ppl.add_argument("--trust-remote-code", action="store_true")
    ppl.add_argument("--include-per-text", action="store_true")
    ppl.add_argument("--expected-segments", type=int)
    ppl.add_argument("--expected-continuation-tokens", type=int)
    ppl.add_argument("--human-ppl-low", type=float, help="Explicit lower human PPL bound.")
    ppl.add_argument("--human-ppl-high", type=float, help="Explicit upper human PPL bound.")
    ppl.add_argument(
        "--human-interval-source",
        help="Required provenance label/path when a human PPL interval is supplied.",
    )
    ppl.add_argument(
        "--human-comparison-aggregation",
        choices=("corpus", "macro_mean"),
        default="corpus",
        help="Which reported model PPL to compare with the human interval.",
    )
    ppl.add_argument("--output", help="Write JSON report to this path.")
    ppl.set_defaults(func=command_ppl)

    judge = subparsers.add_parser(
        "judge", help="Aggregate supplied forward/reverse pairwise judgments without API calls."
    )
    judge.add_argument("inputs", nargs="+", help="Pairwise judgment JSONL files.")
    judge.add_argument("--judge-model", required=True)
    judge.add_argument("--judge-revision", required=True)
    judge.add_argument("--rubric-id", required=True)
    judge.add_argument("--length-tokenizer", choices=TOKENIZERS, required=True)
    judge.add_argument("--length-ratio-lower", type=float, required=True)
    judge.add_argument("--length-ratio-upper", type=float, required=True)
    judge.add_argument("--include-decisions", action="store_true")
    judge.add_argument("--output", help="Write JSON report to this path.")
    judge.set_defaults(func=command_judge)
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
