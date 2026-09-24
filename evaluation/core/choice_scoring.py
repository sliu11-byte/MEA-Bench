"""Batched conditional-likelihood scoring for M1 multiple-choice tasks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any


SCORING_METHOD = "conditional_loglikelihood_v2"


def with_leading_space(values: Sequence[str]) -> list[str]:
    return [value if value[:1].isspace() else f" {value}" for value in values]


def choice_spec(example: Any) -> dict[str, Any]:
    spec = (getattr(example, "extra", None) or {}).get("choice_scoring")
    if not isinstance(spec, dict) or spec.get("method") != SCORING_METHOD:
        raise ValueError(f"Example {getattr(example, 'example_id', '?')} has no {SCORING_METHOD} specification")
    labels = spec.get("labels")
    continuations = spec.get("continuations")
    if not labels or len(labels) != len(continuations) or len(set(labels)) != len(labels):
        raise ValueError(f"Invalid candidate labels/continuations for {getattr(example, 'example_id', '?')}")
    if spec.get("length_normalize") not in (True, False):
        raise ValueError("choice_scoring.length_normalize must be boolean")
    if example.normalized_gold_answer not in labels:
        raise ValueError(f"Gold answer is outside candidate labels for {getattr(example, 'example_id', '?')}")
    return spec


def _render_context(tokenizer: Any, prompt: str, chat: bool) -> tuple[str, bool]:
    if chat:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        ), False
    return prompt, True


def encode_pair(tokenizer: Any, context: str, continuation: str, add_special_tokens: bool) -> tuple[list[int], int]:
    """Encode a context/continuation pair without losing whitespace at the token boundary."""
    if not continuation:
        raise ValueError("Choice continuation cannot be empty")
    trailing = context[len(context.rstrip()):]
    context = context.rstrip()
    continuation = trailing + continuation
    context_ids = tokenizer.encode(context, add_special_tokens=add_special_tokens)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=add_special_tokens)
    if not context_ids or len(full_ids) <= len(context_ids) or full_ids[:len(context_ids)] != context_ids:
        raise ValueError("Tokenizer did not preserve the context prefix while encoding a choice")
    return full_ids, len(context_ids)


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _score_encoded(model: Any, tokenizer: Any, encoded: Sequence[tuple[list[int], int]], size: int) -> list[dict[str, float | int]]:
    import torch

    if size < 1:
        raise ValueError("Choice scoring batch size must be positive")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Choice scoring requires a pad or EOS token")
    results: list[dict[str, float | int]] = []
    device = _model_device(model)
    for start in range(0, len(encoded), size):
        batch = encoded[start:start + size]
        width = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        for row, (ids, _) in enumerate(batch):
            input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[row, :len(ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            for row, (ids, context_length) in enumerate(batch):
                token_ids = input_ids[row, context_length:len(ids)]
                token_logits = logits[row, context_length - 1:len(ids) - 1].float()
                token_log_probs = torch.log_softmax(token_logits, dim=-1)
                selected = token_log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
                total = selected.sum().item()
                count = len(ids) - context_length
                results.append({"loglikelihood": total, "avg_loglikelihood": total / count,
                                "token_count": count})
                del token_ids, token_logits, token_log_probs, selected
        del logits, input_ids, attention_mask
    return results


def score_examples(model: Any, tokenizer: Any, examples: Sequence[Any], chat: bool, size: int) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Score all candidates and yield one deterministic result row per example."""
    ordered = sorted(examples, key=lambda example: len(choice_spec(example)["prompt"]))
    for offset in range(0, len(ordered), size):
        group = ordered[offset:offset + size]
        encoded: list[tuple[list[int], int]] = []
        specs = []
        counts = []
        for example in group:
            spec = choice_spec(example)
            context, add_special_tokens = _render_context(tokenizer, spec["prompt"], chat)
            candidates = [encode_pair(tokenizer, context, continuation, add_special_tokens)
                          for continuation in spec["continuations"]]
            encoded.extend(candidates)
            specs.append(spec)
            counts.append(len(candidates))
        scored = _score_encoded(model, tokenizer, encoded, size)
        cursor = 0
        for example, spec, count in zip(group, specs, counts):
            candidate_scores = scored[cursor:cursor + count]
            cursor += count
            for label, continuation, values in zip(spec["labels"], spec["continuations"], candidate_scores):
                values.update({"label": label, "continuation": continuation})
            score_key = "avg_loglikelihood" if spec["length_normalize"] else "loglikelihood"
            best = max(candidate_scores, key=lambda item: item[score_key])
            yield example, {
                "scoring_method": SCORING_METHOD,
                "metric": spec["metric"],
                "length_normalize": spec["length_normalize"],
                "prompt": spec["prompt"],
                "choice_scores": candidate_scores,
                "prediction": best["label"],
                "gold": example.normalized_gold_answer,
                "correct": best["label"] == example.normalized_gold_answer,
            }
