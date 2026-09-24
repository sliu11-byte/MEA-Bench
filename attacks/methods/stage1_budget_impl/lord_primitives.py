from __future__ import annotations

"""
======================================================================
LORD_PRIMITIVES ---

Minimal, testable LoRD math primitives.

This module keeps the semantic pieces that should stay explicit:
sequence scoring, candidate comparison, tau resolution, clipping, and
black-box-only preference loss helpers.
======================================================================
"""

from dataclasses import dataclass
from math import exp, log
from typing import Iterable, Mapping, Sequence


class LordPrimitiveError(RuntimeError):
    pass


class LordTauResolutionError(LordPrimitiveError):
    pass


class LordScoreError(LordPrimitiveError):
    pass


def _coerce_float_sequence(values: Sequence[float] | Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _coerce_mask(mask: Sequence[int] | Sequence[bool] | Iterable[int | bool]) -> tuple[int, ...]:
    return tuple(1 if bool(value) else 0 for value in mask)


@dataclass(frozen=True)
class SequenceScoreTrace:
    token_log_probs: tuple[float, ...]
    response_mask: tuple[int, ...]
    sequence_log_prob_sum: float
    length_normalized_mean_log_prob: float
    probability_from_mean_log_prob: float
    effective_token_count: int


@dataclass(frozen=True)
class CandidateComparison:
    reference_score: float
    candidate_score: float
    delta: float
    positive_score: float
    negative_score: float
    swapped: bool


@dataclass(frozen=True)
class LoRDTransitionDecision:
    comparison: CandidateComparison
    tau1_hit: bool
    tau2_hit: bool
    teacher_anchor: bool


@dataclass(frozen=True)
class LoRDSnapshotDecision:
    previous_scores: tuple[float, float]
    current_scores: tuple[float, float]
    deltas: tuple[float, float]
    positive_index: int
    negative_index: int
    swapped: bool
    tau1_hit: bool
    tau2_hit: bool
    teacher_anchor: bool


@dataclass(frozen=True)
class TauSpec:
    code_var: str
    comparison_object: str
    quantity: str
    value_range: tuple[float, float]
    parser_default: float | None = None
    shell_override: float | None = None
    readme_value: float | None = None
    paper_value: float | None = None
    actual_trigger_value: float | None = None


def build_sequence_score_trace(
    token_log_probs: Sequence[float] | Iterable[float],
    response_mask: Sequence[int] | Sequence[bool] | Iterable[int | bool],
) -> SequenceScoreTrace:
    token_log_probs = _coerce_float_sequence(token_log_probs)
    response_mask = _coerce_mask(response_mask)
    if len(token_log_probs) != len(response_mask):
        raise LordScoreError("token_log_probs and response_mask must have the same length.")

    active_log_probs = [value for value, active in zip(token_log_probs, response_mask) if active]
    if not active_log_probs:
        raise LordScoreError("at least one response token must be active in the mask.")

    sequence_log_prob_sum = float(sum(active_log_probs))
    effective_token_count = len(active_log_probs)
    mean_log_prob = sequence_log_prob_sum / effective_token_count
    probability_from_mean = float(exp(mean_log_prob))
    return SequenceScoreTrace(
        token_log_probs=token_log_probs,
        response_mask=response_mask,
        sequence_log_prob_sum=sequence_log_prob_sum,
        length_normalized_mean_log_prob=mean_log_prob,
        probability_from_mean_log_prob=probability_from_mean,
        effective_token_count=effective_token_count,
    )


def compare_candidate_scores(reference_score: float, candidate_score: float) -> CandidateComparison:
    reference_score = float(reference_score)
    candidate_score = float(candidate_score)
    delta = candidate_score - reference_score
    swapped = candidate_score > reference_score
    positive_score = candidate_score if swapped else reference_score
    negative_score = reference_score if swapped else candidate_score
    return CandidateComparison(
        reference_score=reference_score,
        candidate_score=candidate_score,
        delta=delta,
        positive_score=positive_score,
        negative_score=negative_score,
        swapped=swapped,
    )


def evaluate_lord_transition(
    reference_score: float,
    candidate_score: float,
    *,
    tau1: float,
    tau2: float,
    tau_delta: float,
) -> LoRDTransitionDecision:
    comparison = compare_candidate_scores(reference_score, candidate_score)
    tau1_hit = comparison.positive_score < tau1 and comparison.delta < tau_delta
    tau2_hit = comparison.negative_score < tau2
    return LoRDTransitionDecision(
        comparison=comparison,
        tau1_hit=tau1_hit,
        tau2_hit=tau2_hit,
        teacher_anchor=tau1_hit,
    )


def evaluate_lord_snapshot_transition(
    previous_scores: Sequence[float],
    current_scores: Sequence[float],
    *,
    tau1: float,
    tau2: float,
    tau_delta: float,
) -> LoRDSnapshotDecision:
    """Apply LoRD locality rules to exactly two snapshot candidates.

    Candidate polarity is determined by temporal improvement
    ``current - previous``.  The anchor rule then uses the selected
    candidate's current score and temporal delta.  This keeps the two
    comparisons explicit instead of conflating candidate ranking with
    snapshot timing.
    """
    if len(previous_scores) != 2 or len(current_scores) != 2:
        raise LordScoreError("LoRD snapshot comparison requires exactly two candidates.")
    previous = (float(previous_scores[0]), float(previous_scores[1]))
    current = (float(current_scores[0]), float(current_scores[1]))
    deltas = (current[0] - previous[0], current[1] - previous[1])
    comparison = compare_candidate_scores(deltas[0], deltas[1])
    positive_index = 1 if comparison.swapped else 0
    negative_index = 1 - positive_index
    tau1_hit = current[positive_index] < float(tau1) and deltas[positive_index] < float(tau_delta)
    tau2_hit = current[negative_index] < float(tau2)
    return LoRDSnapshotDecision(
        previous_scores=previous,
        current_scores=current,
        deltas=deltas,
        positive_index=positive_index,
        negative_index=negative_index,
        swapped=comparison.swapped,
        tau1_hit=tau1_hit,
        tau2_hit=tau2_hit,
        teacher_anchor=tau1_hit,
    )


def clip_ratio(value: float, epsilon: float = 0.5) -> float:
    if epsilon < 0:
        raise LordPrimitiveError("epsilon must be non-negative.")
    lower = 1.0 - epsilon
    upper = 1.0 + epsilon
    value = float(value)
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def log_clip_ratio(value: float, epsilon: float = 0.2) -> float:
    if epsilon <= 0 or epsilon >= 1:
        raise LordPrimitiveError("epsilon must be in the open interval (0, 1).")
    lower = log(1.0 - epsilon)
    upper = log(1.0 + epsilon)
    value = float(value)
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def resolve_tau_value(
    *,
    code_var: str,
    parser_default: float | None = None,
    shell_override: float | None = None,
    readme_value: float | None = None,
    paper_value: float | None = None,
    actual_trigger_value: float | None = None,
) -> float:
    candidates = {
        "actual_trigger_value": actual_trigger_value,
        "shell_override": shell_override,
        "parser_default": parser_default,
        "readme_value": readme_value,
        "paper_value": paper_value,
    }
    present = {name: float(value) for name, value in candidates.items() if value is not None}
    if not present:
        raise LordTauResolutionError(f"{code_var} has no explicit value to resolve.")

    unique_values = {value for value in present.values()}
    if len(unique_values) != 1:
        raise LordTauResolutionError(
            f"{code_var} is ambiguous: "
            + ", ".join(f"{name}={value!r}" for name, value in present.items())
        )
    return unique_values.pop()


def compute_black_box_pair_loss(
    *,
    positive_score: float,
    negative_score: float,
    anchor_score: float | None = None,
    epsilon: float = 0.2,
) -> float:
    preference_gap = float(positive_score) - float(negative_score)
    loss = -log_clip_ratio(preference_gap, epsilon=epsilon)
    if anchor_score is not None:
        loss += -log_clip_ratio(float(anchor_score) - float(positive_score), epsilon=epsilon)
    return loss


def compute_differentiable_black_box_loss(
    *,
    positive_log_prob,
    negative_log_prob,
    anchor_log_prob,
    lambda1: float = 0.5,
    epsilon: float = 0.2,
):
    """Differentiable equation-11 loss using student log-probabilities only."""
    if not 0.0 < float(lambda1) < 1.0:
        raise LordPrimitiveError("lambda1 must be in the open interval (0, 1).")
    if epsilon <= 0 or epsilon >= 1:
        raise LordPrimitiveError("epsilon must be in the open interval (0, 1).")
    import torch

    preference = negative_log_prob - positive_log_prob
    anchor = torch.clamp(
        negative_log_prob - anchor_log_prob,
        min=log(1.0 - epsilon),
        max=log(1.0 + epsilon),
    )
    return (1.0 - lambda1) * preference + lambda1 * anchor


__all__ = [
    "CandidateComparison",
    "LoRDTransitionDecision",
    "LoRDSnapshotDecision",
    "LordPrimitiveError",
    "LordScoreError",
    "LordTauResolutionError",
    "SequenceScoreTrace",
    "TauSpec",
    "build_sequence_score_trace",
    "clip_ratio",
    "compare_candidate_scores",
    "compute_black_box_pair_loss",
    "compute_differentiable_black_box_loss",
    "evaluate_lord_transition",
    "evaluate_lord_snapshot_transition",
    "log_clip_ratio",
    "resolve_tau_value",
]
