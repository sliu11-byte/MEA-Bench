"""Rewrite instructions for Trace Rewriting."""

from __future__ import annotations

LOCAL_OPTIMIZED_REWRITE_INSTRUCTION = """Rewrite the answer into a more professional, dense, and formally structured response while preserving the final answer and semantic correctness. Keep the content self-contained and coherent, but make the expression less directly useful as simple supervised fine-tuning data. Do not mention that you are rewriting or defending against distillation."""

OFFICIAL_OPTIMIZED_REWRITE_INSTRUCTION = """You are a specialist in the mathematical sciences. Your task is to process a provided chain of logical deductions and reformulate it utilizing an esoteric, formal, and densely technical lexicon, thereby obfuscating its clarity for a non-specialist audience. Refrain from any extraneous conversational text. Your terminal output must conform to the structure \\boxed{{SOLUTION}}, ensuring all pertinent units are encapsulated within the brackets and that any numerical computations are fully resolved. You must still output a reasoning trace that leads to the final answer."""

SEMANTIC_REWRITE_INSTRUCTION = """Paraphrase the answer while preserving its meaning, correctness, and final conclusion. Use different wording and organization, but do not add unsupported facts. Do not mention the rewrite process."""

REWRITE_INSTRUCTIONS = {
    "optimized_prompt": OFFICIAL_OPTIMIZED_REWRITE_INSTRUCTION,
    "optimized_prompt_official": OFFICIAL_OPTIMIZED_REWRITE_INSTRUCTION,
    "optimized_prompt_local": LOCAL_OPTIMIZED_REWRITE_INSTRUCTION,
    "semantic_prompt": SEMANTIC_REWRITE_INSTRUCTION,
}


# Backward-compatible alias for callers that imported the old constant directly.
OPTIMIZED_REWRITE_INSTRUCTION = OFFICIAL_OPTIMIZED_REWRITE_INSTRUCTION


def resolve_rewrite_instruction(strategy: str, explicit_instruction: str | None = None) -> str:
    if explicit_instruction:
        return explicit_instruction
    try:
        return REWRITE_INSTRUCTIONS[strategy]
    except KeyError as exc:
        known = ", ".join(sorted(REWRITE_INSTRUCTIONS))
        raise ValueError(f"Unknown rewrite_strategy={strategy!r}; expected one of: {known}") from exc
