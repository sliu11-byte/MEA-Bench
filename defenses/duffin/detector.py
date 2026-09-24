"""Knowledge-DuFFin detector helpers.

This implementation follows the official Knowledge-DuFFin shape: query both
teacher and suspect student on the same multiple-choice probe set, extract the
selected option letter, and score answer-sequence similarity. Unlike the handoff
version, choices are not limited to A-D; probe options can use A-P.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

CHOICES = tuple("ABCDEFGHIJKLMNOP")


def allowed_choice_letters(options: Any = None) -> list[str]:
    if options is None:
        return list(CHOICES)
    if isinstance(options, dict):
        letters = [str(k).strip().upper() for k in options.keys()]
        allowed = [letter for letter in CHOICES if letter in letters]
        return allowed or list(CHOICES[: min(len(options), len(CHOICES))])
    if isinstance(options, (list, tuple)):
        return list(CHOICES[: min(len(options), len(CHOICES))])
    return list(CHOICES)


def extract_choice(text: Optional[str], *, allowed_choices: Sequence[str] | None = None) -> Optional[str]:
    if text is None:
        return None
    allowed = [c.upper() for c in (allowed_choices or CHOICES)]
    if not allowed:
        return None
    choice_class = "".join(re.escape(c) for c in allowed)
    raw = str(text).strip().upper()

    if raw in allowed:
        return raw

    exact = re.fullmatch(rf"[\(\[]?([{choice_class}])[\)\]]?[\.\:]?", raw)
    if exact:
        return exact.group(1)

    # Only accept an explicitly marked final answer. Matching an arbitrary
    # standalone letter misclassifies ordinary prose such as "You are a ...".
    pattern = rf"(?:THE\s+)?(?:FINAL\s+)?(?:ANSWER|CHOICE|OPTION)\s*(?:IS|:|=)?\s*[\(\[]?([{choice_class}])[\)\]]?\b"
    matches = list(re.finditer(pattern, raw, re.DOTALL))
    if matches:
        return matches[-1].group(1)
    return None


def compute_knowledge_duffin(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        row for row in rows
        if row.get("teacher_choice") is not None and row.get("student_choice") is not None
    ]
    if not valid:
        raise ValueError("No valid DuFFin probe pairs.")

    matches = sum(row["teacher_choice"] == row["student_choice"] for row in valid)
    similarity = matches / len(valid)
    return {
        "score": similarity,
        "score_name": "knowledge_fingerprint_similarity",
        "score_direction": "higher",
        "normalized_hamming_distance": 1.0 - similarity,
        "matches": matches,
        "valid_probes": len(valid),
        "total_probes": len(rows),
    }


def build_probe_prompt(record: Dict[str, Any]) -> tuple[str, list[str]]:
    if record.get("query") is not None:
        query = str(record["query"]).strip()
    elif record.get("question") is not None:
        query = str(record["question"]).strip()
    else:
        raise KeyError("DuFFin probe requires 'query' or 'question'.")

    options = record.get("options") or record.get("choices")
    allowed = allowed_choice_letters(options)
    def format_options(raw_options: Any) -> tuple[list[str], list[str]]:
        letters = allowed_choice_letters(raw_options)
        if isinstance(raw_options, dict):
            normalized = {str(key).strip().upper(): value for key, value in raw_options.items()}
            lines = [f"{letter}. {normalized[letter]}" for letter in letters if letter in normalized]
        else:
            lines = [f"{letters[i]}. {value}" for i, value in enumerate(list(raw_options or [])[: len(letters)])]
        return lines, letters

    few_shot = record.get("few_shot_examples") or []
    if few_shot:
        sections = [
            "The following are multiple-choice questions. Think step by step and end every answer with "
            "the exact sentence 'The answer is (X).', where X is one option letter."
        ]
        for example in few_shot:
            example_options = example.get("options") or example.get("choices")
            example_lines, _ = format_options(example_options)
            example_answer = str(example.get("cot_content") or "").strip()
            if not example_answer:
                letter = str(example.get("answer") or "").strip().upper()
                example_answer = f"The answer is ({letter})."
            sections.append(
                "Question:\n" + str(example["question"]).strip()
                + "\nOptions:\n" + "\n".join(example_lines)
                + "\nAnswer: Let's think step by step. " + example_answer
            )
        lines, allowed = format_options(options)
        sections.append(
            "Question:\n" + query + "\nOptions:\n" + "\n".join(lines)
            + "\nAnswer: Let's think step by step."
        )
        return "\n\n".join(sections), allowed

    if options:
        if isinstance(options, dict):
            normalized_options = {str(key).strip().upper(): value for key, value in options.items()}
            if any(letter in normalized_options for letter in allowed):
                lines = [f"{letter}. {normalized_options[letter]}" for letter in allowed if letter in normalized_options]
            else:
                values = list(options.values())
                lines = [f"{allowed[i]}. {value}" for i, value in enumerate(values[: len(allowed)])]
        else:
            lines = [f"{allowed[i]}. {value}" for i, value in enumerate(list(options)[: len(allowed)])]
        query += "\n\n" + "\n".join(lines)

    return query + "\n\nAnswer with only one letter from: " + ", ".join(allowed) + ".", allowed


def format_chat_prompt(tokenizer: Any, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### User:\n{text}\n\n### Assistant:\n"


def load_local_model(model_path: str, base_model: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    tokenizer_sources = [model_path]
    if base_model and base_model != model_path:
        tokenizer_sources.append(base_model)
    tokenizer = None
    for tokenizer_source in tokenizer_sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
            break
        except (OSError, ValueError):
            continue
    if tokenizer is None:
        raise OSError(f"Could not load tokenizer from: {tokenizer_sources}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if base_model is not None:
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def query_local_model(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    allowed_choices: Sequence[str],
    max_new_tokens: int = 8,
) -> tuple[str, Optional[str]]:
    import torch

    text = format_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return response, extract_choice(response, allowed_choices=allowed_choices)


def query_local_model_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    allowed_choices: Sequence[Sequence[str]],
    max_new_tokens: int = 1024,
) -> list[tuple[str, Optional[str]]]:
    """Generate a batch while preserving one parsed result per input prompt."""
    import torch

    if len(prompts) != len(allowed_choices):
        raise ValueError("prompts and allowed_choices must have the same length")
    texts = [format_chat_prompt(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    responses = tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)
    return [
        (response.strip(), extract_choice(response, allowed_choices=choices))
        for response, choices in zip(responses, allowed_choices)
    ]
