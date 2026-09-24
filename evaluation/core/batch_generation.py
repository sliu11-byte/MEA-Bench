"""Shared padded generation and resumable item batching for local HF models."""

from __future__ import annotations

import os


def batch_size(kind, *, teacher=False):
    defaults = {"mc": 8, "math": 4, "heldout": 4}
    prefix = "EVAL_TEACHER_BATCH_" if teacher else "EVAL_BATCH_"
    default = (2 if kind == "mc" else 1) if teacher else defaults[kind]
    value = int(os.environ.get(prefix + kind.upper(), default))
    if value < 1:
        raise ValueError(f"{prefix + kind.upper()} must be positive")
    return value


def generate_batch(model, tokenizer, prompts, max_tokens, chat):
    import torch

    if not prompts:
        return []
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Batch generation requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    try:
        if chat:
            # Tokenize rendered chats without inserting a second BOS token.
            texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            ) for prompt in prompts]
            inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
        else:
            inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            result = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.batch_decode(result[:, width:], skip_special_tokens=True)
    finally:
        tokenizer.padding_side = old_side


def iter_generated(model, tokenizer, items, prompt_fn, max_tokens, chat, size):
    """Keep item/output association while grouping similarly sized inputs."""
    if size < 1:
        raise ValueError("Batch size must be positive")
    pending = sorted(items, key=lambda item: len(prompt_fn(item))) if size > 1 else list(items)
    for start in range(0, len(pending), size):
        batch = pending[start:start + size]
        texts = generate_batch(model, tokenizer, [prompt_fn(item) for item in batch], max_tokens, chat)
        if len(texts) != len(batch):
            raise ValueError("Generated batch has wrong output count")
        for item, text in zip(batch, texts):
            yield item, text
