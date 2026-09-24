"""Shared teacher generation helpers used by all defenses."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_causal_lm(
    model_name: str,
    *,
    device: Optional[str] = None,
    torch_dtype: Optional[str] = "auto",
    trust_remote_code: bool = True,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
    dtype = torch_dtype
    if dtype == "auto":
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def format_prompt(tokenizer, query: str, *, system_prompt: Optional[str] = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if system_prompt:
        return f"{system_prompt}\n\nUser: {query}\nAssistant:"
    return query


def generate_responses(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    logits_processor: Optional[Any] = None,
    logits_warper: Optional[Any] = None,
    batch_size: int = 1,
) -> List[str]:
    """Generate completions for a list of already-formatted prompts."""
    import torch

    responses: List[str] = []
    do_sample = temperature is not None and temperature > 0

    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
                gen_kwargs["renormalize_logits"] = True
            if logits_warper is not None:
                from transformers import LogitsProcessorList as LPL

                existing = gen_kwargs.get("logits_processor")
                if existing is None:
                    gen_kwargs["logits_processor"] = LPL([logits_warper])
                else:
                    existing.append(logits_warper)
                gen_kwargs["renormalize_logits"] = True

            import logging
            import time

            logger = logging.getLogger("defended_teacher.generation")
            started = time.monotonic()
            logger.info("generation start: batch=%s input_tokens=%s max_new_tokens=%s eos=%s use_cache=%s",
                        len(batch), inputs["attention_mask"].sum(dim=1).tolist(), max_new_tokens,
                        gen_kwargs["eos_token_id"], getattr(model.generation_config, "use_cache", None))
            diagnostic_hook = None
            if os.environ.get("MEA_GENERATION_DIAGNOSTICS", "0") == "1":
                diagnostic_steps = [0]

                def inspect_forward(module, args, kwargs):
                    if diagnostic_steps[0] >= 5:
                        return
                    diagnostic_steps[0] += 1
                    ids = kwargs.get("input_ids")
                    cache = kwargs.get("past_key_values")
                    logger.info(
                        "generation diagnostic: forward=%s input_shape=%s use_cache=%s cache=%s cache_length=%s",
                        diagnostic_steps[0], tuple(ids.shape) if ids is not None else None,
                        kwargs.get("use_cache"), type(cache).__name__ if cache is not None else None,
                        cache.get_seq_length() if hasattr(cache, "get_seq_length") else None,
                    )

                logger.info(
                    "generation diagnostic: torch=%s attention=%s model_use_cache=%s training=%s gradient_checkpointing=%s device_map=%s",
                    torch.__version__, getattr(model.config, "_attn_implementation", None),
                    getattr(model.config, "use_cache", None), model.training,
                    getattr(model, "is_gradient_checkpointing", None), getattr(model, "hf_device_map", None),
                )
                diagnostic_hook = model.register_forward_pre_hook(inspect_forward, with_kwargs=True)
            try:
                outputs = model.generate(**inputs, **gen_kwargs)
            finally:
                if diagnostic_hook is not None:
                    diagnostic_hook.remove()
            logger.info("generation end: elapsed_seconds=%.3f generated_width=%s",
                        time.monotonic() - started, outputs.shape[1] - inputs["input_ids"].shape[1])
            prompt_len = inputs["input_ids"].shape[1]
            for i in range(outputs.shape[0]):
                gen_ids = outputs[i, prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                responses.append(text)
    return responses


def make_transcript_record(
    *,
    query_id: str,
    query: str,
    response: str,
    defense: str,
    teacher_model: str,
    defense_role: str = "teacher",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "query_id": query_id,
        "query": query,
        "response": response,
        "defense": defense,
        "defense_role": defense_role,
        "teacher_model": teacher_model,
    }
    if extra:
        rec.update(extra)
    return rec


def _openai_url(base_url: str, route: str) -> str:
    return base_url.rstrip("/") + route


def _post_openai_json(url: str, payload: Dict[str, Any], *, api_key: str, timeout: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible request failed: HTTP {exc.code}: {detail}") from exc
    return json.loads(decoded)


def generate_openai_chat_responses(
    prompts: Sequence[str],
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    timeout: int = 600,
    max_retries: int = 3,
    extra_body: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate responses through an OpenAI-compatible chat completions endpoint."""
    responses: List[str] = []
    url = _openai_url(base_url, "/chat/completions")
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if extra_body:
            payload.update(extra_body)
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                result = _post_openai_json(url, payload, api_key=api_key, timeout=timeout)
                choice = result["choices"][0]
                message = choice.get("message") or {}
                content = message.get("content") or message.get("reasoning_content") or choice.get("text") or ""
                responses.append(str(content))
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
        else:
            raise RuntimeError(f"OpenAI-compatible request failed: {last_error}")
    return responses
