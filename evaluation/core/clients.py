from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import openai
from openai import OpenAI

from .config import GenerationSpec, ModelSpec, RetrySpec


def sanitize_base_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    hostname = parts.hostname or ""
    netloc = hostname
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), "", ""))


@dataclass
class RequestResult:
    status: str
    text: str
    retry_count: int
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    error_type: str | None = None
    error_message: str | None = None


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500
    return False


class OpenAICompatibleClient:
    def __init__(self, model: ModelSpec, retry: RetrySpec):
        self.model = model
        self.retry = retry
        self.client = OpenAI(
            base_url=model.base_url,
            api_key=model.api_key or "EMPTY",
            timeout=retry.request_timeout_seconds,
            max_retries=0,
        )

    def list_models(self) -> list[str]:
        return [item.id for item in self.client.models.list().data]

    def generate(
        self,
        prompt: str,
        generation: GenerationSpec,
        max_tokens: int,
        force_completions: bool = False,
    ) -> RequestResult:
        start = time.perf_counter()
        last_error: Exception | None = None
        attempts = self.retry.max_attempts
        for attempt in range(attempts):
            try:
                common: dict[str, Any] = {
                    "model": self.model.request_model_name,
                    "temperature": generation.temperature,
                    "top_p": generation.top_p,
                    "max_tokens": max_tokens,
                    "seed": generation.seed,
                }
                if generation.stop_sequences:
                    common["stop"] = generation.stop_sequences
                extra_body: dict[str, Any] = {}
                if generation.top_k is not None:
                    extra_body["top_k"] = generation.top_k
                if generation.repetition_penalty != 1.0:
                    extra_body["repetition_penalty"] = generation.repetition_penalty
                if extra_body:
                    common["extra_body"] = extra_body

                use_completions = force_completions or self.model.prompt_rendering_mode == "completions"
                if use_completions:
                    response = self.client.completions.create(prompt=prompt, **common)
                    text = response.choices[0].text or ""
                else:
                    response = self.client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "Follow the task exactly and provide the requested final-answer format.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        **common,
                    )
                    text = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                return RequestResult(
                    status="completed",
                    text=text,
                    retry_count=attempt,
                    latency_seconds=time.perf_counter() - start,
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= attempts or not _is_transient(exc):
                    break
                delay = min(
                    self.retry.max_backoff_seconds,
                    self.retry.initial_backoff_seconds * (self.retry.multiplier ** attempt),
                )
                jitter = random.Random(generation.seed + attempt).uniform(0.0, delay * 0.1)
                time.sleep(delay + jitter)

        assert last_error is not None
        error_message = str(last_error)
        if self.model.api_key and self.model.api_key != "EMPTY":
            error_message = error_message.replace(self.model.api_key, "<redacted>")
        return RequestResult(
            status="failed",
            text="",
            retry_count=max(0, min(attempts - 1, attempt)),
            latency_seconds=time.perf_counter() - start,
            error_type=type(last_error).__name__,
            error_message=error_message,
        )
