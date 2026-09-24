from __future__ import annotations

import time
import os
import math
from dataclasses import dataclass
from contextlib import closing

from openai import APITimeoutError, OpenAI


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int | None = 42


def _is_countermeasure_proxy_timeout(error: Exception) -> bool:
    if getattr(error, "status_code", None) != 502:
        return False
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    detail = body.get("error")
    return (
        isinstance(detail, dict)
        and detail.get("type") == "countermeasure_proxy_error"
        and detail.get("message") == "timed out"
    )


def generate_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    mode: str,
    settings: GenerationSettings,
    timeout: float | None = None,
    max_attempts: int = 4,
) -> tuple[str, dict[str, int | None], float, int]:
    if timeout is None:
        timeout = float(os.environ.get("QEDKS_REQUEST_TIMEOUT_SECONDS", "1800"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("QEDKS request timeout must be a finite positive number")
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
    with closing(client):
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            start = time.monotonic()
            try:
                common = {
                    "model": model,
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "max_tokens": settings.max_tokens,
                }
                if settings.seed is not None:
                    common["seed"] = settings.seed
                if mode == "chat":
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        **common,
                    )
                    text = response.choices[0].message.content or ""
                elif mode == "completion":
                    response = client.completions.create(prompt=prompt, **common)
                    text = response.choices[0].text or ""
                else:
                    raise ValueError("mode must be 'chat' or 'completion'")
                usage = getattr(response, "usage", None)
                tokens = {
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                return text, tokens, time.monotonic() - start, attempt
            except APITimeoutError:
                # The oracle may still be generating; retries queue duplicate work.
                raise
            except Exception as exc:  # retries should preserve the final exception context
                if _is_countermeasure_proxy_timeout(exc):
                    # The proxy stopped waiting, but its oracle request may still
                    # be running. Retrying would enqueue duplicate generation.
                    raise
                last_error = exc
                time.sleep(min(30.0, 2.0**attempt))
        assert last_error is not None
        raise last_error


