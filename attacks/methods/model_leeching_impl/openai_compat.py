from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int | None = 42


def generate_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    mode: str,
    settings: GenerationSettings,
    timeout: float = 600.0,
    max_attempts: int = 4,
) -> tuple[str, dict[str, int | None], float, int]:
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
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
        except Exception as exc:  # retries should preserve the final exception context
            last_error = exc
            time.sleep(min(30.0, 2.0**attempt))
    assert last_error is not None
    raise last_error


