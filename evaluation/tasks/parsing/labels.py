from __future__ import annotations

import re

from .base import ParseResult


def _canonical(value: str) -> str:
    return re.sub(r"[\s_]+", "-", value.strip().lower())


class LabelParser:
    name = "categorical_label_v1"

    def __init__(self, labels: list[str], aliases: dict[str, str] | None = None):
        self.labels = {_canonical(label): label for label in labels}
        self.aliases = {_canonical(key): value for key, value in (aliases or {}).items()}
        alternatives = sorted(
            set(labels) | set((aliases or {}).keys()),
            key=len,
            reverse=True,
        )
        escaped = "|".join(re.escape(item) for item in alternatives)
        self._marked = re.compile(
            rf"(?i)(?:final\s+answer|answer|label|sentiment|relation|stance)\s*"
            rf"(?:is|:|=|-)?\s*[\"']?({escaped})[\"']?\b"
        )
        self._any = re.compile(rf"(?i)(?<![A-Za-z0-9])({escaped})(?![A-Za-z0-9])")

    def normalize(self, value: str) -> str | None:
        key = _canonical(value)
        if key in self.aliases:
            key = _canonical(self.aliases[key])
        return key if key in self.labels else None

    def parse(self, response: str) -> ParseResult:
        text = response or ""
        marked = list(self._marked.finditer(text))
        candidates = marked if marked else list(self._any.finditer(text))
        if not candidates:
            return ParseResult(None, None, "failed", self.name)
        raw = candidates[-1].group(1)
        normalized = self.normalize(raw)
        if normalized is None:
            return ParseResult(raw, None, "failed", self.name)
        return ParseResult(raw, normalized, "success", self.name)

