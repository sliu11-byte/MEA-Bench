from __future__ import annotations

import re

from .base import ParseResult


class MultipleChoiceParser:
    name = "multiple_choice_v1"

    def __init__(self, labels: list[str]):
        self.labels = [str(label).upper() for label in labels]
        alternatives = "|".join(re.escape(label) for label in sorted(self.labels, key=len, reverse=True))
        self._marked_patterns = [
            re.compile(
                rf"(?i)(?:final\s+answer|answer|correct\s+(?:answer|option)|correct\s+choice)\s*"
                rf"(?:is|:|=|-)?\s*\(?\s*({alternatives})\s*\)?\b"
            ),
            re.compile(
                rf"(?i)(?:therefore|thus|hence|so)[^\n.]{{0,100}}?"
                rf"(?:answer|option|choice)?\s*(?:is|:|=)?\s*\(?\s*({alternatives})\s*\)?\b"
            ),
        ]
        self._exact = re.compile(rf"(?i)^\s*\(?\s*({alternatives})\s*\)?[\s.!]*$")
        self._standalone = re.compile(rf"(?i)(?<![A-Za-z0-9])\(?({alternatives})\)?(?![A-Za-z0-9])")

    def parse(self, response: str) -> ParseResult:
        text = response or ""
        marked: list[tuple[int, str]] = []
        for pattern in self._marked_patterns:
            marked.extend((match.start(), match.group(1).upper()) for match in pattern.finditer(text))
        if marked:
            value = max(marked, key=lambda item: item[0])[1]
            return ParseResult(value, value, "success", self.name)

        exact = self._exact.match(text)
        if exact:
            value = exact.group(1).upper()
            return ParseResult(value, value, "success", self.name)

        nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if nonempty_lines:
            matches = list(self._standalone.finditer(nonempty_lines[-1]))
            if len(matches) == 1:
                value = matches[0].group(1).upper()
                return ParseResult(value, value, "success", self.name)

        return ParseResult(None, None, "failed", self.name)

