from __future__ import annotations

import re

from .base import ParseResult


class CodeCompletionParser:
    name = "humaneval_completion_v1"

    def parse(self, response: str) -> ParseResult:
        text = (response or "").rstrip()
        if not text.strip():
            return ParseResult(None, None, "failed", self.name)
        blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        completion = blocks[-1].strip("\n") if blocks else text
        if not completion.strip():
            return ParseResult(None, None, "failed", self.name)
        return ParseResult(completion, completion, "success", self.name)
