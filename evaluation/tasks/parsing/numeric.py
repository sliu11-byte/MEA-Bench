from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .base import ParseResult


NUMBER = r"[-+]?\s*[$€£¥]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def normalize_number(value: str) -> str | None:
    cleaned = re.sub(r"[$€£¥,\s]", "", value or "")
    try:
        number = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    # Decimal.quantize uses the active context precision and raises on very
    # long model-generated integers. Fixed-point formatting does not.
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", "+0", ""} else normalized


class NumericParser:
    name = "gsm8k_numeric_v1"

    def parse(self, response: str) -> ParseResult:
        text = response or ""
        marked_patterns = [
            re.compile(rf"(?i)####\s*({NUMBER})"),
            re.compile(rf"(?i)(?:final\s+answer|answer)\s*(?:is|:|=)?\s*({NUMBER})"),
        ]
        matches: list[tuple[int, str]] = []
        for pattern in marked_patterns:
            matches.extend((match.start(), match.group(1)) for match in pattern.finditer(text))
        if matches:
            raw = max(matches, key=lambda item: item[0])[1]
        else:
            all_numbers = list(re.finditer(NUMBER, text))
            if not all_numbers:
                return ParseResult(None, None, "failed", self.name)
            raw = all_numbers[-1].group(0)
        normalized = normalize_number(raw)
        if normalized is None:
            return ParseResult(raw, None, "failed", self.name)
        return ParseResult(raw.strip(), normalized, "success", self.name)
