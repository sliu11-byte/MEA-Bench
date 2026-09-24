from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ParseResult:
    parsed_prediction: str | None
    normalized_prediction: str | None
    status: str
    parser_name: str


class Parser(Protocol):
    name: str

    def parse(self, response: str) -> ParseResult:
        ...

