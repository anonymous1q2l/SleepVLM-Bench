from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..constants import STAGES


LABEL_PATTERN = re.compile(r"(?<![A-Z0-9])(REM|N1|N2|N3|W)(?![A-Z0-9])", re.IGNORECASE)
PARSER_MODES = ("first", "last", "majority")


@dataclass(frozen=True, slots=True)
class ParseResult:
    label: str | None
    matches: tuple[str, ...]
    mode: str


def parse_stage(text: str | None, mode: str = "first") -> ParseResult:
    if mode not in PARSER_MODES:
        raise ValueError(f"unknown parser mode {mode!r}; expected one of {PARSER_MODES}")
    matches = tuple(match.upper() for match in LABEL_PATTERN.findall(text or ""))
    if not matches:
        return ParseResult(label=None, matches=(), mode=mode)
    if mode == "first":
        label = matches[0]
    elif mode == "last":
        label = matches[-1]
    else:
        counts = Counter(matches)
        maximum = max(counts.values())
        tied = {label for label, count in counts.items() if count == maximum}
        label = next(candidate for candidate in matches if candidate in tied)
    if label not in STAGES:
        raise AssertionError(f"parser produced an unexpected label: {label}")
    return ParseResult(label=label, matches=matches, mode=mode)

