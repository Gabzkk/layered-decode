"""Decimal ASCII codes (e.g. `72 105` -> `Hi`). Emits real bytes."""

import re
from typing import List, Tuple

from .base import Family, Layer

_SPLIT = re.compile(rb"[\s,]+")


class AsciiDecimalDetector(Layer):
    name = "AsciiDecimal"
    priority = 30
    family = Family.STRUCTURAL

    @staticmethod
    def _parts(data: bytes) -> list:
        return [p for p in _SPLIT.split(data.strip()) if p]

    def detect(self, data: bytes) -> float:
        parts = self._parts(data)
        if not parts:
            return 0.0
        for p in parts:
            if not p.isdigit() or not (32 <= int(p) <= 126):
                return 0.0
        return 0.9

    def decode(self, data: bytes) -> List[bytes]:
        parts = self._parts(data)
        if not parts:
            return []
        try:
            values = [int(p) for p in parts]
        except ValueError:
            return []
        if any(not (32 <= v <= 126) for v in values):
            return []
        return [bytes(values)]

    def match_ratio(self, data: bytes) -> float:
        if not data:
            return 0.0
        return sum(
            1 for c in data.decode("latin1") if c in "0123456789 \t\r\n,"
        ) / len(data)

    def find_matching_spans(self, data: bytes) -> List[Tuple[int, int, float]]:
        text = data.decode("latin1")
        return [
            (m.start(), m.end(), 0.9)
            for m in re.finditer(r"(?:(?:1[0-2][0-9]|[3-9][0-9])[\s,]+){3,}", text)
        ]
