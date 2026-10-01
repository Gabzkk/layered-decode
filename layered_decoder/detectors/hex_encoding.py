r"""Hex. Tolerates the usual separators (`0x`, `\x`, colons, whitespace, commas)."""

import re
from typing import List, Tuple

from .base import Family, Layer

_STRIP = re.compile(rb"[\s:,\\|]|0x|\\x", re.IGNORECASE)


class HexDetector(Layer):
    name = "Hex"
    priority = 20
    family = Family.STRUCTURAL

    @staticmethod
    def _strip(data: bytes) -> bytes:
        return _STRIP.sub(b"", data)

    def detect(self, data: bytes) -> float:
        stripped = self._strip(data)
        if not stripped or len(stripped) % 2 != 0:
            return 0.0
        try:
            bytes.fromhex(stripped.decode("ascii"))
        except ValueError:
            return 0.0
        return 0.85

    def decode(self, data: bytes) -> List[bytes]:
        stripped = self._strip(data)
        if not stripped or len(stripped) % 2 != 0:
            return []
        try:
            return [bytes.fromhex(stripped.decode("ascii"))]
        except ValueError:
            return []

    def match_ratio(self, data: bytes) -> float:
        if not data:
            return 0.0
        hex_chars = set("0123456789abcdefABCDEF")
        return sum(1 for c in data.decode("latin1") if c in hex_chars) / len(data)

    def find_matching_spans(self, data: bytes) -> List[Tuple[int, int, float]]:
        text = data.decode("latin1")
        return [
            (m.start(), m.end(), 0.85)
            for m in re.finditer(r"(?:0x)?[0-9a-fA-F]{12,}", text)
        ]
