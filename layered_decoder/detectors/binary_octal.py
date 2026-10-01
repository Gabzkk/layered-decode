"""Binary and octal.

These now emit real bytes rather than `chr(value)`. The old `chr()` path
silently mangled anything above 127 (and raised on some platforms), which made
binary-wrapped non-ASCII payloads undecodable.
"""

import re
from typing import List, Tuple

from .base import Family, Layer

_BIN_STRIP = re.compile(rb"[\s,]+")
_BIN = re.compile(rb"^[01]+$")
_OCT_SPLIT = re.compile(rb"[\s,]+")
_OCT_TOKEN = re.compile(rb"^[0-7]{1,3}$")


class BinaryDetector(Layer):
    name = "Binary"
    priority = 25
    family = Family.STRUCTURAL

    @staticmethod
    def _clean(data: bytes) -> bytes:
        return _BIN_STRIP.sub(b"", data)

    def detect(self, data: bytes) -> float:
        cleaned = self._clean(data)
        if not cleaned or len(cleaned) % 8 != 0 or not _BIN.match(cleaned):
            return 0.0
        return 0.9

    def decode(self, data: bytes) -> List[bytes]:
        cleaned = self._clean(data)
        if not cleaned or len(cleaned) % 8 != 0 or not _BIN.match(cleaned):
            return []
        try:
            return [
                bytes(
                    int(cleaned[i:i + 8], 2)
                    for i in range(0, len(cleaned), 8)
                )
            ]
        except ValueError:
            return []

    def match_ratio(self, data: bytes) -> float:
        if not data:
            return 0.0
        return sum(1 for c in data.decode("latin1") if c in "01 \t\r\n,") / len(data)

    def find_matching_spans(self, data: bytes) -> List[Tuple[int, int, float]]:
        text = data.decode("latin1")
        return [
            (m.start(), m.end(), 0.9)
            for m in re.finditer(r"(?:[01]{8}[\s,]*){2,}", text)
        ]


class OctalDetector(Layer):
    name = "Octal"
    priority = 26
    family = Family.STRUCTURAL

    @staticmethod
    def _parts(data: bytes) -> list:
        return [p for p in _OCT_SPLIT.split(data.strip()) if p]

    def detect(self, data: bytes) -> float:
        parts = self._parts(data)
        if not parts:
            return 0.0
        for p in parts:
            if not _OCT_TOKEN.match(p) or int(p, 8) > 255:
                return 0.0
        return 0.85

    def decode(self, data: bytes) -> List[bytes]:
        parts = self._parts(data)
        if not parts:
            return []
        try:
            values = [int(p, 8) for p in parts]
        except ValueError:
            return []
        if any(v > 255 for v in values):
            return []
        return [bytes(values)]

    def match_ratio(self, data: bytes) -> float:
        if not data:
            return 0.0
        return sum(1 for c in data.decode("latin1") if c in "01234567 \t\r\n,") / len(data)

    def find_matching_spans(self, data: bytes) -> List[Tuple[int, int, float]]:
        text = data.decode("latin1")
        return [
            (m.start(), m.end(), 0.85)
            for m in re.finditer(r"(?:[0-7]{1,3}[\s,]+){3,}", text)
        ]
