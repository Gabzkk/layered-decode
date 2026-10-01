"""Caesar / ROT-N -- family B (statistical).

Replaces the old RotationDetector, which guessed a single best shift internally
and returned one candidate. That conflated detection and decoding: the search
could not see the runner-up shifts, so a shift that only becomes obvious after
the *next* layer decoded was unreachable.

Now all 25 non-identity shifts are returned and compete in the beam alongside
Base64 and the 255 XOR candidates.

The one guard worth keeping is the alphabetic precondition: a Caesar shift over
non-alphabetic bytes is a no-op, so brute-forcing it is pure waste. That is a
domain restriction, not a signature -- it does not identify the shift.
"""

import string
from typing import List

from .base import Family, Layer
from .xor import STATISTICAL_CONFIDENCE

_LETTERS = frozenset(string.ascii_letters)


class CaesarBruteForceLayer(Layer):
    name = "Caesar"
    priority = 50
    family = Family.STATISTICAL

    @staticmethod
    def _is_shiftable(data: bytes) -> bool:
        letters = sum(1 for b in data if chr(b) in _LETTERS)
        return letters >= 4 and letters / len(data) >= 0.5

    def detect(self, data: bytes) -> float:
        if not data or not self._is_shiftable(data):
            return 0.0
        return STATISTICAL_CONFIDENCE

    def decode(self, data: bytes) -> List[bytes]:
        if not self._is_shiftable(data):
            return []
        out = []
        for shift in range(1, 26):
            out.append(bytes(
                (
                    (b - ord("a") + shift) % 26 + ord("a")
                    if ord("a") <= b <= ord("z")
                    else (b - ord("A") + shift) % 26 + ord("A")
                    if ord("A") <= b <= ord("Z")
                    else b
                )
                for b in data
            ))
        return out
