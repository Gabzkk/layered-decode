"""Interleaving and every-Nth-character noise.

XorSingleByte has moved to its own module (`xor.py`) as an explicit family-B
layer: brute-forcing 255 keys and returning all of them is a different contract
from "guess one key internally", and burying it here alongside text heuristics
was what made that distinction invisible.
"""

import re
from typing import List, Tuple

from .base import Family, Layer
from ..scoring import DICTIONARY, score_text

WORDS = frozenset(DICTIONARY)
MAX_NOISE_CANDIDATES = 6


def _word_hits(text: str) -> int:
    return sum(1 for w in re.findall(r"[a-zA-Z]{3,}", text.lower()) if w in WORDS)


class InterleavedDetector(Layer):
    name = "Interleaved"
    priority = 70
    family = Family.STRUCTURAL

    def detect(self, data: bytes) -> float:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return 0.0
        if len(text) < 10:
            return 0.0
        orig = _word_hits(text)
        if _word_hits(text[::2]) > orig or _word_hits(text[1::2]) > orig:
            return 0.7
        return 0.0

    def decode(self, data: bytes) -> List[bytes]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return []
        if len(text) < 10:
            return []
        odd, even = text[::2], text[1::2]
        orig = _word_hits(text)
        odd_hits, even_hits = _word_hits(odd), _word_hits(even)
        if odd_hits <= orig and even_hits <= orig:
            return []
        ordered = sorted(
            (c for c in (odd, even) if _word_hits(c) > orig),
            key=_word_hits,
            reverse=True,
        )
        return [c.encode("utf-8") for c in ordered]


class NthCharNoiseDetector(Layer):
    name = "NthCharNoise"
    priority = 75
    family = Family.STRUCTURAL

    # A removed character must be one the text does not otherwise need.
    # Deleting *letters* is not noise-stripping, it is corruption -- and the
    # scorer rewards it, because `english_word_ratio` is a ratio: dropping one
    # letter that breaks a word raises it. Measured on correct plaintext, that
    # turned "a dead end" into "adead end" at a *higher* score (0.608 vs 0.598).
    #
    # "Non-alphanumeric" is the wrong test: the `X` in `the aXnd thXen` is a
    # letter and still noise. The reliable signal is frequency -- injected noise
    # uses characters the surrounding text does not need, so a removed character
    # that also occurs elsewhere is a real letter.
    @staticmethod
    def _is_injected_noise(removed: list, text: str) -> bool:
        if len(removed) < 2:
            return False
        for char in set(removed):
            used_as_noise = removed.count(char)
            used_in_text = text.count(char)
            if used_in_text > used_as_noise:
                return False  # the text needs this character elsewhere
        return True

    def detect(self, data: bytes) -> float:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return 0.0
        if len(text) < 10:
            return 0.0
        orig_hits = _word_hits(text)
        for candidate, removed in self._candidates(text):
            if _word_hits(candidate) > orig_hits and self._is_injected_noise(removed, text):
                return 0.75
        return 0.0

    @staticmethod
    def _candidates(text: str) -> List[Tuple[str, list]]:
        out: List[Tuple[str, list]] = []
        for n in range(2, 16):
            for offset in range(n):
                out.append((
                    "".join(c for i, c in enumerate(text) if i % n != offset),
                    [c for i, c in enumerate(text) if i % n == offset],
                ))
        return out

    def decode(self, data: bytes) -> List[bytes]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return []
        if len(text) < 10:
            return []
        orig_hits = _word_hits(text)

        out = {
            candidate
            for candidate, removed in self._candidates(text)
            if candidate != text
            and _word_hits(candidate) > orig_hits
            and self._is_injected_noise(removed, text)
        }
        if not out:
            return []
        ranked = sorted(out, key=lambda c: (_word_hits(c), len(c)), reverse=True)
        return [c.encode("utf-8") for c in ranked[:MAX_NOISE_CANDIDATES]]
