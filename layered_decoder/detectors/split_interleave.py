"""Odd/even and half-split stream separation.

Like SymbolNoise, this already enumerated many recombinations internally and
then threw all but one away. It now returns them all (capped) and lets the
scorer choose.
"""

from typing import List

from .base import Family, Layer
from ..scoring import english_word_score, has_known_patterns, score_text

NOISE_CHARS = set("#@$%&*?!()[]<>+=^~|\\/;`'\"")
MAX_CANDIDATES = 8


class SplitInterleaveDetector(Layer):
    name = "SplitInterleave"
    priority = 68
    family = Family.STRUCTURAL

    def _candidates(self, text: str) -> List[str]:
        n = len(text)
        if n < 8:
            return []
        candidates: List[str] = []

        for k in (2, 1):
            if n % (2 * k) == 0:
                chunks = [text[i:i + k] for i in range(0, n, k)]
                odd_chunks = chunks[1::2]
                even_chunks = chunks[0::2]
                for s_odd, s_even in (
                    ("".join(odd_chunks)[::-1], "".join(even_chunks)[::-1]),
                    ("".join(odd_chunks[::-1]), "".join(even_chunks[::-1])),
                ):
                    for s1, s2 in ((s_even, s_odd), (s_odd, s_even)):
                        inter = "".join(a + b for a, b in zip(s1, s2))
                        candidates.append(inter)
                        for offset in range(4):
                            candidates.append(
                                "".join(c for i, c in enumerate(inter) if i % 4 != offset)
                            )
                        cleaned = "".join(c for c in inter if c not in NOISE_CHARS)
                        if cleaned != inter:
                            candidates.append(cleaned)

        mid = n // 2
        h1, h2 = text[:mid], text[mid:]
        for s1, s2 in ((h1[::-1], h2[::-1]), (h2[::-1], h1[::-1])):
            candidates.append("".join(a + b for a, b in zip(s1, s2)))

        return candidates

    @staticmethod
    def _rank(cands: List[str]):
        return lambda s: (
            len(has_known_patterns(s)) > 0,
            english_word_score(s),
            score_text(s),
        )

    def detect(self, data: bytes) -> float:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return 0.0
        orig_patterns = has_known_patterns(text)
        orig_words = english_word_score(text)
        orig_score = score_text(text)
        if orig_patterns or orig_words >= 0.85:
            return 0.0
        cands = self._candidates(text)
        if not cands:
            return 0.0
        best = max(cands, key=self._rank(cands))
        if has_known_patterns(best) and not orig_patterns:
            return 0.90
        w_score = english_word_score(best)
        s_score = score_text(best)
        if w_score >= 0.40 and w_score > orig_words + 0.15 and s_score > orig_score + 0.10:
            return 0.80
        return 0.0

    def decode(self, data: bytes) -> List[bytes]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return []
        cands = self._candidates(text)
        if not cands:
            return []
        orig_words = english_word_score(text)
        orig_score = score_text(text)
        keep = [
            c for c in cands
            if has_known_patterns(c)
            or english_word_score(c) > orig_words
            or score_text(c) > orig_score + 0.05
        ]
        ranked = sorted(set(keep), key=self._rank(cands), reverse=True)
        return [c.encode("utf-8") for c in ranked[:MAX_CANDIDATES]]
