"""Symbol-noise stripping (e.g. `e%ncode@d` -> `encoded`).

The detector used to pick its own best cleaning internally and hand back one
string. It already computed several candidate cleanings, so it now returns them
all and lets the scorer arbitrate -- same information, no internal guess.
"""

import re
from typing import List

from .base import Family, Layer
from ..scoring import english_word_score, score_text

NOISE_CHARS = set("#@$%&*?!()[]<>+=^~|\\/;`'\"")
MAX_CANDIDATES = 4


class SymbolNoiseDetector(Layer):
    name = "SymbolNoise"
    priority = 72
    family = Family.STRUCTURAL

    def _clean_candidates(self, text: str) -> List[str]:
        if len(text) < 5:
            return []
        results: List[str] = []

        strip_all = "".join(c for c in text if c not in NOISE_CHARS)
        if strip_all != text:
            results.append(strip_all)

        embedded = re.sub(
            r"(?<=[a-zA-Z0-9])[#@$%&*?!()<>+=^~|\\/]+(?=[a-zA-Z0-9])", "", text
        )
        embedded = re.sub(
            r"(?<=\s)[#@$%&*?!()<>+=^~|\\/]+(?=[a-zA-Z0-9])", "", embedded
        )
        embedded = re.sub(
            r"(?<=[a-zA-Z0-9])[#@$%&*?!()<>+=^~|\\/]+(?=\s|$)", "", embedded
        )
        if embedded != text and embedded not in results:
            results.append(embedded)

        # Greedily drop one symbol at a time while the word score climbs.
        curr = text
        symbols = {c for c in curr if not c.isalnum() and c not in " \t\r\n.,;:\"-"}
        curr_score = english_word_score(curr)
        improved = True
        while improved:
            improved = False
            best_sym, best_sc, best_cand = None, curr_score, None
            for sym in symbols:
                cand = curr.replace(sym, "")
                sc = english_word_score(cand)
                if sc > best_sc + 0.02:
                    best_sc, best_cand, best_sym = sc, cand, sym
            if best_cand is not None:
                curr, curr_score = best_cand, best_sc
                symbols.remove(best_sym)
                improved = True
        if curr != text and curr not in results:
            results.append(curr)

        return results

    def detect(self, data: bytes) -> float:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return 0.0
        orig_words = english_word_score(text)
        cands = self._clean_candidates(text)
        if not cands:
            return 0.0
        best = max(
            cands,
            key=lambda s: (
                english_word_score(s),
                -sum(1 for c in s if c in NOISE_CHARS),
                score_text(s),
            ),
        )
        best_words = english_word_score(best)
        if best_words >= 0.40 and best_words > orig_words + 0.05:
            return 0.85
        if score_text(best) > score_text(text) + 0.10 and best_words > orig_words:
            return 0.75
        return 0.0

    def decode(self, data: bytes) -> List[bytes]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return []
        cands = self._clean_candidates(text)
        if not cands:
            return []
        orig_words = english_word_score(text)
        orig_score = score_text(text)
        ranked = sorted(
            cands,
            key=lambda s: (
                english_word_score(s),
                -sum(1 for c in s if c in NOISE_CHARS),
                score_text(s),
            ),
            reverse=True,
        )
        keep = [
            c for c in ranked
            if english_word_score(c) > orig_words or score_text(c) > orig_score
        ]
        return [c.encode("utf-8") for c in keep[:MAX_CANDIDATES]]
