"""Vigenere / repeating-key XOR -- family B (statistical).

Brute force over 26^k is hopeless past k=2, so this does the standard two-step
instead:

1. Index of coincidence guesses the key *length*. Split the ciphertext into k
   columns; the correct k (and its multiples) push average column IoC toward
   English's ~0.065, while wrong k sit near random's ~0.038.
2. Once k is known, each column is an *independent* Caesar shift, so solve each
   by chi-squared frequency fit against English letter frequencies. That is k
   evaluations, not 26^k.

Family B rules still hold: no signature, fixed moderate confidence, and every
plausible key length returned as a separate candidate for the scorer to rank.
"""

import math
from typing import List

from .base import Family, Layer
from .xor import STATISTICAL_CONFIDENCE

MAX_KEY_LENGTH = 16
# Number of plausible key lengths to return. Small multiples of the true
# length also score high on IoC, so returning a few lets the scorer arbitrate.
TOP_KEY_LENGTHS = 3

# Minimum letters before per-column IoC carries any signal at all.
MIN_LETTERS = 24

# IoC calibration. Random text sits ~0.038, English ~0.065. Anything at or below
# the floor is structureless bytes rather than a cipher, and must not be offered.
IOC_FLOOR = 0.042
IOC_SCALE = 18.0

# Relative English letter frequencies (percent).
ENGLISH_FREQ = (
    8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153,
    0.772, 4.025, 2.406, 6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056,
    2.758, 0.978, 2.360, 0.150, 1.974, 0.074,
)

# IoC thresholds: random text sits ~0.038, English ~0.065.
IOC_RANDOM = 0.038
IOC_ENGLISH = 0.060

# log of relative English letter frequency, for scoring decrypted text.
_LOG_FREQ = tuple(math.log(freq / 100.0) for freq in ENGLISH_FREQ)


def _is_letter(b: int) -> bool:
    return ord("a") <= b <= ord("z") or ord("A") <= b <= ord("Z")


def _index_of_coincidence(sequence) -> float:
    """IoC over letters only.

    N counts *letters*, not sequence length. Including spaces in the
    denominator biases every column downward and destroys the English-vs-random
    separation the key-length guess depends on.
    """
    counts = [0] * 26
    total = 0
    for b in sequence:
        if ord("a") <= b <= ord("z"):
            counts[b - ord("a")] += 1
            total += 1
        elif ord("A") <= b <= ord("Z"):
            counts[b - ord("A")] += 1
            total += 1
    if total < 2:
        return 0.0
    return sum(c * (c - 1) for c in counts) / (total * (total - 1))


def _average_column_ioc(data: bytes, k: int) -> float:
    """Mean column IoC for key length `k`.

    Columns are cut by *position in the original data*, skipping non-letters
    without consuming a key slot. Filtering the bytes first would shift every
    column and silently decorrelate the real key positions.
    """
    if k < 1:
        return 0.0
    columns: list[list[int]] = [[] for _ in range(k)]
    for i, b in enumerate(data):
        if _is_letter(b):
            columns[i % k].append(b)
    usable = [c for c in columns if len(c) >= 2]
    if not usable:
        return 0.0
    return sum(_index_of_coincidence(col) for col in usable) / len(usable)


def _solve_column(column: bytes) -> int:
    """Caesar shift whose decryption best matches English letter frequencies.

    Scores the *decrypted* text directly -- summed log-frequency over the
    recovered letters -- rather than chi-squared against the ciphertext
    distribution. On the short columns a Vigenere key produces (often fewer than
    30 letters), the ciphertext distribution is nearly flat, so chi-squared is
    dominated by sampling noise and picks confidently wrong shifts. Decrypting
    first and comparing to expected frequencies is stable at this size and costs
    the same.
    """
    letters = [b for b in column if _is_letter(b)]
    if not letters:
        return 0

    best_shift, best_fit = 0, float("-inf")
    for shift in range(26):
        fit = 0.0
        for b in letters:
            base = ord("a") if b <= ord("z") else ord("A")
            index = (b - base - shift) % 26
            fit += _LOG_FREQ[index]
        if fit > best_fit:
            best_fit, best_shift = fit, shift
    return best_shift


class VigenereBruteForceLayer(Layer):
    name = "Vigenere"
    priority = 81
    family = Family.STATISTICAL

    def _candidate_key_lengths(self, data: bytes) -> List[int]:
        """Plausible key lengths, smallest first.

        A correct k and its multiples all score high on IoC, and a multiple is
        *not* interchangeable with the true length: key positions are assigned
        by `i % k` over the raw index, so k=6 splits `i % 3` alignment and
        decrypts to garbage even when the recovered shift values look right.
        So the smallest strong candidates are all returned and the scorer picks
        whichever actually yields English.
        """
        n_letters = sum(1 for b in data if _is_letter(b))
        if n_letters < MIN_LETTERS:
            # Below this a column holds too few letters for IoC to mean
            # anything, and every k scores as noise.
            return []
        limit = min(MAX_KEY_LENGTH, max(1, n_letters // 6))
        scored = [(_average_column_ioc(data, k), k) for k in range(1, limit + 1)]
        strong = [(ioc, k) for ioc, k in scored if ioc >= IOC_RANDOM + 0.010]
        if not strong:
            strong = scored
        best = max(ioc for ioc, _ in strong)
        near_ties = sorted(k for ioc, k in strong if ioc >= best - 0.020)
        return near_ties[:TOP_KEY_LENGTHS]

    def detect(self, data: bytes) -> float:
        """Confidence that this looks like polyalphabetic ciphertext.

        Not a constant. A flat "maybe" tells the search nothing and lets the layer
        fire on arbitrary text; the discriminating statistic is the best average
        column index of coincidence, since the correct key length is the one that
        lifts columns toward English (~0.065) while wrong lengths sit near random
        (~0.038).

        This is still evidence, not identification. Caesar is the k=1 case of this
        same family, and any monoalphabetic text also has a high IoC, so nothing
        here can distinguish Vigenere from Caesar -- only the scorer, comparing
        the actual decryptions, can do that.
        """
        n_letters = sum(1 for b in data if _is_letter(b))
        if n_letters < MIN_LETTERS:
            return 0.0
        limit = min(MAX_KEY_LENGTH, max(1, n_letters // 6))
        best_ioc = max(
            (_average_column_ioc(data, k) for k in range(1, limit + 1)),
            default=0.0,
        )
        if best_ioc < IOC_FLOOR:
            # Below the random baseline: structureless bytes, not a cipher.
            return 0.1
        # Scaled so a clearly-Vigenere-shaped input lands near the family-B
        # constant and a marginal one sits well below it.
        return min(STATISTICAL_CONFIDENCE, (best_ioc - IOC_FLOOR) * IOC_SCALE)

    def decode(self, data: bytes) -> List[bytes]:
        if not self._candidate_key_lengths(data):
            return []

        out: List[bytes] = []
        for k in self._candidate_key_lengths(data):
            columns: list[list[int]] = [[] for _ in range(k)]
            positions: list[list[int]] = [[] for _ in range(k)]
            for i, b in enumerate(data):
                if _is_letter(b):
                    columns[i % k].append(b)
                    positions[i % k].append(i)
            if not any(len(c) >= 2 for c in columns):
                continue

            key = [_solve_column(bytes(col)) for col in columns]
            if not any(key):
                continue

            shifted = bytearray(data)
            for col, shift in enumerate(key):
                if not shift:
                    continue
                for pos, b in zip(positions[col], columns[col]):
                    base = ord("a") if b <= ord("z") else ord("A")
                    # Subtract: encryption added the shift, so undoing it removes.
                    shifted[pos] = (b - base - shift) % 26 + base
            out.append(bytes(shifted))
        return out
