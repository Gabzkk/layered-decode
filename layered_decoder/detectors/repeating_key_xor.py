"""Repeating-key XOR with strict printable-ASCII elimination and joint scoring."""

from collections import Counter
from heapq import heappop, heappush
from itertools import combinations, product
from math import prod

from .base import CandidateEvidence, Family, Layer
from .xor import STATISTICAL_CONFIDENCE
from ..scoring import DEFAULT_FLAG_PREFIXES, score

MAX_COMBINATIONS = 10_000
BOUNDED_COMBINATIONS = 256
KEY_LENGTH_CANDIDATES = 5

# English percentages: US Naval Academy, SI110 frequency-analysis table.
# https://www.usna.edu/Users/cs/wcbrown/courses/si110AY13S/resources/ceasar-shift/freqAnalysis.html
LETTER_FREQUENCIES = (
    8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966,
    0.153, 0.772, 4.025, 2.406, 6.749, 7.507, 1.929, 0.095, 5.987,
    6.327, 9.056, 2.758, 0.978, 2.360, 0.150, 1.974, 0.074,
)


def hamming_distance(left: bytes, right: bytes) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length blocks")
    return sum((a ^ b).bit_count() for a, b in zip(left, right))


def rank_key_lengths(data: bytes) -> list[tuple[float, int]]:
    ranked = []
    for length in range(2, min(20, len(data) // 4) + 1):
        blocks = [data[i:i + length] for i in range(0, min(len(data) // length, 8) * length, length)]
        distances = [hamming_distance(a, b) / length for a, b in combinations(blocks, 2)]
        ranked.append((sum(distances) / len(distances), length))
    return sorted(ranked)


def printable_keys(column: bytes) -> list[int]:
    values = set(column)
    return [key for key in range(256) if all(32 <= byte ^ key <= 126 for byte in values)]


def column_chi_squared(decoded: bytes) -> float:
    counts = Counter(decoded.lower())
    letters = sum(counts[c] for c in range(97, 123))
    if not letters:
        return float("inf")
    return sum(
        (counts[c] - letters * frequency / 100) ** 2 / (letters * frequency / 100)
        for c, frequency in enumerate(LETTER_FREQUENCIES, 97)
    )


def _bounded_keys(columns: list[list[tuple[float, int]]]):
    """Visit complete keys in ascending total column cost without materializing the product."""
    start = (0,) * len(columns)
    heap = [(sum(col[0][0] for col in columns), start)]
    seen = {start}
    for _ in range(BOUNDED_COMBINATIONS):
        if not heap:
            break
        _, indices = heappop(heap)
        yield bytes(col[i][1] for col, i in zip(columns, indices))
        for axis, col in enumerate(columns):
            if indices[axis] + 1 >= len(col):
                continue
            following = indices[:axis] + (indices[axis] + 1,) + indices[axis + 1:]
            if following in seen:
                continue
            seen.add(following)
            cost = sum(c[i][0] for c, i in zip(columns, following))
            heappush(heap, (cost, following))


def _prefix_columns(
    data: bytes, columns: list[list[tuple[float, int]]], prefix: bytes,
) -> list[list[tuple[float, int]]] | None:
    required = {}
    for i, byte in enumerate(prefix):
        axis = i % len(columns)
        key = data[i] ^ byte
        if axis in required and required[axis] != key:
            return None
        required[axis] = key
    constrained = [
        [entry for entry in col if axis not in required or entry[1] == required[axis]]
        for axis, col in enumerate(columns)
    ]
    return constrained if all(constrained) else None


class RepeatingKeyXorLayer(Layer):
    name = "RepeatingKeyXor"
    priority = 82
    family = Family.STATISTICAL

    def __init__(self, flag_prefixes: tuple[str, ...] = DEFAULT_FLAG_PREFIXES):
        self.flag_prefixes = tuple(flag_prefixes)

    def detect(self, data: bytes) -> float:
        return STATISTICAL_CONFIDENCE if len(data) >= 8 else 0.0

    def decode(self, data: bytes) -> list[bytes]:
        survivors = []
        # Eliminate across the ranked list before frequency scoring; impossible
        # lengths must not consume the five candidate slots on short inputs.
        for _, length in rank_key_lengths(data):
            columns = [data[i::length] for i in range(length)]
            keys = [printable_keys(column) for column in columns]
            if all(keys):
                survivors.append((columns, keys))
            if len(survivors) == KEY_LENGTH_CANDIDATES:
                break

        results = {}
        for columns, keys in survivors:
            decoded_columns = [
                {key: bytes(b ^ key for b in col) for key in options}
                for col, options in zip(columns, keys)
            ]
            ranked = [
                sorted((column_chi_squared(decoded), key) for key, decoded in col.items())
                for col in decoded_columns
            ]
            if prod(map(len, keys)) <= MAX_COMBINATIONS:
                candidates = (bytes(key) for key in product(*keys))
                searches = [candidates]
            else:
                searches = [_bounded_keys(ranked)]
                # Preserve exact-prefix hypotheses even if rare flag characters
                # put their column keys outside the ordinary frequency shortlist.
                for prefix in self.flag_prefixes:
                    raw = prefix.encode("utf-8")
                    if not raw or len(raw) > len(data):
                        continue
                    constrained = _prefix_columns(data, ranked, raw)
                    if constrained:
                        searches.append(_bounded_keys(constrained))
            best = None
            best_score = float("-inf")
            seen_keys = set()
            buffer = bytearray(len(data))
            for search in searches:
                for key in search:
                    if len(set(key)) == 1:  # Already covered by XorSingleByte.
                        continue
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    for i, byte in enumerate(key):
                        buffer[i::len(key)] = decoded_columns[i][byte]
                    decoded = bytes(buffer)
                    value = score(decoded, self.flag_prefixes)
                    if value > best_score:
                        best, best_score = decoded, value
            if best is not None:
                results[best] = best_score
        return sorted(results, key=results.get, reverse=True)

    def candidate_evidence(self, data: bytes, output: bytes) -> CandidateEvidence:
        stream = bytes(a ^ b for a, b in zip(data, output))
        for length in range(1, min(20, len(stream)) + 1):
            key = stream[:length]
            if all(byte == key[i % length] for i, byte in enumerate(stream)):
                return CandidateEvidence(key, len(data) < 10 * length)
        raise ValueError("Output is not a repeating-key XOR candidate")
