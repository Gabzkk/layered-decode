"""Compound / concatenated input detection and splitting.

Byte positions from the detectors have to line up with the character-class
changepoint analysis, so both run over the same latin-1 view of the data. latin-1
is a bijection on bytes, which is what keeps the indices aligned -- utf-8 would
shift every index after the first non-ASCII byte.
"""

from __future__ import annotations

import string
from typing import Callable, Dict, List, Optional, Tuple

from .models import DecodeResult, DecodeStep
from .scoring import (
    DEFAULT_FLAG_PREFIXES,
    cached_score,
    english_word_score,
    has_known_patterns,
    score,
)

B64_CHARS = frozenset(string.ascii_letters + string.digits + "+/=")
HEX_CHARS = frozenset("0123456789abcdefABCDEF")
NOISE_CHARS = frozenset("#@$%&*?!()[]<>+=^~|\\/;`'\"")
SPACES = frozenset(" \t\r\n")
PUNCT = frozenset(".,:;_")

# Split points to try. Each one costs two full beam decodes, so this bounds the
# work; the highest-scoring boundaries carry nearly all the signal.
MAX_BOUNDARY_CANDIDATES = 4


def get_character_class(c: str) -> str:
    if c in SPACES:
        return "space"
    if c in NOISE_CHARS:
        return "noise"
    if c in HEX_CHARS:
        return "hex"
    if c in B64_CHARS:
        return "b64"
    if c in PUNCT:
        return "punct"
    return "other"


def compute_class_distribution(s: str) -> Dict[str, float]:
    if not s:
        return {}
    counts: Dict[str, int] = {}
    for c in s:
        cls = get_character_class(c)
        counts[cls] = counts.get(cls, 0) + 1
    total = len(s)
    return {k: v / total for k, v in counts.items()}


def distribution_distance(d1: Dict[str, float], d2: Dict[str, float]) -> float:
    all_keys = set(d1.keys()) | set(d2.keys())
    return sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in all_keys) / 2.0


def detect_compound_indicators(data: bytes, detectors: list) -> Tuple[bool, bool]:
    """Signals that `data` is several payloads concatenated.

    Returns (partial_match_detected, likely_compound_input).
    """
    n = len(data)
    if n < 16:
        return (False, False)

    for d in detectors:
        spans = getattr(d, "find_matching_spans", None)
        if callable(spans):
            for start, end, _ in spans(data):
                span_len = end - start
                if 12 <= span_len <= int(0.85 * n) and (start >= 8 or end <= n - 8):
                    return (True, True)

        ratio_fn = getattr(d, "match_ratio", None)
        if callable(ratio_fn):
            ratio = ratio_fn(data)
            # Lower bound is 0.40, not something smaller: charsets here are
            # nested (hex is a subset of base64), so *any* base64 payload sits
            # near 0.31 hex ratio by chance. A weak ratio is not evidence of a
            # partial match -- it is arithmetic.
            if 0.40 <= ratio <= 0.85:
                return (True, True)

    text = data.decode("latin1")
    window = min(16, max(6, n // 4))
    if window >= 4:
        for i in range(window, n - window, 2):
            left_d = compute_class_distribution(text[i - window:i])
            right_d = compute_class_distribution(text[i:i + window])
            if distribution_distance(left_d, right_d) >= 0.35:
                return (True, True)

    return (False, False)


def find_candidate_boundaries(data: bytes, detectors: list) -> List[int]:
    """Candidate split positions, ranked by character-class changepoint score."""
    n = len(data)
    if n < 16:
        return []

    text = data.decode("latin1")
    window = min(16, max(6, n // 4))

    shift_scores: Dict[int, float] = {}
    for i in range(8, n - 8):
        left = text[max(0, i - window):i]
        right = text[i:min(n, i + window)]
        l_len, r_len = len(left), len(right)
        if not l_len or not r_len:
            continue
        shift_scores[i] = sum(
            abs(
                sum(c in charset for c in left) / l_len
                - sum(c in charset for c in right) / r_len
            )
            for charset in (HEX_CHARS, SPACES, NOISE_CHARS, B64_CHARS)
        )

    span_endpoints: set = set()
    for d in detectors:
        spans = getattr(d, "find_matching_spans", None)
        if not callable(spans):
            continue
        for start, end, _ in spans(data):
            if 8 <= start <= n - 8:
                span_endpoints.add(start)
            if 8 <= end <= n - 8:
                span_endpoints.add(end)
            base_len = ((end - start) // 4) * 4
            if base_len > 0 and 8 <= start + base_len <= n - 8:
                span_endpoints.add(start + base_len)

    scored: List[Tuple[float, int]] = []
    for i, shift in shift_scores.items():
        value = shift
        if i in span_endpoints:
            value += 0.30
        if i % 4 == 0 or (n - i) % 4 == 0:
            value += 0.10
        if i % 2 == 0 or (n - i) % 2 == 0:
            value += 0.05
        scored.append((value, i))

    scored.sort(reverse=True, key=lambda t: t[0])
    # Bounded: each candidate costs two full beam decodes, and the top few
    # boundaries carry almost all of the signal.
    return [idx for _, idx in scored[:MAX_BOUNDARY_CANDIDATES]]


def try_segmentation_decode(
    data: bytes,
    decode_func: Callable[[bytes], DecodeResult],
    detectors: list,
    max_recursion: int = 2,
    flag_prefixes: tuple[str, ...] = DEFAULT_FLAG_PREFIXES,
) -> Optional[DecodeResult]:
    """Search for boundaries, decode each side, recombine if both succeed."""
    n = len(data)
    if n < 16 or max_recursion <= 0:
        return None

    candidates = find_candidate_boundaries(data, detectors)
    if not candidates:
        return None

    best_split: Optional[Tuple[int, DecodeResult, DecodeResult]] = None
    best_score = -1.0

    for p in candidates:
        res_l = decode_func(data[:p])
        res_r = decode_func(data[p:])

        comp_l = res_l.status == "solved"
        comp_r = res_r.status == "solved"
        score_l = cached_score(res_l.final_bytes, flag_prefixes)
        score_r = cached_score(res_r.final_bytes, flag_prefixes)
        words_l = english_word_score(res_l.final_output)
        words_r = english_word_score(res_r.final_output)
        pat_l = bool(has_known_patterns(res_l.final_output))
        pat_r = bool(has_known_patterns(res_r.final_output))

        combined_quality = (
            (4.0 if comp_l else 0.0)
            + (4.0 if comp_r else 0.0)
            + (2.0 if pat_l else 0.0)
            + (2.0 if pat_r else 0.0)
            + (words_l * 2.0)
            + (words_r * 2.0)
            + score_l
            + score_r
        )

        valid_split = (
            (comp_l and comp_r)
            or (
                (comp_l or comp_r)
                and (len(res_l.steps) > 0 or len(res_r.steps) > 0)
                and (score_l >= 0.40 and score_r >= 0.40)
            )
            or (
                len(res_l.steps) > 0
                and len(res_r.steps) > 0
                and (words_l >= 0.35 or words_r >= 0.35)
                and (score_l >= 0.40 and score_r >= 0.40)
            )
        )

        if valid_split and combined_quality > best_score:
            best_score = combined_quality
            best_split = (p, res_l, res_r)

        if comp_l and comp_r and (words_l >= 0.4 or words_r >= 0.4 or pat_l or pat_r):
            break

    if best_split is None:
        return None

    p, res_l, res_r = best_split

    all_segments: List[DecodeResult] = []
    for part, res in ((data[:p], res_l), (data[p:], res_r)):
        if res.status != "solved" and len(part) >= 24 and max_recursion > 1:
            sub = try_segmentation_decode(part, decode_func, detectors, max_recursion - 1, flag_prefixes)
            if sub and sub.segments:
                all_segments.extend(sub.segments)
            else:
                all_segments.append(res)
        else:
            all_segments.append(res)

    final_outputs = [seg.final_output for seg in all_segments]
    all_steps: List[DecodeStep] = [s for seg in all_segments for s in seg.steps]
    layers_detected = [
        f"Segment {idx + 1}: {' -> '.join(seg.layers_detected)}"
        for idx, seg in enumerate(all_segments)
    ]

    all_solved = all(seg.status == "solved" for seg in all_segments)

    return DecodeResult(
        layers_detected=layers_detected,
        intermediate_values=[_render(data)] + final_outputs,
        steps=all_steps,
        branches=None,
        final_output="\n".join(final_outputs),
        final_bytes=b"\n".join(seg.final_bytes for seg in all_segments),
        confidence="high" if all_solved else "medium",
        status="solved" if all_solved else "partial_decode_low_confidence",
        stopped_reason="completed" if all_solved else "segmented_completed",
        partial_match_detected=True,
        likely_compound_input=True,
        segments=all_segments,
    )


def _render(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="backslashreplace")
