"""Split strategies for interleaved streams.

Operates on bytes to match the rest of the pipeline; the recombined halves are
joined with a space because they are independent payloads, not adjacent bytes.
"""

from typing import Callable, List, Optional, Tuple

from .models import BranchResult, DecodeResult
from .scoring import DEFAULT_FLAG_PREFIXES, cached_score_text


class BranchStrategy:
    """Different ways to split input into branches."""

    @staticmethod
    def odd_even_split(data: bytes) -> Tuple[bytes, bytes]:
        return data[0::2], data[1::2]

    @staticmethod
    def half_split(data: bytes) -> Tuple[bytes, bytes]:
        mid = len(data) // 2
        return data[:mid], data[mid:]

    @staticmethod
    def interleave(a: bytes, b: bytes) -> bytes:
        out = bytearray()
        for i in range(max(len(a), len(b))):
            if i < len(a):
                out.append(a[i])
            if i < len(b):
                out.append(b[i])
        return bytes(out)

    @staticmethod
    def concatenate(a: bytes, b: bytes) -> bytes:
        return a + b


def try_branch_decode(
    input_string: bytes,
    decoder_func: Callable[[bytes], DecodeResult],
    min_score_improvement: float = 0.1,
    flag_prefixes: tuple[str, ...] = DEFAULT_FLAG_PREFIXES,
) -> Optional[DecodeResult]:
    """Split, decode each side independently, recombine, keep it only if it wins.

    Branches are joined with a space: they are separate payloads, so
    concatenating them raw would fabricate a word boundary that was never there.
    """
    strategies: List[Tuple[str, Callable, Callable]] = [
        ("odd_even_interleave", BranchStrategy.odd_even_split, BranchStrategy.interleave),
        ("odd_even_concat", BranchStrategy.odd_even_split, BranchStrategy.concatenate),
        ("half_concat", BranchStrategy.half_split, BranchStrategy.concatenate),
    ]

    best_result: Optional[DecodeResult] = None
    best_score = cached_score_text(input_string, flag_prefixes)

    for strategy_name, split_fn, recombine_fn in strategies:
        try:
            part_a, part_b = split_fn(input_string)
            result_a = decoder_func(part_a)
            result_b = decoder_func(part_b)
            recombined = recombine_fn(result_a.final_bytes, result_b.final_bytes)
            combined_score = cached_score_text(recombined, flag_prefixes)
        except Exception:
            continue

        if combined_score <= best_score + min_score_improvement:
            continue

        best_score = combined_score
        branch_a = BranchResult(
            branch_id=f"{strategy_name}_A",
            steps=result_a.steps,
            final_output=result_a.final_output,
        )
        branch_b = BranchResult(
            branch_id=f"{strategy_name}_B",
            steps=result_b.steps,
            final_output=result_b.final_output,
        )
        best_result = DecodeResult(
            layers_detected=[f"branch_split:{strategy_name}"]
            + result_a.layers_detected
            + result_b.layers_detected,
            intermediate_values=[
                _render(input_string), _render(part_a), _render(part_b),
                result_a.final_output, result_b.final_output, _render(recombined),
            ],
            steps=result_a.steps + result_b.steps,
            branches=[branch_a, branch_b],
            final_output=_render(recombined),
            final_bytes=recombined,
            confidence="medium",
            status="partial_decode_low_confidence",
            stopped_reason="branch_recombined",
        )

    return best_result


def _render(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="backslashreplace")
