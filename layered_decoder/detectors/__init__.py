"""Layer registry.

Ordering is a tiebreak only. The beam ranks candidates by score, so a layer's
position here does not decide correctness -- which is the point: decode order is
the reverse of encode order, and the search discovers it rather than having it
hardcoded.
"""

from .base import Family, Layer, StopSignal, BaseDetector
from .base_encoding import Base64Detector, Base64UrlDetector, Base32Detector
from .hex_encoding import HexDetector
from .url_encoding import UrlEncodingDetector
from .morse import MorseDetector
from .binary_octal import BinaryDetector, OctalDetector
from .ascii_decimal import AsciiDecimalDetector
from .compression import GzipDetector
from .reversal import ReversalDetector
from .split_interleave import SplitInterleaveDetector
from .symbol_noise import SymbolNoiseDetector
from .substitution import InterleavedDetector, NthCharNoiseDetector
from .xor import XorSingleByteLayer
from .repeating_key_xor import RepeatingKeyXorLayer
from ..scoring import DEFAULT_FLAG_PREFIXES
from .caesar import CaesarBruteForceLayer
from .vigenere import VigenereBruteForceLayer
from .crypto_guard import CryptoGuardDetector


def get_all_detectors(flag_prefixes: tuple[str, ...] = DEFAULT_FLAG_PREFIXES) -> list[Layer]:
    """Every layer, ordered by priority (lower = considered earlier)."""
    layers: list[Layer] = [
        GzipDetector(),            # 5   structural
        Base64Detector(),          # 10
        Base64UrlDetector(),       # 11
        Base32Detector(),          # 12
        UrlEncodingDetector(),     # 15
        HexDetector(),             # 20
        BinaryDetector(),          # 25
        OctalDetector(),           # 26
        AsciiDecimalDetector(),    # 30
        MorseDetector(),           # 40
        ReversalDetector(),        # 60
        SplitInterleaveDetector(),  # 68
        InterleavedDetector(),     # 70
        SymbolNoiseDetector(),     # 72
        NthCharNoiseDetector(),    # 75
        CaesarBruteForceLayer(),   # 50  statistical
        XorSingleByteLayer(),      # 80  statistical
        VigenereBruteForceLayer(), # 81  statistical
        RepeatingKeyXorLayer(flag_prefixes), # 82 statistical
        CryptoGuardDetector(),     # 100 stop-signal
    ]
    layers.sort(key=lambda layer: layer.priority)
    return layers


def get_stop_signals() -> list[StopSignal]:
    return [l for l in get_all_detectors() if isinstance(l, StopSignal)]


__all__ = [
    "Family", "Layer", "StopSignal", "BaseDetector",
    "get_all_detectors", "get_stop_signals",
    "RepeatingKeyXorLayer",
]
