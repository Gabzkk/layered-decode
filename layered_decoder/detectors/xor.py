"""Single-byte XOR -- family B (statistical).

There is no signature for XOR. You cannot detect the key, and you cannot
partially detect it; the only correct strategy is to try all 255 non-identity
keys and let the scorer rank them. `detect` therefore returns a fixed moderate
confidence, and `decode` returns every key's output.

This is the layer that most needs a bytes-native pipeline. XOR output is
arbitrary binary: 250 of the 255 candidates are not valid UTF-8 at all, and a
str-based scorer would either crash on them or silently mangle them via
latin-1 before ranking.
"""

from typing import List

from .base import Family, Layer

# Family B has no signature, so this is a constant. It sits above the engine's
# detection floor but below any real structural match, so a confident Base64 or
# Hex hit always outranks the brute-force spray.
STATISTICAL_CONFIDENCE = 0.5


class XorSingleByteLayer(Layer):
    name = "XorSingleByte"
    priority = 80
    family = Family.STATISTICAL

    def detect(self, data: bytes) -> float:
        if not data:
            return 0.0
        return STATISTICAL_CONFIDENCE

    def decode(self, data: bytes) -> List[bytes]:
        if not data:
            return []
        # Key 0 is the identity and can never be a layer.
        return [bytes(b ^ key for b in data) for key in range(1, 256)]
