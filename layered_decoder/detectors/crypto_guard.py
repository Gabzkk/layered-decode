"""CryptoGuard -- a stop-signal, not a decoder.

Refuses to let the beam claim it brute-forced real cryptography. Everything
here is a *veto*: high confidence means "stop, this is not an encoding problem".

Two details worth preserving if this is ever rewritten:

* Entropy threshold is sample-size aware (`log2(min(len, 256))`). A fixed
  threshold misfires badly on short inputs, where Shannon entropy cannot
  physically reach 8 bits/byte.
* gzip/zlib headers are excluded explicitly. Compressed data is high-entropy
  but is an *encoding*, and treating it as crypto would abort every legitimate
  base64-gzip payload.
"""

import base64
import math
import re
import string
from collections import Counter

from .base import StopSignal

_B64 = re.compile(rb"^[A-Za-z0-9+/]+={0,2}$")
_B64_URL = re.compile(rb"^[A-Za-z0-9\-_]+={0,2}$")
_B32 = re.compile(rb"^[A-Z2-7]+={0,6}$")
_HEXISH = re.compile(rb"^[0-9a-fA-F\s:]+$")
_SEP = re.compile(rb"[\s:]+")


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    length = len(data)
    counts = Counter(data)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def _chi_squared_uniformity(data: bytes) -> float:
    """Byte-distribution uniformity, normalised so ~1.0 means random."""
    if len(data) < 16:
        return 0.0
    counts = Counter(data)
    expected = len(data) / 256.0
    chi2 = sum((counts.get(i, 0) - expected) ** 2 / expected for i in range(256))
    normalised = max(0.0, 1.0 - (chi2 - 256) / (256 * 10))
    return min(normalised, 1.0)


class CryptoGuardDetector(StopSignal):
    name = "CryptoGuard"
    priority = 100

    @staticmethod
    def _is_encoding_wrapper(data: bytes) -> bool:
        """True if this text is a valid encoding to peel, not ciphertext.

        The guard base64-decodes its input before judging it, so a base64 string
        wrapping XOR'd ciphertext trips it. Because a trip restricts expansion to
        family B, that silently removes Base64 from the available layers and
        makes the wrapper *unpeelable* -- the payload dead-ends at exactly the
        step that would have progressed.

        A wrapper is structure. Judge its contents, not its shell.
        """
        stripped = data.strip()
        if not stripped or len(stripped) % 4 == 1:
            return False
        if _B64.match(stripped) or _B64_URL.match(stripped):
            return True
        upper = stripped.upper()
        if len(upper) % 8 != 1 and _B32.match(upper):
            return True
        compact = _SEP.sub(b"", stripped)
        if compact and len(compact) % 2 == 0 and _HEXISH.match(stripped):
            try:
                bytes.fromhex(compact.decode("ascii"))
                return True
            except ValueError:
                return False
        return False

    def detect(self, data: bytes) -> float:
        if len(data) < 16:
            return 0.0

        if self._is_encoding_wrapper(data):
            # Structure, not ciphertext. Peel it and judge what is inside.
            return 0.0

        try:
            raw: bytes | None = base64.b64decode(data.strip(), validate=True)
        except Exception:
            raw = None
        if raw is None:
            raw = data
        if len(raw) < 16:
            return 0.0

        if raw.startswith(b"\x1f\x8b"):
            return 0.0
        if _looks_like_zlib(raw):
            return 0.0

        entropy = _byte_entropy(raw)
        max_possible = math.log2(min(len(raw), 256))
        entropy_ratio = entropy / max_possible if max_possible > 0 else 0.0

        uniformity = _chi_squared_uniformity(raw)

        try:
            text = raw.decode("utf-8", errors="strict")
            printable = sum(1 for c in text if c in string.printable)
            if len(text) > 0 and printable / len(text) > 0.9:
                return 0.0
        except Exception:
            pass

        if entropy_ratio > 0.85:
            if len(raw) % 16 == 0:
                return 0.95
            return 0.85

        if entropy_ratio > 0.75 and uniformity > 0.3:
            return 0.8

        return 0.0


def _looks_like_zlib(raw: bytes) -> bool:
    if len(raw) < 2:
        return False
    cmf, flg = raw[0], raw[1]
    return (cmf & 0x0F) == 8 and (cmf >> 4) <= 7 and ((cmf << 8) + flg) % 31 == 0
