"""Gzip / zlib decompression.

Keeps the 8 MiB output cap. This is a real defence, not decoration: the beam
explores far more states than the old greedy loop did, so a decompression bomb
would be reached more often and decompressed more times.

Accepts both raw compressed bytes and base64-wrapped ones. The base64 path is
now partly redundant -- Base64 yields raw bytes and the beam carries them --
but it lets a single layer recognise "base64 of gzip" in one step, which is
cheaper than paying for the intermediate state.
"""

import base64
import gzip
import io
import zlib
from typing import List

from .base import Family, Layer

MAX_DECOMPRESSED_BYTES = 8 * 1024 * 1024


def _looks_like_zlib(raw: bytes) -> bool:
    if len(raw) < 2:
        return False
    cmf, flg = raw[0], raw[1]
    return (cmf & 0x0F) == 8 and (cmf >> 4) <= 7 and ((cmf << 8) + flg) % 31 == 0


def _decompress_limited(raw: bytes, method: str) -> bytes | None:
    """Decompress, refusing anything over the cap.

    Reads MAX+1 and rejects on overflow -- reading exactly MAX would silently
    truncate a bomb into a plausible-looking partial result.
    """
    try:
        if method == "gzip":
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as archive:
                output = archive.read(MAX_DECOMPRESSED_BYTES + 1)
        else:
            decompressor = zlib.decompressobj()
            output = decompressor.decompress(raw, MAX_DECOMPRESSED_BYTES + 1)
            output += decompressor.flush(MAX_DECOMPRESSED_BYTES + 1 - len(output))
        return output if len(output) <= MAX_DECOMPRESSED_BYTES else None
    except (EOFError, OSError, ValueError, zlib.error):
        return None


class GzipDetector(Layer):
    name = "Gzip/Zlib"
    priority = 5
    family = Family.STRUCTURAL

    @staticmethod
    def _raw_candidates(data: bytes) -> List[bytes]:
        candidates = []
        stripped = data.strip()
        if stripped:
            try:
                candidates.append(base64.b64decode(stripped, validate=True))
            except (ValueError, TypeError):
                pass
        candidates.append(data)
        return candidates

    def detect(self, data: bytes) -> float:
        for raw in self._raw_candidates(data):
            if raw.startswith(b"\x1f\x8b"):
                return 0.9 if raw is not data else 0.8
            if _looks_like_zlib(raw):
                return 0.85 if raw is not data else 0.75
        return 0.1

    def decode(self, data: bytes) -> List[bytes]:
        for raw in self._raw_candidates(data):
            method = "gzip" if raw.startswith(b"\x1f\x8b") else "zlib"
            decoded = _decompress_limited(raw, method)
            if decoded is not None:
                return [decoded]
        return []
