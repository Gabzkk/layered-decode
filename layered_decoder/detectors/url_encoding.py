"""URL percent-encoding. Operates on bytes via unquote_to_bytes so that
percent-encoded binary survives intact."""

import re
import urllib.parse
from typing import List, Tuple

from .base import Family, Layer

_ESCAPE = re.compile(rb"%[0-9a-fA-F]{2}")


class UrlEncodingDetector(Layer):
    name = "UrlEncoding"
    priority = 15
    family = Family.STRUCTURAL

    def detect(self, data: bytes) -> float:
        return 0.85 if _ESCAPE.search(data) else 0.0

    def decode(self, data: bytes) -> List[bytes]:
        if not _ESCAPE.search(data):
            return []
        try:
            return [urllib.parse.unquote_to_bytes(data.decode("latin1"))]
        except Exception:
            return []

    def match_ratio(self, data: bytes) -> float:
        if not data:
            return 0.0
        escapes = len(_ESCAPE.findall(data))
        return min(1.0, (escapes * 3) / len(data))

    def find_matching_spans(self, data: bytes) -> List[Tuple[int, int, float]]:
        text = data.decode("latin1")
        return [
            (m.start(), m.end(), 0.85)
            for m in re.finditer(r"(?:%[0-9a-fA-F]{2}[A-Za-z0-9_\-\.~]*){2,}", text)
        ]
