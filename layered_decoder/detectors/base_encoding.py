"""Base64 / Base64Url / Base32.

Decoding yields raw bytes with no UTF-8 requirement and no printability check.
Both of those used to live here and both belong to the scorer now: a base64
wrapper around a gzip blob decodes to bytes no text decoder accepts, and
rejecting it here is what previously forced a separate Gzip detector to
re-implement base64.
"""

import base64
import re
import string
from typing import List, Tuple

from .base import Family, Layer

_B64_STD = re.compile(rb"^[A-Za-z0-9+/]+={0,2}$")
_B64_URL = re.compile(rb"^[A-Za-z0-9\-_]+={0,2}$")
_B32 = re.compile(rb"^[A-Z2-7]+={0,6}$")


class Base64Detector(Layer):
    name = "Base64"
    priority = 10
    family = Family.STRUCTURAL

    def detect(self, data: bytes) -> float:
        stripped = data.strip()
        if not stripped or len(stripped) % 4 == 1:
            return 0.0
        if not _B64_STD.match(stripped):
            return 0.0
        return 0.9

    def decode(self, data: bytes) -> List[bytes]:
        stripped = data.strip()
        if not stripped:
            return []
        stripped += b"=" * (-len(stripped) % 4)
        try:
            return [base64.b64decode(stripped, validate=True)]
        except Exception:
            return []

    def match_ratio(self, data: bytes) -> float:
        if not data:
            return 0.0
        charset = set(string.ascii_letters + string.digits + "+/=")
        return sum(1 for c in data.decode("latin1") if c in charset) / len(data)

    def find_matching_spans(self, data: bytes) -> List[Tuple[int, int, float]]:
        text = data.decode("latin1")
        return [
            (m.start(), m.end(), 0.9)
            for m in re.finditer(r"[A-Za-z0-9+/]{12,}={0,2}", text)
        ]


class Base64UrlDetector(Layer):
    name = "Base64Url"
    priority = 11
    family = Family.STRUCTURAL

    def detect(self, data: bytes) -> float:
        stripped = data.strip()
        if not stripped or len(stripped) % 4 == 1:
            return 0.0
        if not _B64_URL.match(stripped):
            return 0.0
        return 0.9

    def decode(self, data: bytes) -> List[bytes]:
        stripped = data.strip()
        if not stripped:
            return []
        stripped += b"=" * (-len(stripped) % 4)
        try:
            return [base64.urlsafe_b64decode(stripped)]
        except Exception:
            return []

    def match_ratio(self, data: bytes) -> float:
        if not data:
            return 0.0
        charset = set(string.ascii_letters + string.digits + "-_=")
        return sum(1 for c in data.decode("latin1") if c in charset) / len(data)

    def find_matching_spans(self, data: bytes) -> List[Tuple[int, int, float]]:
        text = data.decode("latin1")
        return [
            (m.start(), m.end(), 0.9)
            for m in re.finditer(r"[A-Za-z0-9\-_]{12,}={0,2}", text)
        ]


class Base32Detector(Layer):
    name = "Base32"
    priority = 12
    family = Family.STRUCTURAL

    def detect(self, data: bytes) -> float:
        stripped = data.strip()
        if not stripped or not _B32.match(stripped.upper()):
            return 0.0
        return 0.8

    def decode(self, data: bytes) -> List[bytes]:
        stripped = data.strip().upper()
        if not stripped:
            return []
        padding = len(stripped) % 8
        if padding:
            stripped += b"=" * (8 - padding)
        try:
            return [base64.b32decode(stripped)]
        except Exception:
            return []

    def match_ratio(self, data: bytes) -> float:
        if not data:
            return 0.0
        charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567=")
        return sum(1 for c in data.decode("latin1").upper() if c in charset) / len(data)

    def find_matching_spans(self, data: bytes) -> List[Tuple[int, int, float]]:
        text = data.decode("latin1").upper()
        return [
            (m.start(), m.end(), 0.8)
            for m in re.finditer(r"[A-Z2-7]{12,}={0,6}", text)
        ]
