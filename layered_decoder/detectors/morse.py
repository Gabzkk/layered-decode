"""Morse code. Inherently ASCII, but still routed through bytes so the Layer
interface stays uniform."""

import re
from typing import List, Tuple

from .base import Family, Layer

MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3',
    '....-': '4', '.....': '5', '-....': '6', '--...': '7',
    '---..': '8', '----.': '9', '-----': '0', '--..--': ',',
    '.-.-.-': '.', '..--..': '?', '-..-.': '/', '-....-': '-',
    '-.--.': '(', '-.--.-': ')'
}


class MorseDetector(Layer):
    name = "Morse"
    priority = 40
    family = Family.STRUCTURAL

    def detect(self, data: bytes) -> float:
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError:
            return 0.0
        chars = set(text)
        if chars.issubset({'.', '-', ' ', '/'}):
            if "." in chars or "-" in chars:
                return 0.9
        return 0.0

    def decode(self, data: bytes) -> List[bytes]:
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError:
            return []
        words = text.split(" / ") if " / " in text else text.split("   ")
        decoded = []
        for word in words:
            decoded.append("".join(MORSE_CODE_DICT.get(c, "?") for c in word.split()))
        return [" ".join(decoded).encode("ascii", errors="replace")]

    def match_ratio(self, data: bytes) -> float:
        if not data:
            return 0.0
        charset = set(".-/ \t\r\n")
        return sum(1 for c in data.decode("latin1") if c in charset) / len(data)

    def find_matching_spans(self, data: bytes) -> List[Tuple[int, int, float]]:
        text = data.decode("latin1")
        return [
            (m.start(), m.end(), 0.9)
            for m in re.finditer(r"[.\-/ ]{12,}", text)
            if "." in m.group(0) or "-" in m.group(0)
        ]
