"""The `Layer` interface: one contract for every decodable transformation.

Everything is `bytes` in and `bytes` out. `str` is a *rendering* concern and
belongs only at the very end (cli.py). Coercing early breaks the pipeline in two
ways: XOR output is arbitrary binary and raises on `.decode('utf-8')`, and
base64-wrapped gzip decodes to bytes no text decoder will accept.

Two families, because they need genuinely different detection strategies:

STRUCTURAL  Pattern-detectable from the bytes alone; exactly one decode.
            Base64, Hex, Reversal, ...  `detect` does real work: charset,
            length and padding checks.

STATISTICAL  No fixed signature. You cannot identify the key before trying it.
            XOR, Caesar, Vigenere. `detect` returns a fixed moderate
            confidence -- you cannot rule them out, and you only know after
            scoring -- while `decode` returns *every* candidate for the scorer
            to rank. detect and decode deliberately collapse into one
            brute-force-and-score step. Do not try to write a clever XOR
            detector that recovers the key first; there isn't one.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class Family(str, Enum):
    STRUCTURAL = "structural"
    STATISTICAL = "statistical"

    @property
    def is_statistical(self) -> bool:
        return self is Family.STATISTICAL


@dataclass(frozen=True)
class CandidateEvidence:
    key: Optional[bytes] = None
    low_confidence_short_ciphertext: bool = False


class Layer(ABC):
    """One decodable transformation.

    Subclasses set `name`, `priority` (lower = considered earlier; a tiebreak
    only, since the beam ranks by score rather than by order), and `family`.
    """

    name: str
    priority: int
    family: Family = Family.STRUCTURAL

    @abstractmethod
    def detect(self, data: bytes) -> float:
        """Confidence 0.0-1.0 that this layer applies to `data`.

        For STATISTICAL layers this is intentionally a constant: absence of a
        signature is not evidence of absence.
        """

    @abstractmethod
    def decode(self, data: bytes) -> List[bytes]:
        """Candidate decoded outputs.

        STRUCTURAL layers return zero or one candidate. STATISTICAL layers
        return all of them (256 XOR keys, 26 Caesar shifts) and let the scorer
        decide.
        """

    def detect_and_decode(self, data: bytes) -> Tuple[float, List[bytes]]:
        conf = self.detect(data)
        if conf <= 0.0:
            return (0.0, [])
        return (conf, self.decode(data))

    def candidate_evidence(self, data: bytes, output: bytes) -> CandidateEvidence:
        """Optional evidence for an individual output, independent of detect()."""
        return CandidateEvidence()

    # ── optional hooks, used by segmentation for compound-input splitting ──

    def match_ratio(self, data: bytes) -> float:
        """Fraction of `data` matching this layer's charset or pattern.

        Defaults to 0.0, not `detect(data)`. A charset ratio is a different
        question from "does this layer apply", and only layers that genuinely
        have a charset answer it -- they override this. Defaulting to `detect`
        made every statistical layer report a match ratio of 0.5, and since
        compound detection treats 0.25-0.85 as "looks concatenated", *every*
        input looked compound and single payloads got split.
        """
        return 0.0

    def find_matching_spans(self, data: bytes) -> List[Tuple[int, int, float]]:
        """Localized (start, end, confidence) matches. Empty if not supported."""
        return []


class StopSignal(Layer):
    """A layer that never decodes -- it only reports that we should stop.

    CryptoGuard is the motivating case: refuse to pretend that brute-forcing a
    real cipher counts as solving it.
    """

    family = Family.STRUCTURAL

    @abstractmethod
    def detect(self, data: bytes) -> float: ...

    def decode(self, data: bytes) -> List[bytes]:
        return []


# Back-compat alias. The class was renamed BaseDetector -> Layer when the
# interface became bytes-native; existing imports keep working.
BaseDetector = Layer


def as_text(data: bytes) -> Optional[str]:
    """UTF-8 view of `data` for layers that must analyse text to decide.

    Returns None for non-UTF-8 input. Such a layer should decline rather than
    guess -- except where a non-text decode is still meaningful, in which case
    work from `data` directly instead of calling this.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None
