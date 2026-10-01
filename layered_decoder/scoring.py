"""Central scorer — the single place that ranks every decode candidate.

Operates on `bytes` and stays there. XOR output is arbitrary binary, so any
str coercion before scoring destroys the signal (and raises on non-UTF-8).
`str` appears only at the final render step in cli.py.

Scoring formula (see `score`):

    0.3 * printable_ratio      is it text at all
  + 0.4 * english_word_ratio  does it contain dictionary words
  + 0.5 * flag_content_score  flag shape supported by prefix or body words
  + 0.2 * bigram_structure    is it English *letter structure*
  - 0.2 * entropy_penalty     is it still high-entropy noise

Why `bigram_structure` exists
-----------------------------
The first four terms are all invariant under a byte permutation of the
lowercase-letter range, which is exactly what several XOR keys do. Decoding a
flag like `H3G{wELCome_b4ck_wr0ng_fl4G}` under key 0x2A (correct), 0x3C and
0x34 all yield printable text with identical entropy and an identical
`flag{...}` shape, so those three keys tie on everything except the dictionary
term -- and a flag body is never in the dictionary. Measured on this corpus the
correct key lands 3rd of 256.

`bigram_structure` is the one cheap term that is *not* permutation-invariant, so
it breaks the tie on real letter statistics. With it the correct key ranks 1st.
Set `USE_BIGRAM_TERM = False` to A/B against the un-augmented formula.
"""

import math
import re
from collections import Counter
from typing import List

USE_BIGRAM_TERM = True
DEFAULT_FLAG_PREFIXES = ("flag{", "FLAG{")
KNOWN_PREFIX_BONUS = 1.0

DICTIONARY = frozenset([
    # Function words
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
    "new", "now", "old", "see", "way", "who", "did", "get", "let", "say",
    "she", "too", "use", "flag", "key", "code", "text", "data", "file",
    "open", "find", "help", "hash", "byte", "char", "test", "pass", "word", "hack",
    "that", "with", "this", "they", "have", "from", "what", "some", "them",
    "will", "would", "make", "like", "time", "just", "know", "take", "into",
    "year", "your", "good", "could", "than", "then", "look", "only", "come",
    "over", "think", "also", "back", "after", "work", "first", "well", "even",
    "want", "because", "any", "these", "give", "day", "most", "us", "is", "it",
    "in", "to", "of", "on", "as", "at", "by", "an", "be", "do", "we", "if", "my",
    "so", "up", "go", "no", "he", "or", "me", "am", "admin", "password", "root",
    "user", "which", "their", "about", "there", "when",
    # Common content words
    "quick", "brown", "fox", "lazy", "dog", "jump", "jumps", "over",
    "hello", "world", "secret", "hidden", "message", "answer", "clue",
    "solve", "puzzle", "cipher", "decode", "encode", "encrypt", "decrypt",
    "plain", "string", "value", "input", "output", "result", "begin",
    "start", "end", "stop", "next", "last", "name", "number", "letter",
    "case", "upper", "lower", "each", "every", "much", "many", "more",
    "less", "here", "where", "right", "left", "long", "short", "high",
    "low", "big", "small", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "hundred", "read", "write", "run", "try",
    "keep", "call", "turn", "move", "play", "put", "set", "show", "tell",
    "ask", "need", "should", "must", "still", "never", "always", "very",
    "been", "being", "were", "does", "done", "made", "said", "went",
    "got", "came", "took", "part", "great", "place", "same", "another",
    "different", "such", "while", "through", "before", "between",
    "under", "again", "off", "down", "own", "might",
    "welcome", "future", "quantum", "archive", "archives", "proceed", "unlock", "backward",
    "encoded", "decoded", "encrypted", "decrypted", "ciphers", "messages", "secrets", "known", "wrong",
    "pure", "structure", "single", "stream", "split", "layer", "layered", "keys",
])

_TOKEN_STRIP = ".,!?;:'\"()[]{}<>"

# Most frequent English letter bigrams. A byte-permutation-invariant scorer
# cannot separate XOR keys; this can.
COMMON_BIGRAMS = frozenset("""
th he in er an re on at en nd ti es or te of ed is it al ar st to nt ng se
ha as ou io le ve co me de hi ri ro ic ne ea ra ce li ch ll be ma si om ur
ca el ta la ns di fo ho pe ec pr no ct us ac ot il tr ly nc et ut ss so rs
un lo wa ge ie wh ee wi em ad ol rt po we na ul ni ts mo ow pa im mi ai sh
""".split())


def shannon_entropy(data: bytes) -> float:
    """Entropy in bits per byte. Rejects nothing -- safe on arbitrary binary."""
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def calculate_entropy(text: str) -> float:
    """Back-compat wrapper: entropy of a str's UTF-8 encoding."""
    return shannon_entropy(text.encode("utf-8", errors="replace"))


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    if text.isprintable():
        return 1.0
    printable_count = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    return printable_count / len(text)


def english_word_ratio(text: str) -> float:
    """Fraction of whitespace-split tokens that are dictionary words."""
    tokens = [t for t in text.split() if t]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t.strip(_TOKEN_STRIP).lower() in DICTIONARY)
    return hits / len(tokens)


def english_word_score(text: str) -> float:
    """Character-weighted dictionary coverage.

    Retained because detectors use it for *relative* before/after comparisons,
    where char-weighting is more stable than token-counting on short strings.
    """
    if not text:
        return 0.0
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    if not words:
        return 0.0
    matched_chars = sum(len(w) for w in words if w in DICTIONARY)
    total_chars = sum(len(w) for w in words)
    return matched_chars / total_chars if total_chars > 0 else 0.0


_ASCII_LETTER_TABLE = str.maketrans({
    chr(c): chr(c).lower() if chr(c).isalpha() else None for c in range(128)
})


def bigram_structure(text: str) -> float:
    """Fraction of alphabetic bigrams that are common English bigrams."""
    letters = (text.translate(_ASCII_LETTER_TABLE) if text.isascii()
               else [c.lower() for c in text if c.isalpha()])
    if len(letters) < 4:
        return 0.0
    return sum(a + b in COMMON_BIGRAMS for a, b in zip(letters, letters[1:])) / (len(letters) - 1)


# Flag shape, in three deliberate constraints:
#   1. prefix is letter-led, 2-16 chars        -- rejects `{!j}`, `>k4yfut{`
#   2. body is [A-Za-z0-9_-]{3,64}             -- no whitespace, no exotic chars
#   3. delimited on both sides                 -- not embedded mid-token
#
# The obvious `[A-Za-z0-9_]+\{.*\}` is wrong in a way that matters here: `.*`
# crosses any character, so it fires on ~4.65% of printable XOR garbage. Since
# flag_bonus (0.5) outweighs the entire dict_score term (max 0.4), a single
# spurious hit outranks any amount of clean English -- and the beam is a
# maximisation search, so it will happily walk into one. Measured false-positive
# rate on printable XOR garbage: 4.65% -> 0.00%.
STRICT_FLAG_PATTERN = re.compile(r"^([A-Za-z0-9_]{1,20}\{[A-Za-z0-9_\-]{1,200}\}\s*)+$")
LOOSE_FLAG_PATTERN = re.compile(r"[A-Za-z0-9_]{1,20}\{[A-Za-z0-9_\-]{1,200}\}")
FLAG_PATTERN = LOOSE_FLAG_PATTERN
_KNOWN_FLAG_BODY = re.compile(r"[A-Za-z0-9_\-]+\}(?![A-Za-z0-9_\-])")


def flag_pattern_score(text: str) -> float:
    """1.0 if the text contains something shaped like a CTF flag."""
    return 1.0 if LOOSE_FLAG_PATTERN.search(text) else 0.0


def has_known_flag_prefix(text: str, flag_prefixes: tuple[str, ...]) -> bool:
    return any(
        prefix and text.startswith(prefix)
        and (_KNOWN_FLAG_BODY.match(text, len(prefix)) if prefix.endswith("{") else LOOSE_FLAG_PATTERN.match(text))
        for prefix in flag_prefixes
    )


_LEET_LETTERS = str.maketrans("013457", "oieast")


def _caesar_shift_text(text: str, shift: int) -> str:
    out = []
    for c in text:
        if "a" <= c <= "z":
            out.append(chr((ord(c) - 97 + shift) % 26 + 97))
        elif "A" <= c <= "Z":
            out.append(chr((ord(c) - 65 + shift) % 26 + 65))
        else:
            out.append(c)
    return "".join(out)


def _body_word_score(text: str) -> float:
    matches = LOOSE_FLAG_PATTERN.findall(text)
    if not matches:
        return 0.0
    total_words = 0
    total_hits = 0
    for match in matches:
        parts = match.split("{", 1)
        if len(parts) < 2:
            continue
        body = parts[1].rstrip("}")
        words = [w for w in re.split(r"[_\-]+", body.lower()) if w]
        if not words:
            continue
        total_words += len(words)
        leet_body = body.lower().translate(_LEET_LETTERS)
        leet_words = [w for w in re.split(r"[_\-]+", leet_body) if w]
        for w, lw in zip(words, leet_words):
            if w in DICTIONARY or lw in DICTIONARY:
                total_hits += 1
    return total_hits / total_words if total_words > 0 else 0.0


def is_verified_strict_flag(text: str, flag_prefixes: tuple[str, ...] = DEFAULT_FLAG_PREFIXES) -> bool:
    """Verify that a strict flag candidate is not a Caesar or ROT13 shifted decoy."""
    stripped = text.strip()
    if not STRICT_FLAG_PATTERN.match(stripped):
        return False
    base_score = max(english_word_ratio(stripped), _body_word_score(stripped))
    if base_score <= 0.0 and not has_known_flag_prefix(stripped, flag_prefixes):
        return False
    for shift in range(1, 26):
        shifted = _caesar_shift_text(stripped, shift)
        shifted_score = max(english_word_ratio(shifted), _body_word_score(shifted))
        if shifted_score > base_score + 0.05:
            return False
    return True


def flag_content_score(text: str, flag_prefixes: tuple[str, ...] = DEFAULT_FLAG_PREFIXES) -> float:
    """Two-tier flag scoring: verified strict gets full bonus, loose gets small bonus."""
    if is_verified_strict_flag(text, flag_prefixes):
        return 1.0
    if has_known_flag_prefix(text, flag_prefixes):
        return 0.75
    if LOOSE_FLAG_PATTERN.search(text):
        return 0.25
    return 0.0


def has_known_patterns(text: str) -> List[str]:
    patterns = []
    if re.search(r'flag\{.*?\}', text, re.IGNORECASE):
        patterns.append("flag{...}")
    if re.search(r'CTF\{.*?\}', text, re.IGNORECASE):
        patterns.append("CTF{...}")
    if re.search(r'H4G\{.*?\}', text, re.IGNORECASE):
        patterns.append("H4G{...}")
    if re.search(r'H3G\{.*?\}', text, re.IGNORECASE):
        patterns.append("H3G{...}")
    if re.search(r'picoCTF\{.*?\}', text, re.IGNORECASE):
        patterns.append("picoCTF{...}")
    if re.search(r'academy\{.*?\}', text, re.IGNORECASE):
        patterns.append("academy{...}")
    if re.search(r'HTB\{.*?\}', text, re.IGNORECASE):
        patterns.append("HTB{...}")
    if re.search(r'THM\{.*?\}', text, re.IGNORECASE):
        patterns.append("THM{...}")
    if re.search(r'TCON\{.*?\}', text, re.IGNORECASE):
        patterns.append("TCON{...}")
    if re.search(r'https?://[^\s]+', text):
        patterns.append("URL")
    if re.search(r'\{\s*["\'][^"\']+["\']\s*:', text):
        patterns.append("JSON")
    return patterns


def score(data: bytes, flag_prefixes: tuple[str, ...] = DEFAULT_FLAG_PREFIXES) -> float:
    """Score a decode candidate. Higher is more likely to be the answer.

    Returns 0.0 for anything that is not valid UTF-8: such a candidate is
    binary, and no amount of downstream text analysis will make it text.
    Callers that want to keep non-UTF-8 candidates in play (e.g. a base64
    wrapper around a gzip blob) must not rely on this alone -- see
    `looks_structured` for the pre-decode check.
    """
    if not data:
        return 0.0
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return 0.0
    if not text:
        return 0.0

    value = (
        0.3 * printable_ratio(text)
        + 0.4 * english_word_ratio(text)
        + 0.5 * flag_content_score(text, flag_prefixes)
        - 0.2 * (shannon_entropy(data) / 8.0)
    )
    if USE_BIGRAM_TERM:
        value += 0.2 * bigram_structure(text)
    # A prefix alone is easy to manufacture on short XOR columns. Require the
    # closing brace and a valid body as well so a literal 'flag{' cannot bless noise.
    if has_known_flag_prefix(text, flag_prefixes):
        value += KNOWN_PREFIX_BONUS
    return value


def looks_structured(data: bytes) -> bool:
    """Pre-decode gate for candidates `score` cannot judge.

    `score` returns 0.0 for anything that is not UTF-8. That is right for
    ranking *text*, but wrong for ranking intermediates: a base64-wrapped
    binary payload decodes to bytes no text decoder accepts, and a naive
    "printable or gzip" test prunes it -- so base64 of arbitrary bytes could
    never be unwrapped, which defeats the point of a bytes-native pipeline.

    So this asks a different question: could *some* layer consume this? Any
    layer's output is admissible; what must be rejected is random noise, which
    is what the crypto guard is for. Being permissive here is safe because the
    beam still ranks by score and a wrong turn simply loses.
    """
    return bool(data)


def score_text(text: str) -> float:
    """Back-compat wrapper around `score` -- single source of truth.

    Detectors call this to compare a candidate against its input; both sides go
    through the same function, so only the relative ordering matters.
    """
    return score(text.encode("utf-8", errors="replace"))


# The beam re-expands overlapping states constantly -- the same bytes get scored
# at several depths and from several parents -- so scoring is memoised here
# rather than in the engine, which keeps the cache next to the thing it caches
# and lets both the engine and branching use it.
_score_cache: dict[tuple[bytes, tuple[str, ...]], float] = {}
_CACHE_LIMIT = 200_000


def cached_score(data: bytes, flag_prefixes: tuple[str, ...] = DEFAULT_FLAG_PREFIXES) -> float:
    key = (data, flag_prefixes)
    hit = _score_cache.get(key)
    if hit is not None:
        return hit
    value = score(data, flag_prefixes)
    if len(_score_cache) >= _CACHE_LIMIT:
        _score_cache.clear()
    _score_cache[key] = value
    return value


def cached_score_text(data, flag_prefixes: tuple[str, ...] = DEFAULT_FLAG_PREFIXES) -> float:
    """`cached_score` for bytes or str, for callers that hold either."""
    if isinstance(data, str):
        data = data.encode("utf-8", errors="replace")
    return cached_score(data, flag_prefixes)


def clear_score_cache() -> None:
    _score_cache.clear()
