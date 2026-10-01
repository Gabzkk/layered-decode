"""Decryption and brute-force cracking for recognized cryptographic envelopes."""
from __future__ import annotations

import base64
import hashlib
import os
import re
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .key_requirement import KeyRequirement, identify_key_requirement
from .models import DecodeResult, DecodeStep
from .scoring import cached_score, shannon_entropy


# ── Common Dictionary Passwords ────────────────────────────────────
COMMON_PASSWORDS: list[str] = [
    # Top general passwords
    "password", "123456", "12345678", "123456789", "12345", "qwerty",
    "admin", "secret", "root", "toor", "guest", "system", "master",
    "welcome", "letmein", "hunter2", "iloveyou", "dragon", "shadow",
    "supersecret", "password123", "p@ssword", "pass123", "pass", "test",
    # CTF / Security / Cryptography themes
    "ctf", "flag", "key", "cipher", "crypto", "hackme", "challenge",
    "security", "matrix", "future", "archive", "quantum", "portal",
    "terminal", "decoder", "caesar", "secretkey", "passphrase", "access",
    "open", "unlock", "private", "hidden", "admin123", "root123", "testing",
    "demo", "magic", "token", "payload", "agent", "binary", "reverse",
    "exploit", "hacker", "cyber", "freedom", "trustnoone",
    "illuminati", "classified", "confidential", "topsecret",
]


def load_wordlist(wordlist_source: str | Sequence[str] | None) -> list[str]:
    """Load a list of candidate words from a file path or sequence."""
    if wordlist_source is None:
        return []
    if isinstance(wordlist_source, str):
        if os.path.isfile(wordlist_source):
            try:
                with open(wordlist_source, "r", encoding="utf-8", errors="ignore") as fh:
                    return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
            except Exception:
                return []
        else:
            return [wordlist_source]
    return list(wordlist_source)


def evp_bytes_to_key(
    password: bytes,
    salt: bytes,
    key_len: int,
    iv_len: int,
    hash_name: str = "md5",
    count: int = 1,
) -> tuple[bytes, bytes]:
    """Derive key and IV using OpenSSL legacy EVP_BytesToKey."""
    d = b""
    d_i = b""
    while len(d) < key_len + iv_len:
        h = hashlib.new(hash_name)
        if d_i:
            h.update(d_i)
        h.update(password)
        h.update(salt)
        d_i = h.digest()
        for _ in range(1, count):
            d_i = hashlib.new(hash_name, d_i).digest()
        d += d_i
    return d[:key_len], d[key_len : key_len + iv_len]


def is_likely_decrypted_payload(data: bytes) -> bool:
    """Validate decrypted plaintext against random garbage from false PKCS7 padding."""
    if not data:
        return False

    # Check for valid compression streams
    if data.startswith(b"\x1f\x8b"):
        import gzip
        try:
            gzip.decompress(data)
            return True
        except Exception:
            pass

    if data.startswith((b"\x78\x01", b"\x78\x9c", b"\x78\xda")):
        import zlib
        try:
            zlib.decompress(data)
            return True
        except Exception:
            pass

    # Common binary / archive magic
    if data.startswith((b"PK\x03\x04", b"\x7fELF", b"\x89PNG", b"GIF8")):
        return True

    # CTF Flag signatures
    if re.search(rb"(?i)(flag|ctf|h4g|h3g|htb|thm)\{[^}]+\}", data):
        return True

    # Printable text ratio (text, base64, hex, json, urls, etc.)
    printable = sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13))
    ratio = printable / len(data)
    if ratio >= 0.70:
        return True

    return False


def decrypt_fernet(data: bytes, key_or_password: str) -> tuple[bytes, str] | None:
    """Decrypt a raw Fernet binary payload (starts with 0x80)."""
    if not (len(data) >= 73 and data[0] == 0x80):
        return None

    token = base64.urlsafe_b64encode(data)
    pw_bytes = key_or_password.encode("utf-8")

    # 1. Try as exact raw Fernet key (32 bytes urlsafe-base64 = 44 chars)
    try:
        f = Fernet(pw_bytes)
        pt = f.decrypt(token)
        return pt, "Fernet (raw key)"
    except (ValueError, InvalidToken, Exception):
        pass

    # 2. Try as password -> SHA-256 derived Fernet key
    try:
        derived = base64.urlsafe_b64encode(hashlib.sha256(pw_bytes).digest())
        f = Fernet(derived)
        pt = f.decrypt(token)
        return pt, "Fernet (SHA256 password)"
    except (ValueError, InvalidToken, Exception):
        pass

    # 3. Try as password -> MD5 derived Fernet key (16 bytes repeated to 32)
    try:
        md5_digest = hashlib.md5(pw_bytes).digest()
        derived_md5 = base64.urlsafe_b64encode(md5_digest * 2)
        f = Fernet(derived_md5)
        pt = f.decrypt(token)
        return pt, "Fernet (MD5 password)"
    except (ValueError, InvalidToken, Exception):
        pass

    # 4. Try as raw 32-byte key encoded to urlsafe base64
    if len(pw_bytes) == 32:
        try:
            raw32_key = base64.urlsafe_b64encode(pw_bytes)
            f = Fernet(raw32_key)
            pt = f.decrypt(token)
            return pt, "Fernet (32-byte key)"
        except (ValueError, InvalidToken, Exception):
            pass

    return None


def decrypt_openssl(data: bytes, password: str) -> tuple[bytes, str] | None:
    """Decrypt OpenSSL salted ciphertext (starts with Salted__)."""
    if not (data.startswith(b"Salted__") and len(data) >= 32):
        return None

    salt = data[8:16]
    ciphertext = data[16:]
    pw_bytes = password.encode("utf-8")

    # Common OpenSSL configurations in order of popularity
    configs: list[tuple[str, int, int, str, str, int]] = [
        # (cipher_name, key_len, iv_len, kdf_type, hash_name, iterations)
        ("aes-256-cbc", 32, 16, "pbkdf2", "sha256", 10000),
        ("aes-256-cbc", 32, 16, "evp", "md5", 1),
        ("aes-128-cbc", 16, 16, "pbkdf2", "sha256", 10000),
        ("aes-128-cbc", 16, 16, "evp", "md5", 1),
        ("aes-256-cbc", 32, 16, "evp", "sha256", 1),
        ("aes-128-cbc", 16, 16, "evp", "sha256", 1),
        ("aes-192-cbc", 24, 16, "pbkdf2", "sha256", 10000),
        ("aes-192-cbc", 24, 16, "evp", "md5", 1),
        ("aes-256-cbc", 32, 16, "pbkdf2", "sha1", 10000),
    ]

    for cipher_name, klen, ivlen, kdf, hname, iters in configs:
        try:
            if kdf == "pbkdf2":
                key_iv = hashlib.pbkdf2_hmac(hname, pw_bytes, salt, iters, klen + ivlen)
                key, iv = key_iv[:klen], key_iv[klen : klen + ivlen]
            else:
                key, iv = evp_bytes_to_key(pw_bytes, salt, klen, ivlen, hname, iters)

            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            pt_padded = decryptor.update(ciphertext) + decryptor.finalize()

            pad = pt_padded[-1]
            if 1 <= pad <= 16 and pt_padded[-pad:] == bytes([pad]) * pad:
                pt = pt_padded[:-pad]
                if is_likely_decrypted_payload(pt):
                    return pt, f"OpenSSL {cipher_name} ({kdf}-{hname})"
        except Exception:
            continue

    return None


def decrypt_envelope(data: bytes, format_name: str | None, key_or_password: str) -> tuple[bytes, str] | None:
    """Attempt decryption of an envelope with the given key/password."""
    if not format_name:
        req = identify_key_requirement(data)
        if req:
            format_name = req.format_name

    if format_name == "Fernet":
        return decrypt_fernet(data, key_or_password)
    elif format_name == "OpenSSL salted":
        return decrypt_openssl(data, key_or_password)
    else:
        # Try both
        res = decrypt_fernet(data, key_or_password)
        if res:
            return res
        return decrypt_openssl(data, key_or_password)


def bruteforce_envelope(
    data: bytes,
    format_name: str | None = None,
    wordlist: Sequence[str] | str | None = None,
    contextual_words: Sequence[str] | None = None,
    max_attempts: int = 50000,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[bytes, str, str] | None:
    """Brute-force crack an encrypted envelope using candidate passwords.

    Returns (decrypted_bytes, matched_password, method_name) or None.
    """
    if not format_name:
        req = identify_key_requirement(data)
        if req:
            format_name = req.format_name

    candidates: list[str] = []
    seen: set[str] = set()

    def add_cand(w: str) -> None:
        w_clean = w.strip()
        if w_clean and w_clean not in seen:
            seen.add(w_clean)
            candidates.append(w_clean)

    # 1. Add contextual words from earlier layers or input strings
    if contextual_words:
        for text in contextual_words:
            # Extract words from text
            tokens = re.findall(r"[a-zA-Z0-9_\-]+", text)
            for tok in tokens:
                add_cand(tok)
                add_cand(tok.lower())

    # 2. Add user wordlist or default common passwords
    loaded_words = load_wordlist(wordlist)
    if loaded_words:
        for w in loaded_words:
            add_cand(w)
    else:
        for w in COMMON_PASSWORDS:
            add_cand(w)

    total = min(len(candidates), max_attempts)
    for idx, cand in enumerate(candidates[:total]):
        if on_progress and (idx % 100 == 0 or idx == total - 1):
            on_progress(idx + 1, total, cand)

        res = decrypt_envelope(data, format_name, cand)
        if res is not None:
            pt, method = res
            return pt, cand, method

    return None


def continue_with_decrypted(
    decoder: "LayeredDecoder",
    prior_result: DecodeResult,
    decrypted_bytes: bytes,
    used_key: str,
    method_name: str,
    allow_segmentation: bool = True,
) -> DecodeResult:
    """Stitch a successful decryption step into prior_result and continue peeling inner layers."""
    score_before = (
        prior_result.steps[-1].score_after
        if prior_result.steps
        else cached_score(prior_result.final_bytes, decoder.flag_prefixes)
    )
    score_after = cached_score(decrypted_bytes, decoder.flag_prefixes)
    entropy_before = shannon_entropy(prior_result.final_bytes)
    entropy_after = shannon_entropy(decrypted_bytes)

    def _render_bytes(b: bytes) -> str:
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return b.decode("utf-8", errors="backslashreplace")

    input_val = (
        prior_result.steps[-1].output_value
        if prior_result.steps
        else _render_bytes(prior_result.final_bytes)
    )
    output_val = _render_bytes(decrypted_bytes)

    decrypt_step = DecodeStep(
        layer_number=len(prior_result.steps) + 1,
        encoding_detected=f"Decryption ({method_name})",
        confidence=1.0,
        input_value=input_val,
        output_value=output_val,
        score_before=score_before,
        score_after=score_after,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        key_hex=used_key,
    )

    remaining = decoder.decode(decrypted_bytes, allow_segmentation=allow_segmentation)

    offset = len(prior_result.steps) + 1
    for step in remaining.steps:
        step.layer_number += offset

    combined_steps = list(prior_result.steps) + [decrypt_step] + list(remaining.steps)
    combined_layers = (
        list(prior_result.layers_detected)
        + [f"Decryption ({method_name})"]
        + list(remaining.layers_detected)
    )
    combined_intermediate = (
        list(prior_result.intermediate_values)
        + [output_val]
        + (list(remaining.intermediate_values[1:]) if len(remaining.intermediate_values) > 1 else [])
    )

    return DecodeResult(
        key_required=remaining.key_required,
        encryption_format=remaining.encryption_format,
        key_requirement_evidence=remaining.key_requirement_evidence,
        required_decryption_info=remaining.required_decryption_info,
        layers_detected=combined_layers,
        intermediate_values=combined_intermediate,
        steps=combined_steps,
        branches=remaining.branches,
        final_output=remaining.final_output,
        final_bytes=remaining.final_bytes,
        confidence=remaining.confidence,
        status=remaining.status,
        stopped_reason=remaining.stopped_reason,
        partial_match_detected=remaining.partial_match_detected,
        likely_compound_input=remaining.likely_compound_input,
        segments=remaining.segments,
        low_confidence_short_ciphertext=remaining.low_confidence_short_ciphertext,
    )

