"""Recognize encrypted envelopes without claiming to authenticate ciphertext."""
from dataclasses import dataclass


@dataclass(frozen=True)
class KeyRequirement:
    format_name: str
    evidence: str
    required_info: str


def identify_key_requirement(data: bytes) -> KeyRequirement | None:
    if data.startswith(b'Salted__') and len(data) > 16:
        return KeyRequirement(
            'OpenSSL salted',
            'Salted__ magic header followed by salt/payload bytes; '
            'the cipher and key derivation settings are not encoded in this header.',
            'Original password plus cipher, key derivation method, digest, '
            'iteration count and salt length used by the producer.',
        )

    # A version byte alone is too weak. Require the entire binary layout and
    # a timestamp representable as a calendar date (through year 9999).
    if (
        len(data) >= 73
        and data[0] == 0x80
        and (len(data) - 57) % 16 == 0
        and 0 < int.from_bytes(data[1:9], 'big') <= 253402300799
    ):
        return KeyRequirement(
            'Fernet',
            'Version 0x80, plausible 8-byte timestamp, 16-byte IV, '
            'block-aligned ciphertext and 32-byte HMAC field match the Fernet layout; '
            'the HMAC has not been verified.',
            'Original URL-safe Base64-encoded 32-byte Fernet key.',
        )
    return None
