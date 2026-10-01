"""Tests for crypto envelope decryption and brute-forcing."""
import base64
import gzip
import hashlib
import json
import subprocess
import tempfile

import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from layered_decoder import LayeredDecoder
from layered_decoder.crypto_decryptor import (
    bruteforce_envelope,
    continue_with_decrypted,
    decrypt_envelope,
    decrypt_fernet,
    decrypt_openssl,
    evp_bytes_to_key,
    load_wordlist,
)
from tests.test_cli import run_cli


def _make_fernet(payload: bytes, key_or_password: str, is_derived: bool = False) -> tuple[bytes, str]:
    if is_derived:
        key = base64.urlsafe_b64encode(hashlib.sha256(key_or_password.encode()).digest())
        token = Fernet(key).encrypt(payload)
    else:
        key = key_or_password.encode()
        token = Fernet(key).encrypt(payload)
    raw = base64.urlsafe_b64decode(token)
    return raw, key_or_password


def _make_openssl(payload: bytes, password: str, kdf: str = "pbkdf2", cipher: str = "aes-256-cbc") -> bytes:
    cmd = ["openssl", "enc", f"-{cipher}", "-salt", "-pass", f"pass:{password}"]
    if kdf == "pbkdf2":
        cmd.append("-pbkdf2")
    else:
        cmd.extend(["-md", "md5"])
    p = subprocess.run(cmd, input=payload, stdout=subprocess.PIPE, check=True)
    return p.stdout


# ── Fernet Unit Tests ──────────────────────────────────────────────

def test_fernet_decrypt_raw_key():
    raw_key = Fernet.generate_key().decode()
    raw_f, _ = _make_fernet(b"H4G{fernet_raw_key_flag}", raw_key, is_derived=False)
    pt, method = decrypt_fernet(raw_f, raw_key)
    assert pt == b"H4G{fernet_raw_key_flag}"
    assert "raw key" in method


def test_fernet_decrypt_derived_password():
    raw_f, _ = _make_fernet(b"H4G{fernet_derived_flag}", "masterkey", is_derived=True)
    pt, method = decrypt_fernet(raw_f, "masterkey")
    assert pt == b"H4G{fernet_derived_flag}"
    assert "SHA256 password" in method


def test_fernet_decrypt_invalid_key():
    raw_f, _ = _make_fernet(b"H4G{valid}", "correct_pass", is_derived=True)
    res = decrypt_fernet(raw_f, "wrong_pass")
    assert res is None


def test_fernet_bruteforce_common_passwords():
    # 'secret' is in COMMON_PASSWORDS
    raw_f, _ = _make_fernet(b"H4G{fernet_cracked}", "secret", is_derived=True)
    res = bruteforce_envelope(raw_f, "Fernet")
    assert res is not None
    pt, matched_pw, method = res
    assert pt == b"H4G{fernet_cracked}"
    assert matched_pw == "secret"


def test_fernet_bruteforce_custom_wordlist():
    raw_f, _ = _make_fernet(b"H4G{custom_wordlist_hit}", "custom123_pass", is_derived=True)
    res = bruteforce_envelope(raw_f, "Fernet", wordlist=["wrongA", "wrongB", "custom123_pass"])
    assert res is not None
    assert res[1] == "custom123_pass"
    assert res[0] == b"H4G{custom_wordlist_hit}"


# ── OpenSSL Unit Tests ─────────────────────────────────────────────

def test_openssl_decrypt_pbkdf2():
    enc = _make_openssl(b"H4G{openssl_pbkdf2_flag}", "secret123", kdf="pbkdf2", cipher="aes-256-cbc")
    pt, method = decrypt_openssl(enc, "secret123")
    assert pt == b"H4G{openssl_pbkdf2_flag}"
    assert "pbkdf2" in method


def test_openssl_decrypt_evp_md5():
    enc = _make_openssl(b"H4G{openssl_evp_flag}", "legacy_pass", kdf="evp", cipher="aes-256-cbc")
    pt, method = decrypt_openssl(enc, "legacy_pass")
    assert pt == b"H4G{openssl_evp_flag}"
    assert "evp-md5" in method


def test_openssl_decrypt_invalid_password():
    enc = _make_openssl(b"H4G{valid}", "correct", kdf="pbkdf2")
    res = decrypt_openssl(enc, "wrong_guess")
    assert res is None


def test_openssl_bruteforce_common_passwords():
    # 'dragon' is in COMMON_PASSWORDS
    enc = _make_openssl(b"H4G{openssl_cracked}", "dragon", kdf="pbkdf2")
    res = bruteforce_envelope(enc, "OpenSSL salted")
    assert res is not None
    pt, matched_pw, method = res
    assert pt == b"H4G{openssl_cracked}"
    assert matched_pw == "dragon"


def test_openssl_bruteforce_wordlist_file(tmp_path):
    wordlist_file = tmp_path / "wordlist.txt"
    wordlist_file.write_text("cand1\ncand2\nsupersecret_phrase\ncand3\n")
    enc = _make_openssl(b"H4G{wordlist_hit}", "supersecret_phrase", kdf="pbkdf2")
    res = bruteforce_envelope(enc, "OpenSSL salted", wordlist=str(wordlist_file))
    assert res is not None
    assert res[1] == "supersecret_phrase"
    assert res[0] == b"H4G{wordlist_hit}"


# ── Engine Integration Tests ───────────────────────────────────────

def test_engine_auto_decrypt_with_key():
    # Stack: Base64 -> Fernet(key="supersecret") -> Caesar(+3) -> flag
    flag = b"H4G{stacked_fernet_caesar_flag}"
    shifted = bytes(
        (b - ord("A") + 3) % 26 + ord("A") if ord("A") <= b <= ord("Z")
        else (b - ord("a") + 3) % 26 + ord("a") if ord("a") <= b <= ord("z")
        else b for b in flag
    )
    raw_f, _ = _make_fernet(shifted, "supersecret", is_derived=True)
    payload = base64.b64encode(raw_f).decode()

    decoder = LayeredDecoder(flag_prefixes=("H4G{",))

    # Without key: stops at key_required
    r_locked = decoder.decode(payload)
    assert r_locked.key_required is True
    assert r_locked.encryption_format == "Fernet"

    # With key: automatically decrypts and continues through Caesar to solved!
    r_unlocked = decoder.decode(payload, key="supersecret")
    assert r_unlocked.key_required is False
    assert r_unlocked.status == "solved"
    assert r_unlocked.final_output == "H4G{stacked_fernet_caesar_flag}"
    assert any("Decryption" in layer for layer in r_unlocked.layers_detected)


def test_engine_auto_decrypt_with_bruteforce():
    # 'admin' is in common passwords
    enc = _make_openssl(b"H4G{cracked_openssl_stacked}", "admin", kdf="pbkdf2")
    payload = base64.b64encode(enc).decode()

    decoder = LayeredDecoder(flag_prefixes=("H4G{",))

    # Without bruteforce: key_required
    r_locked = decoder.decode(payload)
    assert r_locked.key_required is True

    # With bruteforce: solved!
    r_unlocked = decoder.decode(payload, bruteforce=True)
    assert r_unlocked.key_required is False
    assert r_unlocked.status == "solved"
    assert r_unlocked.final_output == "H4G{cracked_openssl_stacked}"


# ── CLI Tests ──────────────────────────────────────────────────────

def test_cli_with_key_flag():
    enc = _make_openssl(b"H4G{cli_key_flag_test}", "letmein", kdf="pbkdf2")
    payload = base64.b64encode(enc).decode()

    # Pass -k letmein
    output = run_cli("-k", "letmein", "-i", payload)
    assert "H4G{cli_key_flag_test}" in output
    assert "Decryption (OpenSSL" in output

    # Quiet mode with -k
    quiet_output = run_cli("-q", "-k", "letmein", "-i", payload)
    assert quiet_output.strip() == "H4G{cli_key_flag_test}"


def test_cli_with_bruteforce_flag():
    enc = _make_openssl(b"H4G{cli_bruteforce_test}", "qwerty", kdf="pbkdf2")
    payload = base64.b64encode(enc).decode()

    output = run_cli("-B", "-i", payload)
    assert "H4G{cli_bruteforce_test}" in output
    assert "Decryption (OpenSSL" in output


def test_cli_json_with_key():
    enc = _make_openssl(b"H4G{json_key_test}", "password123", kdf="pbkdf2")
    payload = base64.b64encode(enc).decode()

    output = run_cli("--json", "-k", "password123", "-i", payload)
    data = json.loads(output)
    assert data["key_required"] is False
    assert data["status"] == "solved"
    assert data["final_output"] == "H4G{json_key_test}"
    assert any("Decryption" in s["encoding_detected"] for s in data["steps"])
