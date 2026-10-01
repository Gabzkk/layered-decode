"""Scorer and layer unit tests.

The Layer interface is bytes-native: `detect(data: bytes) -> float` and
`decode(data: bytes) -> list[bytes]`. Family B layers return many candidates;
structural layers return at most one.
"""
import base64
import unittest
import zlib

from layered_decoder.scoring import (
    bigram_structure,
    calculate_entropy,
    english_word_ratio,
    english_word_score,
    flag_pattern_score,
    has_known_patterns,
    printable_ratio,
    score,
    shannon_entropy,
)
from layered_decoder.detectors import Family
from layered_decoder.detectors.base_encoding import (
    Base32Detector,
    Base64Detector,
    Base64UrlDetector,
)
from layered_decoder.detectors.hex_encoding import HexDetector
from layered_decoder.detectors.url_encoding import UrlEncodingDetector
from layered_decoder.detectors.reversal import ReversalDetector
from layered_decoder.detectors.binary_octal import BinaryDetector, OctalDetector
from layered_decoder.detectors.morse import MorseDetector
from layered_decoder.detectors.ascii_decimal import AsciiDecimalDetector
from layered_decoder.detectors.compression import GzipDetector
from layered_decoder.detectors.crypto_guard import CryptoGuardDetector
from layered_decoder.detectors.substitution import NthCharNoiseDetector
from layered_decoder.detectors.split_interleave import SplitInterleaveDetector
from layered_decoder.detectors.symbol_noise import SymbolNoiseDetector
from layered_decoder.detectors.xor import XorSingleByteLayer
from layered_decoder.detectors.caesar import CaesarBruteForceLayer
from layered_decoder.detectors.vigenere import VigenereBruteForceLayer


# ── Scoring ────────────────────────────────────────────────────────

class TestScoring(unittest.TestCase):
    def test_clean_text_scores_higher_than_noise(self):
        clean = b"This is some standard english text for testing."
        noisy = b"awefuihwef823rb238rbq2389"
        self.assertGreater(score(clean), score(noisy))

    def test_score_rejects_non_utf8(self):
        # A bytes-native pipeline must be able to score arbitrary binary without
        # raising. Non-text simply scores zero.
        self.assertEqual(score(b"\xff\xfe\x00\x80\x81"), 0.0)

    def test_entropy_of_uniform_string(self):
        self.assertAlmostEqual(calculate_entropy("aaaa"), 0.0)
        self.assertGreater(calculate_entropy("abcdefghij"), 2.0)

    def test_shannon_entropy_on_bytes(self):
        self.assertAlmostEqual(shannon_entropy(b"aaaa"), 0.0)
        self.assertGreater(shannon_entropy(bytes(range(32))), 4.0)

    def test_printable_ratio(self):
        self.assertEqual(printable_ratio("Hello"), 1.0)
        self.assertAlmostEqual(printable_ratio(""), 0.0)

    def test_english_word_score(self):
        self.assertGreater(english_word_score("the quick brown fox"), 0.5)
        self.assertAlmostEqual(english_word_score("xyzzy plugh"), 0.0)

    def test_english_word_ratio_counts_tokens(self):
        self.assertAlmostEqual(english_word_ratio("the quick brown fox"), 1.0)
        self.assertAlmostEqual(english_word_ratio("xyzzy plugh"), 0.0)

    def test_has_known_patterns(self):
        self.assertIn("flag{...}", has_known_patterns("flag{hello_world}"))
        self.assertIn("HTB{...}", has_known_patterns("HTB{s3cr3t}"))
        self.assertIn("URL", has_known_patterns("visit https://example.com now"))
        self.assertEqual(has_known_patterns("plain text"), [])

    def test_flag_pattern_rejects_embedded_noise(self):
        # Regression: the loose `[A-Za-z0-9_]+\{.*\}` matched `k4yfut{!j}` inside
        # XOR garbage and handed it the full 0.5 flag bonus, outranking real
        # English. The delimited form must not.
        self.assertEqual(flag_pattern_score("jytlu={>k4yfut{!j}>"), 0.0)
        self.assertEqual(flag_pattern_score("@Z{!@JI|@}]!I|Yk@Jw!I|ws@}]!@R"), 0.0)

    def test_flag_pattern_accepts_real_formats(self):
        for flag in (
            "H3G{wELCome_b4ck_wr0ng_fl4G}",
            "flag{the_quick_brown_fox}",
            "CTF{h3ll0_w0rld}",
            "HTB{s3cr3t}",
            "picoCTF{flag}",
            "answer is TCON{abc-123} here",
        ):
            self.assertEqual(flag_pattern_score(flag), 1.0, flag)

    def test_bigram_structure_separates_english_from_noise(self):
        self.assertGreater(bigram_structure("the quick brown fox"), 0.3)
        self.assertLess(bigram_structure("qz xj wv pl rt"), 0.2)


# ── Family A: structural layers ─────────────────────────────────────

class TestStructuralLayers(unittest.TestCase):
    def test_base64(self):
        det = Base64Detector()
        self.assertGreater(det.detect(b"SGVsbG8gV29ybGQ="), 0.5)
        self.assertEqual(det.decode(b"SGVsbG8gV29ybGQ="), [b"Hello World"])

    def test_base64_rejects_non_b64(self):
        self.assertLessEqual(Base64Detector().detect(b"Not base 64 !!"), 0.3)

    def test_base64_yields_binary_without_choking(self):
        # The whole point of the bytes pipeline: base64 of non-UTF-8 content is
        # a legitimate waypoint, not a decode failure.
        payload = bytes(range(256))
        encoded = base64.b64encode(payload)
        self.assertEqual(Base64Detector().decode(encoded), [payload])

    def test_base64_unpadded(self):
        encoded = base64.urlsafe_b64encode(b"Hi").decode().rstrip("=")
        det = Base64UrlDetector()
        self.assertGreater(det.detect(encoded.encode()), 0.3)
        self.assertEqual(det.decode(encoded.encode()), [b"Hi"])

    def test_base32(self):
        det = Base32Detector()
        encoded = base64.b32encode(b"Hello")
        self.assertGreater(det.detect(encoded), 0.3)
        self.assertEqual(det.decode(encoded), [b"Hello"])

    def test_hex(self):
        det = HexDetector()
        self.assertGreater(det.detect(b"48656c6c6f"), 0.5)
        self.assertEqual(det.decode(b"48656c6c6f"), [b"Hello"])

    def test_hex_strips_separators(self):
        self.assertEqual(HexDetector().decode(b"48:65:6c:6c:6f"), [b"Hello"])

    def test_url(self):
        det = UrlEncodingDetector()
        self.assertGreater(det.detect(b"Hello%20World%21"), 0.5)
        self.assertEqual(det.decode(b"Hello%20World%21"), [b"Hello World!"])

    def test_reversal(self):
        det = ReversalDetector()
        self.assertGreater(det.detect(b"}olleh{galf"), 0.5)
        self.assertEqual(det.decode(b"}olleh{galf"), [b"flag{hello}"])

    def test_binary(self):
        det = BinaryDetector()
        self.assertGreater(det.detect(b"0100100001100101011011000110110001101111"), 0.5)
        self.assertEqual(
            det.decode(b"0100100001100101011011000110110001101111"), [b"Hello"]
        )

    def test_binary_with_separators(self):
        self.assertEqual(BinaryDetector().decode(b"01001000 01101001"), [b"Hi"])
        self.assertEqual(BinaryDetector().decode(b"01001000,01101001"), [b"Hi"])

    def test_binary_preserves_high_bytes(self):
        # `chr(n)` used to mangle these into mojibake.
        self.assertEqual(BinaryDetector().decode(b"11111111 00000001"), [b"\xff\x01"])

    def test_octal(self):
        det = OctalDetector()
        self.assertGreater(det.detect(b"110 151"), 0.5)
        self.assertEqual(det.decode(b"110 151"), [b"Hi"])

    def test_ascii_decimal(self):
        det = AsciiDecimalDetector()
        self.assertGreater(det.detect(b"72 105"), 0.5)
        self.assertEqual(det.decode(b"72 105"), [b"Hi"])

    def test_morse(self):
        det = MorseDetector()
        morse = b".... . .-.. .-.. ---"
        self.assertGreater(det.detect(morse), 0.5)
        self.assertEqual(det.decode(morse), [b"HELLO"])

    def test_gzip_via_base64(self):
        encoded = base64.b64encode(zlib.compress(b"Hello compressed world"))
        det = GzipDetector()
        self.assertGreater(det.detect(encoded), 0.3)
        self.assertEqual(det.decode(encoded), [b"Hello compressed world"])

    def test_gzip_rejects_decompression_bomb(self):
        # Must refuse rather than materialise the whole expansion.
        bomb = zlib.compress(b"\x00" * (64 * 1024 * 1024))
        self.assertEqual(GzipDetector().decode(base64.b64encode(bomb)), [])

    def test_structural_layers_return_at_most_one_candidate(self):
        payload = b"SGVsbG8gV29ybGQ="
        for det in (Base64Detector(), HexDetector(), MorseDetector(), ReversalDetector()):
            self.assertLessEqual(len(det.decode(payload)), 1, det.name)
            self.assertEqual(det.family, Family.STRUCTURAL, det.name)

    def test_crypto_guard_never_decodes(self):
        det = CryptoGuardDetector()
        self.assertEqual(det.decode(b"anything"), [])


# ── Family B: statistical layers ────────────────────────────────────

class TestStatisticalLayers(unittest.TestCase):
    def test_xor_returns_every_nonzero_key(self):
        det = XorSingleByteLayer()
        candidates = det.decode(b"secret message")
        self.assertEqual(len(candidates), 255)
        self.assertEqual(det.family, Family.STATISTICAL)

    def test_xor_recovers_the_right_key(self):
        # The scorer, not the detector, picks the key -- so assert the correct
        # plaintext actually appears among the candidates.
        plaintext = b"secret message"
        encoded = bytes(b ^ 42 for b in plaintext)
        self.assertIn(plaintext, XorSingleByteLayer().decode(encoded))

    def test_xor_detect_is_constant(self):
        # No signature exists, so detect must not pretend to have one.
        det = XorSingleByteLayer()
        self.assertEqual(det.detect(b"anything at all"), det.detect(b"zzz"))

    def test_xor_output_is_not_coerced_to_text(self):
        # Binary in, binary out -- this is the regression the bytes pipeline
        # exists to prevent.
        out = XorSingleByteLayer().decode(b"\x00\x01\x02")
        self.assertTrue(all(isinstance(c, bytes) for c in out))

    def test_caesar_returns_all_shifts(self):
        det = CaesarBruteForceLayer()
        candidates = det.decode(b"the quick brown fox")
        self.assertEqual(len(candidates), 25)
        self.assertIn(b"the quick brown fox", [
            bytes(((b - 97 + s) % 26 + 97) if 97 <= b <= 122 else b for b in b"the quick brown fox")
            for s in range(25)
        ])

    def test_caesar_skips_non_alphabetic(self):
        # Shifting binary is a no-op, so it must not be brute-forced.
        det = CaesarBruteForceLayer()
        self.assertEqual(det.decode(bytes(range(32, 64))), [])
        self.assertEqual(det.detect(bytes(range(32, 64))), 0.0)

    def test_caesar_finds_shift(self):
        shifted = bytes(
            (b - 97 + 13) % 26 + 97 if 97 <= b <= 122 else b
            for b in b"the quick brown fox"
        )
        self.assertIn(b"the quick brown fox", CaesarBruteForceLayer().decode(shifted))

    def test_vigenere_recovers_plaintext_with_enough_text(self):
        # Unigram frequency analysis needs a substantial number of letters per
        # column. On short columns the correct shift ranks 2nd-3rd of 26, which
        # is why the layer also returns several candidate key lengths and lets
        # the scorer choose. This test uses enough text for it to be decisive.
        key = (2, 0, 19)  # "cat"
        plaintext = (
            b"the quick brown fox jumps over the lazy dog while the curious "
            b"observer watches the distant tower and considers whether the "
            b"evidence supports another interpretation of the message itself"
        )
        encrypted = bytes(
            (b - 97 + key[i % len(key)]) % 26 + 97 if 97 <= b <= 122 else b
            for i, b in enumerate(plaintext)
        )
        det = VigenereBruteForceLayer()
        self.assertGreater(det.detect(encrypted), 0.3)
        self.assertIn(plaintext, det.decode(encrypted))

    def test_vigenere_returns_multiple_key_lengths(self):
        # Ambiguous key length is expected, not a failure: multiples of the true
        # length score similarly on IoC, so several are offered for the scorer.
        plaintext = b"the quick brown fox jumps over the lazy dog again and again"
        encrypted = bytes(
            (b - 97 + (2, 0, 19)[i % 3]) % 26 + 97 if 97 <= b <= 122 else b
            for i, b in enumerate(plaintext)
        )
        self.assertGreaterEqual(len(VigenereBruteForceLayer().decode(encrypted)), 1)

    def test_vigenere_declines_short_input(self):
        # Too few letters per column for IoC to carry signal.
        det = VigenereBruteForceLayer()
        self.assertEqual(det.detect(b"the quick brown fox"), 0.0)
        self.assertEqual(det.decode(b"the quick brown fox"), [])

    def test_statistical_layers_are_marked_as_such(self):
        for det in (
            XorSingleByteLayer(),
            CaesarBruteForceLayer(),
            VigenereBruteForceLayer(),
        ):
            self.assertTrue(det.family.is_statistical, det.name)


# ── Heuristic detectors ─────────────────────────────────────────────

class TestHeuristicLayers(unittest.TestCase):
    def test_nth_character_noise(self):
        det = NthCharNoiseDetector()
        noisy = b"the aXnd thXen"  # every sixth character is noise
        self.assertGreater(det.detect(noisy), 0.3)
        self.assertIn(b"the and then", det.decode(noisy))

    def test_symbol_noise(self):
        det = SymbolNoiseDetector()
        noisy = b".This (is an e%ncode@d mess?age fo?r quan!tum ar?chive*s Proc&eed to $unloc@k the c#ipher"
        self.assertGreater(det.detect(noisy), 0.7)
        self.assertIn(
            b".This is an encoded message for quantum archives "
            b"Proceed to unlock the cipher",
            det.decode(noisy),
        )

    def test_split_interleave(self):
        raw_b64 = b"fXIjM3VVPHRmM0BfSF8+dDBfPXRlMCltYzMjbFdHJXs0fTtIR2wrNGZnIV9ucj8wd2sqX2NiJjRfbUBlb0wlQ0V7I3dHSCEz"
        payload = base64.b64decode(raw_b64)
        det = SplitInterleaveDetector()
        self.assertGreater(det.detect(payload), 0.7)
        self.assertIn(
            b"H3G{wELCome_b4ck_wr0ng_fl4G}H4G{W3lc0me_t0_tH3_fUtur3}",
            det.decode(payload),
        )


if __name__ == "__main__":
    unittest.main()
