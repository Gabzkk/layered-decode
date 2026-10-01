import base64
import json
import unittest
from itertools import product
from unittest.mock import patch

from layered_decoder import LayeredDecoder
from layered_decoder.detectors import Family, get_all_detectors
from layered_decoder.detectors.repeating_key_xor import (
    RepeatingKeyXorLayer,
    column_chi_squared,
    hamming_distance,
    printable_keys,
    rank_key_lengths,
)
from layered_decoder.scoring import cached_score, score
from layered_decoder.models import DecodeResult
from tests.test_cli import run_cli


def encrypt(plaintext, key):
    return bytes(byte ^ key[i % len(key)] for i, byte in enumerate(plaintext))


class TestRepeatingKeyXor(unittest.TestCase):
    def test_registered_as_statistical(self):
        layer = next(l for l in get_all_detectors() if l.name == "RepeatingKeyXor")
        self.assertEqual(layer.family, Family.STATISTICAL)

    def test_hamming_reference_vector(self):
        self.assertEqual(hamming_distance(b"this is a test", b"wokka wokka!!!"), 37)
        with self.assertRaises(ValueError):
            hamming_distance(b"a", b"ab")

    def test_distance_averages_all_pairs_of_four_blocks(self):
        # Six pairs differ in 2, 2, 4, 4, 2, 2 bits, divided by 2 bytes.
        data = b"\x00\x00\x01\x01\x02\x02\x03\x03"
        self.assertAlmostEqual(rank_key_lengths(data)[0][0], 4 / 3)

    def test_lengths_require_four_complete_blocks(self):
        self.assertEqual(rank_key_lengths(b"1234567"), [])
        self.assertEqual([k for _, k in rank_key_lengths(b"12345678")], [2])
        self.assertLessEqual(max(k for _, k in rank_key_lengths(bytes(100))), 20)

    def test_printable_elimination_includes_every_byte_and_all_keys(self):
        self.assertEqual(printable_keys(b"A\xc1"), [])
        self.assertIn(255, printable_keys(bytes([255 ^ 32, 255 ^ 126])))
        self.assertNotIn(0, printable_keys(b"abc\n"))
        self.assertEqual(RepeatingKeyXorLayer().decode(bytes(range(256))), [])

    def test_impossible_lengths_are_eliminated_before_frequency_scoring(self):
        with patch("layered_decoder.detectors.repeating_key_xor.column_chi_squared") as scorer:
            self.assertEqual(RepeatingKeyXorLayer().decode(bytes(range(256))), [])
            scorer.assert_not_called()

    def test_chi_squared_prefers_english_over_printable_noise(self):
        self.assertLess(column_chi_squared(b"the secret is here"), column_chi_squared(b"zxqjzjxqzjxq"))
        self.assertEqual(column_chi_squared(b"12345"), float("inf"))

    def test_small_cartesian_product_beats_greedy_and_scores_every_key(self):
        plaintext = b"flag{ok}"
        data = encrypt(plaintext, b"AB")
        columns = [data[i::2] for i in range(2)]
        keys = [printable_keys(col) for col in columns]
        greedy = bytes(min(options, key=lambda k: column_chi_squared(bytes(b ^ k for b in col)))
                       for col, options in zip(columns, keys))
        self.assertNotEqual(encrypt(data, greedy), plaintext)
        expected_outputs = {encrypt(data, bytes(key)) for key in product(*keys) if key[0] != key[1]}
        with patch("layered_decoder.detectors.repeating_key_xor.score", wraps=score) as scorer:
            result = RepeatingKeyXorLayer().decode(data)
        self.assertEqual(result[0], plaintext)
        self.assertEqual({call.args[0] for call in scorer.call_args_list}, expected_outputs)

    def test_recovers_long_english_with_binary_key(self):
        plaintext = (
            b"The secret message is hidden here and you should read every word. "
            b"We have the key to decode this text and find the answer. "
        ) * 3
        layer = RepeatingKeyXorLayer()
        ciphertext = encrypt(plaintext, b"\x00\xff\x81")
        self.assertEqual(layer.decode(ciphertext)[0], plaintext)

    def test_considers_key_lengths_beyond_the_best_hamming_distance(self):
        plaintext = b"flag{the_secret_key}"
        data = encrypt(plaintext, b"KEY")
        self.assertNotEqual(rank_key_lengths(data)[0][1], 3)
        self.assertEqual(RepeatingKeyXorLayer().decode(data)[0], plaintext)

    def test_large_product_fallback_is_bounded_and_preserves_prefix_keys(self):
        data = encrypt(b"flag{ok}", b"AB")
        with patch("layered_decoder.detectors.repeating_key_xor.MAX_COMBINATIONS", 1):
            with patch("layered_decoder.detectors.repeating_key_xor.score", wraps=score) as scorer:
                result = RepeatingKeyXorLayer().decode(data)
        self.assertEqual(result[0], b"flag{ok}")
        self.assertLessEqual(scorer.call_count, 256 * 3)
        self.assertGreater(scorer.call_count, 1)

    def test_short_custom_flag_is_a_guess_even_when_forced(self):
        plaintext = b"H4G{the_secret_key}"
        ciphertext = encrypt(plaintext, b"ICE")
        decoder = LayeredDecoder(max_depth=1, flag_prefixes=("H4G{",))
        result = decoder.decode(ciphertext, forced_layers={1: "RepeatingKeyXor"})
        self.assertEqual(result.final_bytes, plaintext)
        self.assertEqual(result.status, "best_guess")
        self.assertEqual(result.confidence, "low")
        self.assertTrue(result.to_dict()["low_confidence_short_ciphertext"])
        self.assertEqual(result.steps[0].key_hex, b"ICE".hex())
        self.assertLess(result.steps[0].confidence, 0.5)

    def test_short_warning_survives_result_aggregation(self):
        data = encrypt(b"flag{ok}", b"AB")
        segment = LayeredDecoder(max_depth=1).decode(data, forced_layers={1: "RepeatingKeyXor"})
        combined = DecodeResult([], [], segment.steps, None, segment.final_output, "high", status="solved")
        self.assertEqual(combined.status, "best_guess")
        self.assertEqual(combined.confidence, "low")
        self.assertTrue(combined.to_dict()["low_confidence_short_ciphertext"])

    def test_confidence_boundary_uses_shortest_key_period(self):
        layer = RepeatingKeyXorLayer()
        for size, short in ((19, True), (20, False), (21, False)):
            plaintext = b"a" * size
            evidence = layer.candidate_evidence(encrypt(plaintext, b"ABAB"), plaintext)
            self.assertEqual(evidence.key, b"AB")
            self.assertEqual(evidence.low_confidence_short_ciphertext, short)

    def test_base64_wrapper_uses_registered_layer(self):
        plaintext = b"flag{the_secret_key}"
        result = LayeredDecoder(max_depth=2).decode(
            base64.b64encode(encrypt(plaintext, b"\x80\xfe\x81")),
            allow_segmentation=False,
        )
        self.assertEqual(result.final_bytes, plaintext)
        self.assertEqual(result.layers_detected, ["Base64", "RepeatingKeyXor"])
        self.assertEqual(result.status, "best_guess")

    def test_cli_prefix_and_short_warning(self):
        data = base64.b64encode(encrypt(b"H4G{the_secret_key}", b"\x80\xfe\x81")).decode()
        args = ("--max-depth", "2", "--flag-prefix", "H4G{", "-i", data)
        payload = json.loads(run_cli("--json", *args))
        self.assertEqual(payload["final_output"], "H4G{the_secret_key}")
        self.assertEqual(payload["status"], "best_guess")
        self.assertTrue(payload["low_confidence_short_ciphertext"])
        self.assertIn("low_confidence_short_ciphertext: true", run_cli(*args))


class TestKnownPrefixes(unittest.TestCase):
    def test_bonus_is_exact_case_sensitive_and_anchored(self):
        prefixes = ("H4G{",)
        correct = b"H4G{the_secret_key}"
        wrong = b"H4M{the_secret_key}"
        self.assertGreater(score(correct, prefixes), score(wrong, prefixes) + 0.5)
        for text in (b"x H4G{the_secret_key}", b"h4g{the_secret_key}"):
            self.assertEqual(score(text, prefixes), score(text, ()))

    def test_cache_keeps_prefix_configurations_separate(self):
        text = b"H4G{the_secret_key}"
        self.assertEqual(cached_score(text, ()), score(text, ()))
        self.assertEqual(cached_score(text, ("H4G{",)), score(text, ("H4G{",)))
        self.assertGreater(cached_score(text, ("H4G{",)), cached_score(text, ()))

    def test_prefix_does_not_reward_malformed_flag_noise(self):
        noise = b"flag{' m0%ea`&<e6%O}..."
        self.assertEqual(score(noise), score(noise, ()))
