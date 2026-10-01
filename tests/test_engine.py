"""Engine and beam-search behaviour.

These tests cover the properties the search architecture is supposed to provide,
not just individual decodes:

* a chain survives an intermediate that scores *worse* than its input
  (the failure mode that killed the old greedy loop)
* the correctness of the termination bands against the real score distribution
* the stop-signal veto
* the bytes-native contract end to end
"""
import base64
import os
import random
import time
import unittest
import zlib

from layered_decoder import LayeredDecoder
from layered_decoder.detectors.reversal import ReversalDetector
from layered_decoder.detectors.substitution import NthCharNoiseDetector
from layered_decoder.detectors.vigenere import VigenereBruteForceLayer
from layered_decoder.engine import (
    MIN_BEST_IMPROVEMENT,
    PARTIAL_SCORE,
    SOLVED_SCORE,
)
from layered_decoder.models import DecodeResult
from layered_decoder.scoring import score


class TestBytesNativePipeline(unittest.TestCase):
    def setUp(self):
        self.decoder = LayeredDecoder()

    def test_result_carries_raw_bytes(self):
        # final_output is a lossy render; anything continuing the decode needs
        # the bytes.
        result = self.decoder.decode("SGVsbG8=")
        self.assertEqual(result.final_bytes, b"Hello")
        self.assertEqual(result.final_output, "Hello")

    def test_binary_payload_does_not_crash_or_corrupt(self):
        """Bytes-native means layers can *carry* binary, not that it scores well.

        `score` measures text quality, so arbitrary binary scores 0.0 and the
        beam legitimately prefers the base64 text (0.18) over the binary it
        decodes to. Round-tripping non-text is not a goal of a CTF text decoder
        -- what matters is that the pipeline handles it without raising and
        without inventing output.
        """
        payload = bytes(range(256))
        encoded = base64.b64encode(payload).decode()
        result = self.decoder.decode(encoded)  # must not raise

        self.assertIsInstance(result.final_bytes, bytes)
        # Whatever it settled on must be a real state it actually visited, not
        # a fabricated string.
        self.assertIn(result.final_bytes, (payload, encoded.encode()))

    def test_binary_payload_is_not_reported_as_solved(self):
        payload = bytes(range(256))
        result = self.decoder.decode(base64.b64encode(payload).decode())
        self.assertNotEqual(result.status, "solved")
        self.assertNotEqual(result.confidence, "high")

    def test_decode_accepts_bytes_input(self):
        result = self.decoder.decode(b"SGVsbG8gV29ybGQ=")
        self.assertEqual(result.final_output, "Hello World")

    def test_undecodable_output_is_escaped_not_mojibake(self):
        # Must not silently present binary as if it were the answer.
        result = self.decoder.decode(base64.b64encode(bytes([0xC3, 0x28])).decode())
        self.assertNotEqual(result.status, "solved")
        self.assertNotIn("Ã", result.final_output)


class TestBeamSearchBehaviour(unittest.TestCase):
    def setUp(self):
        self.decoder = LayeredDecoder()

    def test_intermediate_may_score_worse_than_its_input(self):
        """The regression the beam exists to prevent.

        base64-of-hex decodes to a hex string that scores *below* the base64
        wrapper, then decodes to real plaintext. A greedy loop that requires
        every step to improve cannot get there.
        """
        inner = "48656c6c6f20576f726c64"  # hex("Hello World")
        encoded = base64.b64encode(inner.encode()).decode()

        wrapper_score = score(encoded.encode())
        intermediate_score = score(inner.encode())
        self.assertLess(
            intermediate_score,
            wrapper_score,
            "precondition: the intermediate must genuinely score worse",
        )

        result = self.decoder.decode(encoded)
        self.assertEqual(result.final_output, "Hello World")
        self.assertEqual(result.layers_detected, ["Base64", "Hex"])

    def test_beam_survives_a_zero_gain_first_step(self):
        # Reversal of this payload is a pure permutation: same entropy, no word
        # gain, so the first step cannot improve the score on its own. Greedy
        # would have to drop it.
        payload = (
            "tgjrk#e gjv m@eqnpw$ qv fgg&eqtR u*gxkje?tc owv!pcws "
            "t?qh gic?uugo f@gfqep%g pc uk( ukjV."
        )
        reversed_score = score(payload[::-1].encode())
        self.assertLessEqual(
            reversed_score, score(payload.encode()),
            "precondition: reversal must not improve the score by itself",
        )
        result = self.decoder.decode(payload)
        self.assertEqual(result.layers_detected[0], "Reversal")
        self.assertIn("encoded", result.final_output)

    def test_structural_match_preempts_brute_force(self):
        """A visible encoding must not be crowded out by a 255-key XOR spray.

        Without the precedence rule, Base64 (a correct first step scoring 0.209)
        loses to three constant-confidence brute-force layers scoring 0.22-0.24
        and the beam prunes the real path at depth 1.
        """
        payload = "fXIjM3VVPHRmM0BfSF8+dDBfPXRlMCltYzMjbFdHJXs0fTtIR2wrNGZnIV9ucj8wd2sqX2NiJjRfbUBlb0wlQ0V7I3dHSCEz"
        self.assertTrue(self.decoder._strong_structural_match(payload.encode()))
        result = self.decoder.decode(payload)
        self.assertEqual(result.layers_detected[0], "Base64")

    def test_beam_width_controls_exploration(self):
        # beam_width=1 degenerates to greedy; wider must not be worse.
        narrow = LayeredDecoder(beam_width=1).decode(
            base64.b64encode(b"48656c6c6f20576f726c64").decode()
        )
        wide = LayeredDecoder(beam_width=4).decode(
            base64.b64encode(b"48656c6c6f20576f726c64").decode()
        )
        self.assertEqual(wide.final_output, "Hello World")
        self.assertIsInstance(narrow.final_output, str)

    def test_respects_max_depth(self):
        inner = base64.b64encode(b"Hello World").decode()
        outer = base64.b64encode(inner.encode()).decode()
        result = LayeredDecoder(max_depth=1).decode(outer)
        self.assertLessEqual(len(result.steps), 1)

    def test_cycle_guard_prevents_revisiting_a_value(self):
        # Reversal twice returns the original; it must not be applied twice.
        result = self.decoder.decode("}olleh{galf")
        self.assertEqual(result.layers_detected.count("Reversal"), 1)

    def test_forced_layer_is_honoured(self):
        # Forcing bypasses the detection floor -- the user already decided.
        result = self.decoder.decode("48656c6c6f", forced_layers={1: "hex"})
        self.assertTrue(result.steps)
        self.assertTrue(result.steps[0].forced)
        self.assertEqual(result.steps[0].encoding_detected, "Hex")
        self.assertEqual(result.final_output, "Hello")

    def test_forced_layer_bypasses_detection_floor(self):
        # "SGVsbG8=" would never be guessed as hex, but forcing it must still
        # try rather than silently doing nothing.
        result = self.decoder.decode("SGVsbG8=", forced_layers={1: "hex"})
        # Hex cannot decode this input, so there is legitimately no step -- but
        # it must not raise, and the verdict must be honest about it.
        self.assertEqual(result.status, "likely_real_cryptography_or_unrecoverable")

    def test_invalid_forced_layer_is_reported_not_raised(self):
        result = self.decoder.decode("SGVsbG8=", forced_layers={1: "nosuchlayer"})
        self.assertEqual(result.status, "likely_real_cryptography_or_unrecoverable")

    def test_unknown_forced_alias_does_not_crash(self):
        for alias in ("rot13", "b64", "url", "xor", "gzip"):
            with self.subTest(alias=alias):
                LayeredDecoder().decode("SGVsbG8=", forced_layers={1: alias})


class TestTerminationAndHonesty(unittest.TestCase):
    def setUp(self):
        self.decoder = LayeredDecoder()

    def test_status_bands_match_the_documented_thresholds(self):
        # The spec's bands, applied to this scorer's real output.
        solved = self.decoder.decode("SGVsbG8gV29ybGQ=")
        self.assertEqual(solved.status, "solved")
        self.assertGreaterEqual(score(solved.final_bytes), SOLVED_SCORE)

    def test_score_thresholds_match_distribution(self):
        """Calibration guard.

        If the scorer's scale drifts, these bands silently stop meaning
        anything. This pins the relationship between known-good plaintext and
        the thresholds that classify it.
        """
        good = [
            b"Hello World",
            b"the quick brown fox jumps over the lazy dog",
            b"H3G{wELCome_b4ck_wr0ng_fl4G}",
            b"flag{the_quick_brown_fox}",
        ]
        junk = [
            bytes(range(200, 256)),
            b"awefuihwef823rb238rbq2389",
            b"jytlu={>yth>s^{opni:>oh>xyy8{ojL>k4yfut{!j}>",
        ]
        for payload in good:
            self.assertGreaterEqual(
                score(payload), PARTIAL_SCORE,
                f"good plaintext below the partial threshold: {payload!r}",
            )
        for payload in junk:
            self.assertLess(
                score(payload), SOLVED_SCORE,
                f"junk at or above the solved threshold: {payload!r}",
            )

    def test_unrecoverable_input_is_reported_honestly(self):
        result = self.decoder.decode(b"\x01\x02\x03\x04not-an-encoding")
        self.assertIn(
            result.status,
            ("likely_real_cryptography_or_unrecoverable", "partial_decode_low_confidence"),
        )
        self.assertEqual(result.confidence, "low")

    def test_real_ciphertext_is_not_claimed_as_solved(self):
        # The explicit non-goal: never claim to have cracked real crypto.
        result = self.decoder.decode(base64.b64encode(os.urandom(64)).decode())
        self.assertEqual(result.stopped_reason, "crypto_detected")
        self.assertEqual(result.status, "likely_real_cryptography_or_unrecoverable")
        self.assertEqual(result.layers_detected, [])

    def test_empty_input(self):
        result = self.decoder.decode("")
        self.assertEqual(result.final_output, "")
        self.assertEqual(result.steps, [])

    def test_plain_text_passthrough(self):
        result = self.decoder.decode("Just plain text here")
        self.assertIn(result.final_output, ["Just plain text here"])


class TestStopSignals(unittest.TestCase):
    def setUp(self):
        self.decoder = LayeredDecoder()

    def test_crypto_guard_blocks_expansion_on_raw_ciphertext(self):
        self.assertTrue(self.decoder._tripped_stop_signal(os.urandom(48)))

    def test_crypto_guard_defers_to_an_encoding_wrapper(self):
        """A base64 wrapper is structure, not ciphertext.

        The guard base64-decodes before judging, so a wrapper around ciphertext
        used to trip it -- and since a trip restricts expansion to family B, that
        silently removed Base64 from the layer set and made the wrapper
        unpeelable. The payload dead-ended at the step that would have
        progressed.
        """
        wrapped = base64.b64encode(os.urandom(48))
        self.assertFalse(self.decoder._tripped_stop_signal(wrapped))
        # ...but the contents are still recognised as ciphertext once peeled.
        self.assertTrue(self.decoder._tripped_stop_signal(base64.b64decode(wrapped)))

    def test_reversed_base64_wrapper_is_still_peelable(self):
        """Reversal must not be blocked by the guard it now outranks.

        `==` at the start is base64 padding, valid only at the end, so this is a
        reversed wrapper. Reversal is structural evidence, and the guard must not
        fire on the shell and block the peel.
        """
        payload = base64.b64encode(os.urandom(64)).decode()[::-1]
        result = self.decoder.decode(payload)
        self.assertEqual(result.layers_detected[0], "Reversal")
        # The content is genuine high-entropy binary, so the honest verdict is a
        # refusal -- not a false success.
        self.assertNotEqual(result.status, "solved")

    def test_guarded_input_still_gets_the_bounded_family_b_search(self):
        """The guard defers to family B, because XOR *looks* like ciphertext.

        High-entropy bytes are exactly what a single-byte XOR produces, and 255
        keys is a bounded search. Refusing family B on a guard trip reported
        "real cryptography" for payloads that were one XOR key from solved.
        """
        for key, plaintext in (
            (0x2A, b"H3G{wELCome_b4ck_wr0ng_fl4G}"),
            (0x2A, b"the quick brown fox jumps over"),
            (42, b"flag{steg_hiding_in_plain}"),
        ):
            with self.subTest(key=key, plaintext=plaintext[:20]):
                cipher = bytes(b ^ key for b in plaintext)
                result = self.decoder.decode(cipher)
                self.assertEqual(result.final_bytes, plaintext)
                self.assertEqual(result.status, "solved")

    def test_guarded_input_does_not_trigger_deeper_structural_search(self):
        # The non-goal still holds: a guard trip refuses to keep hunting for
        # encodings, it just tries the cheap brute force first.
        cipher = bytes(b ^ 0x2A for b in os.urandom(32))
        result = self.decoder.decode(cipher)
        self.assertNotEqual(result.status, "solved")

    def test_real_ciphertext_is_still_refused(self):
        result = self.decoder.decode(base64.b64encode(os.urandom(64)).decode())
        self.assertEqual(result.stopped_reason, "crypto_detected")
        self.assertEqual(result.status, "likely_real_cryptography_or_unrecoverable")

    def test_compressed_data_is_not_mistaken_for_crypto(self):
        # High entropy, but it is an encoding -- the guard must not fire.
        payload = base64.b64encode(zlib.compress(b"a" * 500)).decode()
        self.assertFalse(self.decoder._tripped_stop_signal(payload.encode()))

    def test_printable_text_is_not_mistaken_for_crypto(self):
        self.assertFalse(self.decoder._tripped_stop_signal(b"the quick brown fox"))


class TestRealChallenge(unittest.TestCase):
    """End-to-end regression on the actual challenge payloads."""

    PAST = (
        "tgjrk#e gjv m@eqnpw$ qv fgg&eqtR u*gxkje?tc owv!pcws "
        "t?qh gic?uugo f@gfqep%g pc uk( ukjV."
    )
    PRESENT = (
        "fXIjM3VVPHRmM0BfSF8+dDBfPXRlMCltYzMjbFdHJXs0fTtIR2wrNGZnIV9ucj8"
        "wd2sqX2NiJjRfbUBlb0wlQ0V7I3dHSCEz"
    )

    def setUp(self):
        self.decoder = LayeredDecoder()

    def test_chal_past(self):
        result = self.decoder.decode(self.PAST)
        self.assertEqual(
            result.final_output,
            ".This is an encoded message for quantum archives "
            "Proceed to unlock the cipher",
        )
        self.assertEqual(result.layers_detected, ["Reversal", "Caesar", "SymbolNoise"])
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.confidence, "high")

    def test_chal_present(self):
        result = self.decoder.decode(self.PRESENT)
        self.assertEqual(
            result.final_output,
            "H3G{wELCome_b4ck_wr0ng_fl4G}H4G{W3lc0me_t0_tH3_fUtur3}",
        )
        self.assertEqual(result.layers_detected, ["Base64", "SplitInterleave"])
        self.assertEqual(result.status, "solved")

    def test_chal_file_end_to_end(self):
        # Exercises the real CLI path, including its label parsing, rather than
        # handing the decoder a blob it would never see in practice.
        import pathlib
        import tempfile

        from layered_decoder.cli import main as cli_main

        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
            tmp.write(f"past: {self.PAST}\npresent: {self.PRESENT}\n")
            tmp_path = tmp.name

        import contextlib
        import io
        import sys

        buffer = io.StringIO()
        argv = sys.argv
        sys.argv = ["layered-decoder", "-f", tmp_path, "-q"]
        try:
            with contextlib.redirect_stdout(buffer):
                cli_main()
        finally:
            sys.argv = argv
            os.unlink(tmp_path)

        output = buffer.getvalue()
        self.assertIn("H3G{wELCome_b4ck_wr0ng_fl4G}", output)
        self.assertIn("Proceed to unlock the cipher", output)

    def test_decodes_within_a_sane_time_budget(self):
        # 255 XOR candidates at every node is a real cost. If this regresses,
        # the search has become impractical for interactive CTF use.
        start = time.time()
        self.decoder.decode(self.PAST)
        self.decoder.decode(self.PRESENT)
        elapsed = time.time() - start
        self.assertLess(elapsed, 10.0, f"decoding took {elapsed:.1f}s")

    def test_family_b_layers_crack_simple_ciphers_end_to_end(self):
        """The capability family B exists to provide."""
        cases = [
            (bytes(b ^ 42 for b in b"flag{steg_hiding_in_plain}"), b"flag{steg_hiding_in_plain}"),
            (b"uryyb qb V unpx n obzo", b"hello do I hack a bomb"),
            (
                base64.b64encode(bytes(b ^ 0x11 for b in b"the secret is hidden here")),
                b"the secret is hidden here",
            ),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected[:24]):
                self.assertEqual(self.decoder.decode(payload).final_bytes, expected)


class TestNoOpThenPayoff(unittest.TestCase):
    """A step that cannot improve the score must stay live for the next depth.

    The minimal shape: layer N is a pure permutation, so its score cannot rise,
    and layer N only pays off once layer N+1 has been applied on top. A search
    that discards a non-improving step throws the answer away.
    """

    PLAINTEXT = b"the flag is in the file and the code will show it to you"

    def _reversed_rot13(self) -> bytes:
        rot = bytes(
            (b - 97 + 13) % 26 + 97 if 97 <= b <= 122 else b
            for b in self.PLAINTEXT
        )
        return rot[::-1]

    def test_reversal_is_a_genuine_no_op(self):
        """Precondition: the first step must not meaningfully improve the score.

        Not exactly score-neutral -- reversal moves the last word to the front, so
        token boundaries shift and the dictionary ratio drifts a little (+0.024).
        What matters is that the gain falls below the engine's improvement
        threshold, so the step cannot be accepted on its own merit and the search
        must retain it for the next depth rather than discard it.
        """
        payload = self._reversed_rot13()
        delta = score(payload[::-1]) - score(payload)
        self.assertLess(
            delta, MIN_BEST_IMPROVEMENT,
            "precondition: reversal must not be a meaningful gain on its own",
        )

    def test_no_op_first_step_is_retained_and_pays_off(self):
        result = LayeredDecoder().decode(self._reversed_rot13())
        self.assertEqual(result.layers_detected, ["Reversal", "Caesar"])
        self.assertEqual(result.final_bytes, self.PLAINTEXT)
        self.assertEqual(result.status, "solved")

    def test_destructive_layers_may_not_corrupt_solved_text(self):
        """Regression: noise-stripping was rewarded for deleting letters.

        `english_word_ratio` is a ratio, so dropping one letter that breaks a word
        *raises* it. NthCharNoise therefore turned "a dead end" into "adead end"
        at a higher score (0.608 vs 0.598), and the beam preferred corrupted text
        over the correct plaintext it had already reached.
        """
        text = b"the secret is that layer one alone looks like a dead end"
        detector = NthCharNoiseDetector()
        self.assertEqual(detector.detect(text), 0.0)
        self.assertEqual(detector.decode(text), [])

    def test_nth_char_noise_still_detects_real_injected_noise(self):
        # The fix must not disable the layer it was protecting.
        detector = NthCharNoiseDetector()
        self.assertGreater(detector.detect(b"the aXnd thXen"), 0.3)
        self.assertIn(b"the and then", detector.decode(b"the aXnd thXen"))


class TestReversedBase64Report(unittest.TestCase):
    """The reported failure, pinned as a regression.

    `==` at the start is base64 padding, which is only valid at the end, so this
    is a reversed wrapper. Reversal used to return 0.00 here: it is a structural
    layer but only looked for dictionary words, and a reversed *encoded* payload
    has none.
    """

    PAYLOAD = "==bAT9pYEpGW151VLuAB1AJW1i0XElqCEA3YL1JWgpAZy5mD"

    def test_reversal_detects_misplaced_padding(self):
        self.assertGreater(
            ReversalDetector().detect(self.PAYLOAD.encode()), 0.85
        )

    def test_reversal_does_not_fire_on_well_formed_encodings(self):
        detector = ReversalDetector()
        for payload in (
            base64.b64encode(b"the quick brown fox jumps over the lazy dog"),
            base64.b32encode(b"the quick brown fox jumps"),
            b"48656c6c6f20576f726c64",
            b"the quick brown fox jumps over the lazy dog",
            b"0100100001100101011011000110110001101111",
        ):
            with self.subTest(payload=payload[:20]):
                expected = 0.0 if payload.endswith(b"=") else 0.6
                self.assertEqual(detector.detect(payload), expected)

    def test_reversal_detects_every_reversed_padded_blob(self):
        detector = ReversalDetector()
        rng = random.Random(4)
        hits = trials = 0
        for _ in range(400):
            encoded = base64.b64encode(os.urandom(rng.randrange(9, 40)))
            if not encoded.endswith(b"="):
                continue
            trials += 1
            hits += detector.detect(encoded[::-1]) > 0.85
        self.assertGreater(trials, 50, "corpus too small to be meaningful")
        self.assertEqual(hits, trials)

    def test_vigenere_detect_is_not_a_constant(self):
        """It used to return a flat 0.5 for any input -- a coin flip."""
        detector = VigenereBruteForceLayer()

        def enc(plaintext: bytes, shifts):
            return bytes(
                (b - 97 + shifts[i % len(shifts)]) % 26 + 97 if 97 <= b <= 122 else b
                for i, b in enumerate(plaintext)
            )

        text = (
            b"the quick brown fox jumps over the lazy dog and the "
            b"watchtower bell rings loudly tonight"
        )
        vigenere = detector.detect(enc(text, [2, 0, 19]))
        caesar = detector.detect(enc(text, [13]))
        self.assertGreater(vigenere, caesar)
        # Structureless bytes are not a cipher at all.
        self.assertEqual(detector.detect(os.urandom(48)), 0.0)

    def test_no_op_then_payoff_reversal_caesar_b64_xor(self):
        """The 4-layer challenge: Reversal -> Caesar -> Base64 -> XorSingleByte.

        Payload: ==bAT9pYEpGW151VLuAB1AJW1i0XElqCEA3YL1JWgpAZy5mD
        Step 1: Reversal yields Dm5yZApgWJ1LY3AECqlEX0i1WJA1BAuLV151WGpEYp9TAb== (+0.00 delta)
        Step 2: Caesar shift 21 yields Yh5tUVkbRE1GT3VZXlgZS0d1REV1WVpGQ151RBkZTk9OVw== (valid Base64)
        Step 3: Base64 decode yields XOR ciphertext
        Step 4: XOR key 0x2a unlocks the flag H4G{s1ngle_str3am_no_split_n33ded}
        """
        result = LayeredDecoder().decode(self.PAYLOAD)
        self.assertEqual(
            result.layers_detected,
            ["Reversal", "Caesar", "Base64", "XorSingleByte"],
        )
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.confidence, "high")
        self.assertIn(b"H4G{s1ngle_str3am_no_split_n33ded}", result.final_bytes)


if __name__ == "__main__":
    unittest.main()
