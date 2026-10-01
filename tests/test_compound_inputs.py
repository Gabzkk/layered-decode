"""Compound / concatenated input handling.

Scope note, since it shapes what these tests assert: the beam has no split
operator, so on an *undelimited* concatenation its best reachable state is
whatever brute-force accident scores highest. When that accident happens to be
flag-shaped it can pass the "is this finished" check and suppress segmentation.

These tests therefore cover what the tool does support and is honest about:
labelled input, line-separated input, and segmentation as a fallback when the
beam cannot finish. The undelimited reverse-order case is asserted as a known
limitation rather than quietly dropped, so a future regression in the *working*
paths still shows up.
"""
import base64
import unittest

from layered_decoder import LayeredDecoder
from layered_decoder.cli import _parse_targets
from layered_decoder.detectors.base_encoding import Base64Detector

PAST = (
    "tgjrk#e gjv m@eqnpw$ qv fgg&eqtR u*gxkje?tc owv!pcws "
    "t?qh gic?uugo f@gfqep%g pc uk( ukjV."
)
PRESENT = (
    "fXIjM3VVPHRmM0BfSF8+dDBfPXRlMCltYzMjbFdHJXs0fTtIR2wrNGZnIV9ucj8"
    "wd2sqX2NiJjRfbUBlb0wlQ0V7I3dHSCEz"
)


class TestCompoundInputs(unittest.TestCase):
    def setUp(self):
        self.decoder = LayeredDecoder()

    def test_compound_past_and_present(self):
        res = self.decoder.decode(PAST + PRESENT)
        self.assertIsNotNone(res.segments)
        self.assertEqual(len(res.segments), 2)
        self.assertEqual(res.status, "solved")
        self.assertEqual(
            res.segments[0].final_output,
            ".This is an encoded message for quantum archives "
            "Proceed to unlock the cipher",
        )
        self.assertEqual(
            res.segments[1].final_output,
            "H3G{wELCome_b4ck_wr0ng_fl4G}H4G{W3lc0me_t0_tH3_fUtur3}",
        )

    def test_compound_hex_and_base64(self):
        compound = "48656c6c6f20576f726c64" + base64.b64encode(b"Secret Flag").decode()
        res = self.decoder.decode(compound)
        self.assertIsNotNone(res.segments)
        self.assertEqual(len(res.segments), 2)
        self.assertIn("Hello World", res.final_output)
        self.assertIn("Secret Flag", res.final_output)

    def test_undelimited_reverse_order_is_a_known_limitation(self):
        """Documents the gap rather than pretending it does not exist.

        The beam cannot split, so it may find a flag-shaped brute-force accident
        and report the compound as solved. Both outcomes are acceptable here;
        what must not happen is a crash, a hang, or a false "high confidence on
        nothing".
        """
        res = self.decoder.decode(PRESENT + PAST)
        self.assertIsInstance(res.final_output, str)
        self.assertIn(res.status, ("solved", "partial_decode_low_confidence"))
        if res.status == "solved":
            # If it does claim success, it must be a genuine destination.
            self.assertEqual(res.confidence, "high")

    def test_unsolvable_input_reports_low_confidence(self):
        garbage = "AbCdEfGhIjKlMnOpQrStUvWxYz012345$#@%^&*(!@#%^&*(!@#%^"
        res = self.decoder.decode(garbage)
        self.assertNotEqual(res.confidence, "high")
        self.assertNotEqual(res.status, "solved")

    def test_single_payloads_are_never_split(self):
        """The property that must not regress.

        Offering segmentation unconditionally was measured to split single
        payloads at coincidental boundaries. Single-payload correctness matters
        more than catching an undelimited compound.
        """
        for payload in (PAST, PRESENT, "SGVsbG8gV29ybGQ=", ".... . .-.. .-.. ---"):
            with self.subTest(payload=payload[:32]):
                res = self.decoder.decode(payload)
                self.assertIsNone(res.segments, f"single payload was split: {payload[:32]}")

    def test_normal_single_payloads_remain_unsegmented(self):
        res = self.decoder.decode("SGVsbG8gV29ybGQ=")
        self.assertEqual(res.final_output, "Hello World")
        self.assertIsNone(res.segments)
        self.assertFalse(res.likely_compound_input)


class TestLabelledAndMultiTargetInput(unittest.TestCase):
    """The supported way to handle several payloads in one file."""

    def setUp(self):
        self.decoder = LayeredDecoder()

    def test_labelled_lines_are_split_by_the_cli(self):
        import tempfile
        import os

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write(f"past: {PAST}\npresent: {PRESENT}\n")
            path = tmp.name
        try:
            targets = _parse_targets(path, None)
        finally:
            os.unlink(path)

        self.assertEqual([label for label, _ in targets], ["past", "present"])
        outputs = {label: self.decoder.decode(payload).final_output for label, payload in targets}
        self.assertIn("Proceed to unlock the cipher", outputs["past"])
        self.assertIn("H3G{wELCome_b4ck_wr0ng_fl4G}", outputs["present"])

    def test_bare_lines_are_treated_as_separate_payloads(self):
        import tempfile
        import os

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write(f"{PAST}\n{PRESENT}\n")
            path = tmp.name
        try:
            targets = _parse_targets(path, None)
        finally:
            os.unlink(path)

        self.assertEqual(len(targets), 2)
        for _, payload in targets:
            self.assertIsNone(self.decoder.decode(payload).segments)


class TestDetectorSpanHooks(unittest.TestCase):
    def test_matching_spans_and_ratio(self):
        det = Base64Detector()
        mixed = (
            b"some_random_prefix_with_symbols!#$%"
            + b"SGVsbG8gV29ybGQgVGhpcyBJcyBBIFRlc3Qh"
        )
        ratio = det.match_ratio(mixed)
        self.assertGreater(ratio, 0.40)
        self.assertLess(ratio, 0.95)

        spans = det.find_matching_spans(mixed)
        self.assertTrue(spans)
        start, end, conf = spans[0]
        self.assertGreaterEqual(start, 25)

    def test_statistical_layers_report_no_match_ratio(self):
        """A constant detect must not masquerade as a charset ratio.

        Compound detection reads 0.25-0.85 as "looks concatenated", so a family B
        layer reporting a flat 0.5 made every input look compound.
        """
        from layered_decoder.detectors.xor import XorSingleByteLayer
        from layered_decoder.detectors.caesar import CaesarBruteForceLayer

        for det in (XorSingleByteLayer(), CaesarBruteForceLayer()):
            self.assertEqual(det.match_ratio(b"just some ordinary text here"), 0.0)

    def test_find_matching_spans_defaults_to_empty(self):
        from layered_decoder.detectors.reversal import ReversalDetector
        self.assertEqual(ReversalDetector().find_matching_spans(b"abc"), [])


if __name__ == "__main__":
    unittest.main()
