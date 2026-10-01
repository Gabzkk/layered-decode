"""Known-answer chains from code references/layered_decoder."""

import base64
import codecs
import unittest
from urllib.parse import quote

from layered_decoder import LayeredDecoder
from layered_decoder.scoring import score


class TestReferencePuzzles(unittest.TestCase):
    def test_structural_chain_continues_past_rot13_flag(self):
        plaintext = b"H4G{n0_k3ys_just_pure_structure}"
        reversed_rot13 = codecs.encode(plaintext.decode()[::-1], "rot_13")
        payload = quote(base64.b64encode(reversed_rot13.encode().hex().encode()).decode())
        result = LayeredDecoder().decode(payload)
        self.assertEqual(result.final_bytes, plaintext)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.layers_detected, ["UrlEncoding", "Base64", "Hex", "Reversal", "Caesar"])

    def test_rot13_flag_is_not_already_solved(self):
        result = LayeredDecoder().decode(b"U4T{a0_x3lf_whfg_cher_fgehpgher}")
        self.assertEqual(result.final_bytes, b"H4G{n0_k3ys_just_pure_structure}")
        self.assertEqual(result.status, "solved")

    def test_single_byte_xor_survives_competing_repeating_key_guess(self):
        plaintext = b"H4G{s1ngle_str3am_no_split_n33ded}"
        wrapped = base64.b64encode(bytes(b ^ 42 for b in plaintext)).decode()[::-1]
        payload = "".join(
            chr((ord(c) - ord("a") + 5) % 26 + ord("a")) if "a" <= c <= "z"
            else chr((ord(c) - ord("A") + 5) % 26 + ord("A")) if "A" <= c <= "Z"
            else c for c in wrapped
        )
        result = LayeredDecoder().decode(payload)
        self.assertEqual(result.final_bytes, plaintext)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.layers_detected, ["Reversal", "Caesar", "Base64", "XorSingleByte"])

    def test_plain_flag_outranks_rotated_flag(self):
        self.assertGreater(
            score(b"H4G{n0_k3ys_just_pure_structure}"),
            score(b"U4T{a0_x3lf_whfg_cher_fgehpgher}") + 0.1,
        )

    def test_custom_prefix_allows_opaque_flag_body(self):
        plaintext = b"CUSTOM{zqxj_93827}"
        result = LayeredDecoder(flag_prefixes=("CUSTOM{",)).decode(plaintext)
        self.assertEqual(result.final_bytes, plaintext)
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.steps, [])

    def test_repeating_key_xor_chain(self):
        payload = (
            "hFzY1QDMxAzM1cDM0ADMzIGMiVDN1QWM4MTZwgzMjVDNwEDMzUjYxQTMlVzMwAzMwATMwY"
            "TNyATYwUGMmBDMzQTN3AzMxAzM0EzY1ETMlVDMwAzMzEzNxQTNiFTNxMWNmBjZxYTNjBDO"
            "zQWM0UDOwkDMmVjYwQTMwIjY1YmM"
        )
        result = LayeredDecoder().decode(payload)
        self.assertEqual(result.final_bytes, b"H4G{l0ng3r_c1ph3rt3xt_g1v3s_th3_hamm1ng_d1st4nc3_a_r34l_ch4nc3}")
        self.assertEqual(result.status, "solved")
        self.assertEqual(result.layers_detected, ["Reversal", "Base64", "Hex", "RepeatingKeyXor"])


if __name__ == "__main__":
    unittest.main()
