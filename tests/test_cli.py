"""CLI behaviour: input parsing, output modes, exit behaviour."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

from layered_decoder.cli import _parse_targets, main


def run_cli(*argv):
    """Invoke the CLI with argv, returning stdout."""
    buffer = io.StringIO()
    saved = sys.argv
    sys.argv = ["layered-decoder", *argv]
    try:
        with contextlib.redirect_stdout(buffer):
            main()
    finally:
        sys.argv = saved
    return buffer.getvalue()


def write_temp(content, suffix=".txt"):
    handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    handle.write(content)
    handle.close()
    return handle.name


class TestTargetParsing(unittest.TestCase):
    def test_labelled_lines(self):
        path = write_temp("past: AAA\npresent: BBB\n")
        try:
            targets = _parse_targets(path, None)
        finally:
            os.unlink(path)
        self.assertEqual(targets, [("past", "AAA"), ("present", "BBB")])

    def test_colon_in_payload_is_not_a_label(self):
        """Regression: `-f file` used to disagree with `-i string`.

        `https://example.com/secret` matched the old label regex, so "https" was
        eaten and the tool reported `//example.com/secret` with low confidence.
        """
        path = write_temp("https://example.com/secret\n")
        try:
            targets = _parse_targets(path, None)
        finally:
            os.unlink(path)
        self.assertEqual(targets, [(None, "https://example.com/secret")])

    def test_bare_lines_are_separate_payloads(self):
        path = write_temp("AAAA\nBBBB\n")
        try:
            targets = _parse_targets(path, None)
        finally:
            os.unlink(path)
        self.assertEqual(targets, [(None, "AAAA"), (None, "BBBB")])

    def test_wrapped_base64_is_rejoined(self):
        wrapped = "SGVsbG8gV29ybGQgdGhpcyBpcyBhIGxvbmdlciBiYXNlNjQgc3RyaW5nIHRoYXQgc3Bh\n" \
                  "bnMgb3ZlciBsaW5lcyBmb3Igd2l0aCB0aGUgb2Zmc2V0IHdyaXBwZWQgZW5jb2Rpbmc=\n"
        path = write_temp(wrapped)
        try:
            targets = _parse_targets(path, None)
        finally:
            os.unlink(path)
        self.assertEqual(len(targets), 1)
        self.assertNotIn("\n", targets[0][1])

    def test_comments_ignored(self):
        path = write_temp("# a comment\nSGVsbG8=\n")
        try:
            targets = _parse_targets(path, None)
        finally:
            os.unlink(path)
        self.assertEqual(targets, [(None, "SGVsbG8=")])

    def test_string_input_passes_through(self):
        self.assertEqual(_parse_targets(None, "hello"), [(None, "hello")])


class TestOutputModes(unittest.TestCase):
    def test_quiet_prints_only_output(self):
        output = run_cli("-q", "-i", "SGVsbG8gV29ybGQ=")
        self.assertEqual(output.strip(), "Hello World")

    def test_json_output_is_valid_json(self):
        output = run_cli("--json", "-i", "SGVsbG8=")
        payload = json.loads(output)
        self.assertEqual(payload["final_output"], "Hello")
        self.assertIn("status", payload)
        self.assertIn("final_bytes_hex", payload)
        self.assertEqual(payload["status"], "solved")

    def test_standard_output_reports_status(self):
        output = run_cli("-i", "SGVsbG8=")
        self.assertIn("Status:", output)
        self.assertIn("Confidence:", output)
        self.assertIn("Hello", output)

    def test_max_depth_flag(self):
        output = run_cli("--max-depth", "1", "-i", "SGVsbG8=")
        self.assertIn("Stopped:", output)

    def test_forced_layer_flag(self):
        output = run_cli("--force", "1:hex", "-i", "48656c6c6f")
        self.assertIn("FORCED", output)

    def test_invalid_force_format_exits(self):
        with self.assertRaises(SystemExit):
            run_cli("--force", "nonsense", "-i", "SGVsbG8=")

    def test_try_branches_flag(self):
        left, right = "SGVsbG8g", "V29ybGQ="
        interleaved = "".join("".join(pair) for pair in zip(left, right))
        output = run_cli("--try-branches", "-i", interleaved)
        self.assertIn("Hello", output)

    def test_beam_width_flag(self):
        output = run_cli("--beam-width", "5", "-i", "SGVsbG8=")
        self.assertIn("Hello", output)

    def test_flag_prefix_flag(self):
        output = run_cli("--flag-prefix", "CUSTOM{", "-i", "SGVsbG8=")
        self.assertIn("Status:", output)

    def test_multiple_targets_are_all_reported(self):
        path = write_temp("past: SGVsbG8g\npresent: V29ybGQ=\n")
        try:
            output = run_cli("-f", path)
        finally:
            os.unlink(path)
        self.assertIn("past", output)
        self.assertIn("present", output)


if __name__ == "__main__":
    unittest.main()
