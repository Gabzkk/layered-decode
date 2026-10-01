import argparse
import json
import os
import re
import sys

from .engine import LayeredDecoder
from .models import DecodeResult


def _parse_targets(file_path: str | None, raw_str: str | None) -> list[tuple[str | None, str]]:
    """Resolve input into a list of (label, payload) pairs."""
    if file_path is None and raw_str is not None and os.path.isfile(raw_str):
        file_path = raw_str
        raw_str = None

    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
        except Exception as exc:
            print(f"Error reading file: {exc}", file=sys.stderr)
            sys.exit(1)

        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]

        # Check if labeled targets exist (e.g. `past: ...`, `present: ...`)
        #
        # A separator is required after the colon, and `scheme://` is excluded.
        # Without both guards any payload containing a colon is misread -- a file
        # holding `https://example.com/secret` had "https" silently eaten as a
        # label, and `-f file` disagreed with `-i string` on identical input. For
        # a tool whose job is faithful byte-level handling, that is not a
        # cosmetic bug.
        labeled: list[tuple[str | None, str]] = []
        for l in lines:
            m = re.match(r"^([a-zA-Z0-9_\-]+):[ \t]+(\S.*)$", l)
            if m and "://" not in l:
                labeled.append((m.group(1), m.group(2).strip()))
            else:
                labeled.append((None, l))

        if any(t[0] is not None for t in labeled):
            return labeled

        # Check if multi-line block is a single continuous wrapped base64 or hex string
        collapsed = "".join(t[1] for t in labeled)
        if re.match(r"^[A-Za-z0-9+/=_-]+$", collapsed) and len(collapsed) > 60:
            return [(None, collapsed)]

        # Return the parsed lines, never the raw file content. Falling back to
        # `content` here leaked comment lines and blank-line padding into the
        # payload for any single-target file.
        return labeled

    elif raw_str is not None:
        return [(None, raw_str)]

    return []


def _print_result(result: DecodeResult, label: str | None = None, verbose: bool = False) -> None:
    """Print the step-by-step decoding trace for a result."""
    if label:
        print(f"=== Target: {label} ===")

    if result.segments:
        print(f"=== Compound Payload: {len(result.segments)} Segments Detected ===")
        for idx, seg in enumerate(result.segments, 1):
            print(f"--- Segment {idx} ---")
            for step in seg.steps:
                forced_tag = " [FORCED]" if step.forced else ""
                print(f"[Layer {step.layer_number}] Detected: {step.encoding_detected} "
                      f"(confidence: {step.confidence:.2f}){forced_tag}")
                if verbose:
                    inp = step.input_value
                    if len(inp) > 120:
                        inp = inp[:117] + "..."
                    print(f"  Input:  {inp}")
                out = step.output_value
                if len(out) > 120:
                    out = out[:117] + "..."
                print(f"  Output: {out}")
                delta = step.score_after - step.score_before
                print(f"  Score:  {step.score_before:.2f} \u2192 {step.score_after:.2f} ({delta:+.2f})")
                print()
            print(f"  Segment {idx} Output: {seg.final_output}")
            print(f"  Segment {idx} Status: {seg.stopped_reason} (conf: {seg.confidence})")
            print()
    else:
        for step in result.steps:
            forced_tag = " [FORCED]" if step.forced else ""
            print(f"[Layer {step.layer_number}] Detected: {step.encoding_detected} "
                  f"(confidence: {step.confidence:.2f}){forced_tag}")
            if verbose:
                inp = step.input_value
                if len(inp) > 120:
                    inp = inp[:117] + "..."
                print(f"  Input:  {inp}")
            out = step.output_value
            if len(out) > 120:
                out = out[:117] + "..."
            print(f"  Output: {out}")
            delta = step.score_after - step.score_before
            print(f"  Score:  {step.score_before:.2f} \u2192 {step.score_after:.2f} ({delta:+.2f})")
            print()

    if result.branches:
        print("=== Branches ===")
        for branch in result.branches:
            print(f"  [{branch.branch_id}] {len(branch.steps)} layers \u2192 {branch.final_output[:80]}")
        print()

    target_suffix = f" [{label}]" if label else ""
    print(f"=== Final Output{target_suffix} ===")
    print(result.final_output)
    print()
    print(f"Status:     {result.status}")
    print(f"Confidence: {result.confidence}")
    print(f"Stopped:    {result.stopped_reason}")
    if result.key_required:
        print(f"Decryption key required: {result.encryption_format}")
        print(f"Evidence: {result.key_requirement_evidence}")
        print(f"Needed: {result.required_decryption_info}")
    if result.low_confidence_short_ciphertext:
        print("low_confidence_short_ciphertext: true")
    if result.likely_compound_input:
        print(f"Compound:   True (partial_match: {result.partial_match_detected})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="layered-decoder",
        description="Universal Layered Decoder — auto-peel stacked encodings for CTF / security analysis",
    )
    parser.add_argument("input_pos", nargs="*", help="Input string or path to file")
    parser.add_argument("-i", "--input", dest="input_opt", help="Input string (alternative to positional)")
    parser.add_argument("-f", "--file", help="Read input from a file")
    parser.add_argument("--max-depth", type=int, default=15, help="Max decode layers (default: 15)")
    parser.add_argument("--beam-width", type=int, default=3, help="Search beam width (default: 3)")
    parser.add_argument(
        "--force",
        action="append",
        default=[],
        metavar="LAYER:ENCODING",
        help="Force a specific encoding at a layer, e.g. --force 3:rot13  (repeatable)",
    )
    parser.add_argument("--try-branches", action="store_true", help="Enable multi-branch split/decode/recombine")
    parser.add_argument(
        "--flag-prefix", action="append", default=[], metavar="PREFIX",
        help="Add an exact flag prefix at the start, e.g. 'H4G{' (repeatable)",
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show intermediate input values")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print the final decoded output")

    args = parser.parse_args()

    # ── resolve input targets ──────────────────────────────────────
    raw_input = args.input_opt
    if raw_input is None and args.input_pos:
        if len(args.input_pos) == 1:
            raw_input = args.input_pos[0]
        else:
            raw_input = " ".join(args.input_pos)

    targets = _parse_targets(args.file, raw_input)
    if not targets:
        parser.print_help()
        sys.exit(1)

    # ── parse forced layers ────────────────────────────────────────
    forced_layers: dict[int, str] = {}
    for spec in args.force:
        try:
            layer_str, enc = spec.split(":", 1)
            forced_layers[int(layer_str)] = enc
        except ValueError:
            print(f"Invalid --force format: {spec!r}  (expected LAYER:ENCODING)", file=sys.stderr)
            sys.exit(1)

    decoder = LayeredDecoder(
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        flag_prefixes=tuple(args.flag_prefix),
    )

    # ── process targets ────────────────────────────────────────────
    results: list[tuple[str | None, DecodeResult]] = []
    for label, payload in targets:
        if args.try_branches:
            res = decoder.decode_with_branches(payload, forced_layers=forced_layers or None)
        else:
            res = decoder.decode(payload, forced_layers=forced_layers or None)
        results.append((label, res))

    # ── format output ──────────────────────────────────────────────
    if args.json_output:
        if len(results) == 1 and results[0][0] is None:
            print(json.dumps(results[0][1].to_dict(), indent=2, ensure_ascii=False))
        else:
            json_dict = {
                (label or f"target_{idx + 1}"): res.to_dict()
                for idx, (label, res) in enumerate(results)
            }
            print(json.dumps(json_dict, indent=2, ensure_ascii=False))
        return

    if args.quiet:
        for label, res in results:
            if label and len(results) > 1:
                print(f"{label}: {res.final_output}")
            else:
                print(res.final_output)
        return

    # Standard human-readable output
    for label, res in results:
        _print_result(res, label=label if len(results) > 1 else None, verbose=args.verbose)


if __name__ == "__main__":
    main()
