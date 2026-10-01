# Universal Layered Crypt

A Python tool that detects and peels stacked encodings commonly found in CTFs and
security analysis. It explores the space of possible decodings with a beam
search, ranks every candidate by how much it looks like real text, and reports
honestly when it cannot find an answer.

**Explicit Non-Goal**: This tool does not break real cryptography (e.g. AES, RSA).
It is solely for layered encodings, obfuscation, and weak ciphers. When input looks
like genuine ciphertext it says so rather than guessing.

## Installation

```bash
pip install -e .
```

## Usage

### CLI

```bash
python -m layered_decoder "SGVsbG8gV29ybGQ="
python -m layered_decoder -f encoded.txt
layered-decoder --try-branches -i "..."
layered-decoder --force 1:rot13 -i "uryyb"
layered-decoder --json -i "..."
layered-decoder --flag-prefix 'H4G{' --json -i "..."
layered-decoder --force 1:RepeatingKeyXor -i "..."
```

A file may contain several payloads, either one per line or as `label: payload`
lines. Comments (`#`) and blank lines are ignored. A payload containing a colon is
not mistaken for a label, and `scheme://` URLs pass through intact.

### Python API

```python
from layered_decoder import LayeredDecoder

result = LayeredDecoder().decode("SGVsbG8gV29ybGQ=")
print(result.final_output)   # rendered text
print(result.final_bytes)    # raw bytes, lossless
print(result.status)         # the honest verdict
```

`final_output` is a human-readable render. `final_bytes` is the real value and
should be used for any further processing.

## Architecture

### Everything is bytes until the last step

`Layer.detect(data: bytes) -> float` and `Layer.decode(data: bytes) -> list[bytes]`.
`str` appears only in `cli.py`, at the point of display.

This is not stylistic. XOR output is arbitrary binary, so a `str` pipeline either
raises on `.decode('utf-8')` or silently mangles bytes through latin-1 *before*
the scorer gets to rank them. A base64-wrapped binary payload decodes to bytes no
text decoder will accept.

### Two families of layer

| Family | Detection | `decode` returns | Members |
|---|---|---|---|
| **Structural** | Real evidence: charset, length, padding | 0 or 1 candidate | Base64, Base32, Hex, URL, Binary, Octal, AsciiDecimal, Morse, Reversal, Gzip/Zlib, SymbolNoise, SplitInterleave |
| **Statistical** | A constant 0.5 when sufficient input exists | Candidates for central scoring | Caesar (26), XorSingleByte (255), Vigenere, RepeatingKeyXor |

For statistical layers, detection and decoding deliberately collapse into one
brute-force-and-score step. You cannot identify an XOR key before trying it, and
a detector that pretends otherwise is worse than useless. Every candidate from
every layer is scored by the same function, which is what lets 255 XOR keys and
one Base64 decode compete fairly.

### The scorer

`layered_decoder/scoring.py`, operating on `bytes`:

```
0.3 * printable_ratio
0.4 * english_word_ratio      dictionary words, by token
0.5 * flag_pattern_score      a `word{...}` flag
0.2 * bigram_structure        English letter statistics
- 0.2 * entropy_penalty
1.0 * exact_known_prefix      anchored, case-sensitive flag prefix
```

Two of these terms exist because of measured failures, not theory:

- **`bigram_structure`.** Every other term is invariant under a byte permutation
  of the lowercase range, which is what several XOR keys do. Decoding
  `H3G{wELCome_b4ck_wr0ng_fl4G}` under keys 0x2A, 0x3C and 0x34 yields equally
  printable text with identical entropy and identical flag *shape*, so they tie on
  everything but the dictionary — and a flag body is never in the dictionary.
  Measured: the correct key ranked 3rd of 256. With this term, 1st.
- **`FLAG_PATTERN` is deliberately strict.** The obvious `[A-Za-z0-9_]+\{.*\}` is
  wrong here: `.*` crosses any character, so it fires on ~4.65% of printable XOR
  garbage. Since `flag_bonus` (0.5) exceeds the entire `dict_score` term (max
  0.4), one spurious hit outranks any amount of clean English — and the beam is a
  maximisation search, so it will walk into one. Measured false positives:
  4.65% → 0.00%.

Set `scoring.USE_BIGRAM_TERM = False` to A/B against the un-augmented formula.

The default exact prefixes are `flag{` and `FLAG{`. Add prefixes with repeatable
`--flag-prefix 'H4G{'` arguments or
`LayeredDecoder(flag_prefixes=("H4G{",))`. Prefixes are literal strings, not regexes;
they affect ranking, and do not establish that a decode is correct. The bonus
also requires a closing brace and a body of letters, digits, underscores, or
hyphens, so a prefix followed by malformed noise receives no bonus.

### Repeating-key XOR

`RepeatingKeyXor` considers key lengths 2 through `min(20, len(data)//4)`.
It ranks lengths by average pairwise normalized Hamming distance over the first
four to eight complete blocks, eliminates lengths with any column that cannot
decode entirely to ASCII bytes `0x20` through `0x7E`, and keeps the five best
surviving lengths. Each valid column key is ranked using chi-squared distance
against the [English letter-frequency table](https://www.usna.edu/Users/cs/wcbrown/courses/si110AY13S/resources/ceasar-shift/freqAnalysis.html).

For each length, up to 50,000 printable-key combinations are exhaustively decoded
and scored as complete texts. Larger spaces use a bounded search of 256 complete
keys ordered by column scores, plus up to 256 per compatible known prefix. Every
visited complete text uses the central scorer. This fallback can miss the correct
key; per-column rankings alone never select the final answer. Constant-byte keys
are left to `XorSingleByte`.

Exhaustive scoring runs on the full ciphertext. Large inputs and segmentation
fallback (which cracks multiple possible splits) can be substantially slower.

Steps report `key_hex`, `key_length`, and `low_confidence_short_ciphertext`.
When ciphertext contains fewer than ten bytes per recovered key byte, the result
is `status: "best_guess"`, `confidence: "low"`, and
`low_confidence_short_ciphertext: true`, even if a flag matches or the layer was
forced. This warning also propagates through later steps and combined results.
The length check uses the shortest repeating period of the recovered key.

### Decryption key requirement

Recognized OpenSSL salted envelopes and structurally plausible Fernet tokens
return `status: "key_required"` and `stopped_reason: "key_required"`.
The human-readable report and JSON include `key_required`, `encryption_format`,
`key_requirement_evidence`, and `required_decryption_info`. Recognition works
on raw bytes and after supported encoding layers, preserving the ciphertext in
`final_bytes` and stopping further guessing or splitting.

This is format evidence, not authentication: a forged header can match, and a
Fernet HMAC cannot be verified without the key. Format recognition is reported
with medium confidence. The OpenSSL header does not identify its cipher or key
derivation settings. Fernet recognition checks the version, complete field
lengths, block alignment, and a positive timestamp through year 9999, following
the [Fernet specification](https://github.com/fernet/spec/blob/master/Spec.md).
OpenSSL salt handling is described in the
[OpenSSL enc documentation](https://docs.openssl.org/master/man1/openssl-enc/).

High entropy alone never sets `key_required`: unknown data may be compressed,
corrupt, or encrypted without an identifiable format. `false` means no supported
key-requiring format was identified, not proof that no key is needed. This
feature reports requirements; it does not accept keys or perform decryption.
Quiet mode continues to emit only the final output; use normal or JSON output
for evidence.

### Beam search

```python
frontier = [(data, score, steps)]
for depth in range(max_depth):
    candidates = [expand(node, layer) for node in frontier for layer in ALL_LAYERS]
    frontier = top_k_diverse(candidates)
```

The old engine was greedy and required every step to improve the score. That kills
valid chains whose intermediate looks worse than its input — a base64-wrapped hex
string scores 0.209 against the wrapper's 0.241, then decodes to plaintext at
0.762. The beam only requires the best *final* state to beat the incumbent.

Two rules keep family B from swamping the search:

- **Per-layer diversity.** Without a cap, XOR's 255 candidates take every beam
  slot and the correct first step is pruned before it can pay off.
- **Strong structural evidence preempts brute force.** Family B's `detect` is a
  constant, so its candidates carry no evidence; a structural layer scoring above
  that constant is asserting something real. If the encoding is visible, do not
  guess at ciphers.

Layer priority is a tiebreak, not a schedule. Decode order is the reverse of encode
order and the search discovers it.

### Termination

Reported as `result.status`:

| Score | Status |
|---|---|
| ≥ 0.60 | `solved` |
| ≥ 0.35 | `partial_decode_low_confidence` |
| < 0.35 | `likely_real_cryptography_or_unrecoverable` |
| Short repeating-key XOR evidence, regardless of score | `best_guess` |

`solved` additionally requires the output to look like a genuine destination:
printable, no implausibly long token, and a corroborated flag. Without those checks
a partially-decoded input clears the numeric bar while still containing a large
undecoded chunk.

Confidence tracks the *evidence*, not just the score. A lucky single XOR key can
reach 0.5 by stumbling onto a dictionary word; a result built entirely from
brute-force guesses is reported as low confidence unless unambiguously solved.

## Supported Encodings

- Base64, URL-safe Base64, Base32 (including unpadded input)
- Hex (with `0x`, `\x`, colon and whitespace separators) and URL percent-encoding
- Caesar/ROT-N (all 25 shifts), single-byte XOR (all 255 keys), Vigenère with
  index-of-coincidence key-length guessing
- Repeating-key XOR with printable-ASCII elimination and joint key scoring
- Morse, binary, octal, decimal ASCII, reversal
- Gzip/zlib, including Base64-wrapped compressed payloads
- Odd/even stream splitting, removable every-Nth-character noise

## Safety

- Decompression output is capped at 8 MiB, reading `MAX+1` and rejecting on
  overflow. Reading exactly `MAX` would silently truncate a bomb into a
  plausible-looking partial result.
- `CryptoGuard` refuses to let the search claim it broke real cryptography. It
  uses a sample-size-aware entropy threshold (`log2(min(len, 256))`) because a
  fixed threshold misfires on short inputs, and it explicitly excludes gzip/zlib
  headers — compressed data is high-entropy but is an *encoding*.
- When the guard fires, the bounded family-B search still runs: high-entropy
  bytes are exactly what single-byte XOR looks like, and refusing them would
  report "real cryptography" for a payload one XOR key from solved. Deeper
  structural search remains available.

## Known limitations

- **Undivided concatenation.** The beam has no split operator, so on an
  *undelimited* concatenation of two payloads its best reachable state is
  whichever brute-force accident scores highest. If that accident is flag-shaped
  enough to pass the finish check, segmentation is skipped and the result is
  wrong. Labelled or line-separated input does not have this problem — that is
  what the CLI produces, and what labelled challenge files contain. Offering
  segmentation unconditionally was measured and is worse: it splits single
  payloads at coincidental boundaries.
- **Compound detection is advisory.** The character-class changepoint signal is
  not separable in practice — a single base64 payload shows the same 0.75 shift as
  two concatenated ones. `likely_compound_input` is a hint, not a decision.
- **Vigenère needs volume.** Unigram frequency analysis is unreliable below
  roughly 50 letters per column; short inputs are declined rather than guessed.
  Several key lengths are returned and the scorer chooses, because a multiple of
  the true length is not interchangeable with it.
- **Repeating-key XOR assumes printable ASCII.** Tabs, newlines, non-ASCII text,
  and binary plaintext are excluded by its hard elimination pass. Inputs shorter
  than eight bytes have no eligible key length. Long or ambiguous keys can exceed
  the exhaustive-search budget, and short ciphertext remains a best guess.
- **Binary round-trip is not a goal.** The scorer measures text quality, so
  arbitrary binary scores 0.0 and the beam prefers the encoded form. The pipeline
  handles binary without raising or emitting mojibake, but it will not prefer it.

## Tests

```bash
python -m pytest -q
```

No runtime dependencies. Coverage is behavioural rather than
implementation-shaped: the scorer tests pin the measured properties above, and
`tests/test_engine.py` asserts the search properties (an intermediate may score
worse than its input; structural evidence preempts brute force; real ciphertext
is refused) rather than a fixed layer sequence.
