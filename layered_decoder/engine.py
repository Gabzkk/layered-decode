"""Beam search over the layer graph.

Replaces the old greedy loop. Greedy accepted a step only if the score did not
decrease, which killed valid chains whose intermediate looked worse than their
input -- a base64-wrapped hex string scores *below* the base64 wrapper, yet
decodes to plaintext immediately after. The beam only requires the best *final*
state to beat the incumbent, so such chains survive.

The engine does not know the encode order and does not need to. Every layer
offers candidates at every state; wrong-order attempts score badly and fall out
of the beam. That is why layer priority is a tiebreak rather than a schedule.
"""

import base64
import re
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .key_requirement import identify_key_requirement
from .models import DecodeResult, DecodeStep
from .scoring import (
    DEFAULT_FLAG_PREFIXES,
    cached_score,
    clear_score_cache as _orig_clear_score_cache,
    english_word_ratio,
    flag_content_score,
    flag_pattern_score,
    is_verified_strict_flag,
    looks_structured,
    score,
    shannon_entropy,
)
from .detectors import Layer, StopSignal, get_all_detectors
from .detectors.xor import STATISTICAL_CONFIDENCE

# Layers that are their own inverses. Guard against immediate re-application.
INVOLUTIONS = frozenset({"Reversal", "ROT13"})

# Cap on candidates that receive the expensive statistical search pass each round.
STATISTICAL_CANDIDATE_CAP = 3

# A candidate must clear this to enter the beam. Guards against a statistical
# layer's 255-key spray drowning out structural signal.
MIN_CANDIDATE_SCORE = 0.05

# Ceiling on beam slots a single layer may take.
#
# Without this, family B breaks the search. A correct first step often looks
# *worse* than brute-force noise: base64-of-hex scores 0.209 while the best of
# 255 XOR keys scores 0.236. Under a pure top-N beam, all 255 keys outrank the
# real step, the beam fills with garbage, and the correct path is pruned at
# depth 1 and never recovers. Capping slots per layer guarantees that if a
# structural layer produced a viable candidate, it is represented in the beam
# regardless of how many XOR keys scored higher.
#
# 1 means "one slot per distinct layer" -- the beam stays diverse, which is what
# lets depth-1 dips survive to pay off at depth 2.
MAX_SLOTS_PER_LAYER = 1

# A structural layer reporting at least this confidence preempts family B.
#
# Family B's detect is a *constant* 0.5, meaning "cannot be ruled out" -- it
# asserts nothing. A structural layer scoring above that constant is asserting
# positive evidence (Reversal at 0.75, Base64 at 0.90), so its candidate
# outranks anything a blind 255-key spray produces.
#
# The threshold is the family-B constant itself, not a magic number. Without
# the rule, brute force wins on score alone and prunes the correct path at depth
# 1: on the real challenge Base64 (0.209) ranks 4th behind three brute-force
# layers, and Reversal (0.214, a pure permutation that cannot improve the score
# on its own) ranks 4th the same way. Both are correct first steps.
#
# This is a precedence rule between families, not a hardcoded decode order --
# the search still discovers ordering within each family.
STRONG_STRUCTURAL_CONFIDENCE = 0.80

# Termination bands. Calibrated against this scorer's actual distribution --
# see tests/test_beam_search.py::test_score_thresholds_match_distribution.
SOLVED_SCORE = 0.60
PARTIAL_SCORE = 0.35

# Shortest input worth offering to the segmenter. Below this a "split" is more
# likely to be a coincidental boundary than a real concatenation.
COMPOUND_MIN_LENGTH = 32

# Occam's-razor discount applied per statistical step when ranking complete
# solutions.
#
# A structural step is evidence -- Base64 matched a charset. A statistical step
# is a guess that happened to score well, and with 255 XOR keys per node some of
# them will score well by luck. Ranking them equally lets a lucky 4-step
# brute-force chain beat a correct 2-step structural one, which is how a
# compound input ends up "solved" as garbage: its real answer needs a split, and
# splitting is not a layer, so the only reachable high-scoring states are
# brute-force accidents.
STAT_STEP_PENALTY = 0.05
SHORT_KEY_GUESS_PENALTY = 0.10

# Longest whitespace-delimited token still plausible as a real word. Prose does
# not produce 40+ character words; undecoded base64/hex does. Used to tell a
# finished decode from a partially-decoded one.
MAX_PLAUSIBLE_TOKEN = 40

# Minimum share of alphanumeric characters before a flag-shaped match is
# believed. Real CTF flags sit at 0.80-0.82; coincidental brace pairs in XOR
# noise land at 0.38-0.72.
FLAG_ALNUM_RATIO = 0.75

# How much better a challenger must be than the incumbent before it replaces it.
# A +0.06 nudge is inside scorer noise; a +0.40 hit is not.
MIN_BEST_IMPROVEMENT = 0.05

__all__ = ["LayeredDecoder", "SOLVED_SCORE", "PARTIAL_SCORE", "clear_score_cache"]

_B64_RE = re.compile(rb"^[A-Za-z0-9+/]+={0,2}$")


def _score_caesar_shift(produced: bytes, flag_prefixes: Tuple[str, ...] = DEFAULT_FLAG_PREFIXES) -> float:
    """Evaluate if a Caesar shift unmasks a valid Base64 payload or flag."""
    stripped = produced.strip()
    if len(stripped) >= 16 and len(stripped) % 4 == 0 and _B64_RE.match(stripped):
        try:
            raw = base64.b64decode(stripped, validate=True)
            if raw and len(raw) >= 8:
                raw_score = cached_score(raw, flag_prefixes)
                if raw_score >= 0.4:
                    return raw_score * 0.90
                for k in range(1, 256):
                    cand = bytes(b ^ k for b in raw)
                    try:
                        text = cand.decode("utf-8")
                        if flag_pattern_score(text) > 0 or english_word_ratio(text) >= 0.4:
                            xs = cached_score(cand, flag_prefixes)
                            if xs >= 0.4:
                                return xs * 0.90
                    except Exception:
                        pass
        except Exception:
            pass
    return 0.0


def clear_score_cache() -> None:
    _orig_clear_score_cache()


class _State:
    """A point in the search, carrying everything needed to report it."""

    __slots__ = ("data", "score", "rank", "layer_names", "steps", "ancestors", "unscorable")

    def __init__(
        self,
        data: bytes,
        value: float,
        layer_names: Tuple[str, ...],
        steps: Tuple[DecodeStep, ...],
        ancestors: FrozenSet[bytes],
        unscorable: bool = False,
    ):
        self.data = data
        self.score = value
        self.layer_names = layer_names
        self.steps = steps
        self.ancestors = ancestors
        # True when `score` returned 0.0 because the bytes are not UTF-8, not
        # because the candidate is worthless. Such states are waypoints: a later
        # layer may consume them, so the beam must keep one.
        self.unscorable = unscorable
        # `rank` is the *discounted* score used to choose between complete
        # solutions: `score` minus a penalty per statistical step. It is
        # recomputed from the step list rather than accumulated across steps --
        # accumulating makes a long chain of mediocre guesses look better than a
        # short correct one, which is exactly backwards.
        self.rank = (
            value
            - STAT_STEP_PENALTY * sum(1 for s in steps if s.statistical)
            - SHORT_KEY_GUESS_PENALTY * sum(s.low_confidence_short_ciphertext for s in steps)
        )

    @property
    def n_statistical(self) -> int:
        return sum(1 for step in self.steps if step.statistical)


class LayeredDecoder:
    def __init__(
        self,
        max_depth: int = 15,
        min_confidence: float = 0.3,
        detectors: Optional[List[Layer]] = None,
        beam_width: int = 3,
        flag_prefixes: Optional[Tuple[str, ...]] = None,
        key: Optional[str] = None,
        wordlist: Optional[Sequence[str] | str] = None,
        bruteforce: bool = False,
    ):
        self.max_depth = max_depth
        self.min_confidence = min_confidence
        self.beam_width = max(1, beam_width)
        self.flag_prefixes = DEFAULT_FLAG_PREFIXES + tuple(flag_prefixes or ())
        self.detectors: List[Layer] = (
            detectors if detectors is not None else get_all_detectors(self.flag_prefixes)
        )
        self.stop_signals: List[StopSignal] = [
            d for d in self.detectors if isinstance(d, StopSignal)
        ]
        self.decode_layers: List[Layer] = [
            d for d in self.detectors if not isinstance(d, StopSignal)
        ]
        self.key = key
        self.wordlist = wordlist
        self.bruteforce = bruteforce

    # ── public API ──────────────────────────────────────────────────

    def decode(
        self,
        input_string,
        forced_layers: Optional[Dict[int, str]] = None,
        allow_segmentation: bool = True,
        key: Optional[str] = None,
        wordlist: Optional[Sequence[str] | str] = None,
        bruteforce: Optional[bool] = None,
    ) -> DecodeResult:
        data = _as_bytes(input_string)
        res = self._beam(data, forced_layers or {})

        active_key = key if key is not None else self.key
        active_wordlist = wordlist if wordlist is not None else self.wordlist
        active_bruteforce = bruteforce if bruteforce is not None else self.bruteforce

        if res.key_required and (active_key is not None or active_bruteforce or active_wordlist is not None):
            from .crypto_decryptor import decrypt_envelope, bruteforce_envelope, continue_with_decrypted
            decrypted_info = None
            if active_key is not None:
                dec = decrypt_envelope(res.final_bytes, res.encryption_format, active_key)
                if dec is not None:
                    pt, method = dec
                    decrypted_info = (pt, active_key, method)
            if decrypted_info is None and (active_bruteforce or active_wordlist is not None):
                ctx_words = [s.output_value for s in res.steps]
                bf_res = bruteforce_envelope(
                    res.final_bytes,
                    res.encryption_format,
                    wordlist=active_wordlist,
                    contextual_words=ctx_words,
                )
                if bf_res is not None:
                    decrypted_info = bf_res

            if decrypted_info is not None:
                pt, used_key, method = decrypted_info
                res = continue_with_decrypted(
                    self,
                    res,
                    pt,
                    used_key,
                    method,
                    allow_segmentation=allow_segmentation,
                )

        if res.key_required or forced_layers or not allow_segmentation:
            return res

        from .segmentation import try_segmentation_decode, detect_compound_indicators

        # Segmentation is a fallback for when the beam could not finish.
        if res.status != "solved" and len(data) >= COMPOUND_MIN_LENGTH:
            segmented = try_segmentation_decode(
                data,
                lambda d: self.decode(
                    d,
                    forced_layers=None,
                    allow_segmentation=False,
                    key=active_key,
                    wordlist=active_wordlist,
                    bruteforce=active_bruteforce,
                ),
                self.decode_layers,
                flag_prefixes=self.flag_prefixes,
            )
            if segmented is not None:
                return segmented

        partial_match, likely_compound = detect_compound_indicators(
            data, self.decode_layers
        )
        if partial_match or likely_compound:
            res.partial_match_detected = partial_match
            res.likely_compound_input = likely_compound
        return res

    def decode_with_branches(
        self,
        input_string,
        forced_layers: Optional[Dict[int, str]] = None,
        key: Optional[str] = None,
        wordlist: Optional[Sequence[str] | str] = None,
        bruteforce: Optional[bool] = None,
    ) -> DecodeResult:
        """Single beam first; fall back to split strategies if it stalled."""
        data = _as_bytes(input_string)
        base_result = self.decode(
            data,
            forced_layers,
            key=key,
            wordlist=wordlist,
            bruteforce=bruteforce,
        )

        if base_result.key_required:
            return base_result

        if len(base_result.steps) >= 3 or cached_score(base_result.final_bytes, self.flag_prefixes) >= 0.5:
            return base_result

        from .branching import try_branch_decode

        branch_result = try_branch_decode(
            data,
            lambda d: self.decode(
                d,
                forced_layers,
                key=key,
                wordlist=wordlist,
                bruteforce=bruteforce,
            ),
            flag_prefixes=self.flag_prefixes,
        )
        return branch_result if branch_result is not None else base_result

    # ── beam search ────────────────────────────────────────────────

    def _beam(self, data: bytes, forced_layers: Dict[int, str]) -> DecodeResult:
        root = _State(data, cached_score(data, self.flag_prefixes), (), (), frozenset({data}))
        if identify_key_requirement(data):
            return self._build_result(root, data, saw_crypto=True)
        best = root
        frontier: List[_State] = [root]
        saw_crypto = False
        expanded_structural: Set[bytes] = set()
        expanded_statistical: Set[bytes] = set()

        # Already-decoded input is the answer. Without this, the beam keeps
        # hunting and will happily mangle clean prose: "Just plain text here"
        # (score 0.747) loses to a coincidentally flag-shaped XOR of itself
        # (0.821), because a flat +0.5 flag bonus ignores everything else about
        # the string. Nothing should be peeled off text that is already text.
        if not forced_layers and _is_finished(data, self.flag_prefixes) and root.rank >= SOLVED_SCORE:
            return self._build_result(root, data, saw_crypto=False)

        for depth in range(1, self.max_depth + 1):
            any_progress = False

            if forced_layers.get(depth) is not None:
                candidates: List[_State] = []
                for node in frontier:
                    candidates.extend(self._expand(node, depth, forced_layers, guarded=False, family=None))
                if not candidates:
                    break
                frontier = self._select_diverse(candidates)
                frontier.sort(key=lambda s: s.rank, reverse=True)
                if frontier and frontier[0].rank > best.rank:
                    best = frontier[0]
                continue

            protected: List[_State] = []
            statistical_candidates: List[_State] = []

            # Structural pass on all candidates in frontier
            for node in frontier:
                if node.data in expanded_structural:
                    continue
                expanded_structural.add(node.data)
                guarded = self._tripped_stop_signal(node.data)
                if guarded:
                    saw_crypto = True
                new_structural = self._expand(node, depth, forced_layers, guarded=guarded, family="structural")
                for state in new_structural:
                    if state.rank > node.rank or (
                        state.steps
                        and not state.steps[-1].statistical
                        and state.steps[-1].confidence >= STRONG_STRUCTURAL_CONFIDENCE
                    ):
                        any_progress = True
                protected.extend(new_structural)

            # Fix 3: Determine candidates for statistical pass:
            # Structural nodes (including root and low-scoring waypoints like pre-XOR binary)
            # plus top-scoring nodes in frontier up to candidate cap.
            stat_nodes = []
            for n in frontier:
                if not n.steps or not n.steps[-1].statistical:
                    stat_nodes.append(n)
            for n in frontier:
                if len(stat_nodes) >= max(STATISTICAL_CANDIDATE_CAP, self.beam_width):
                    break
                if n not in stat_nodes:
                    stat_nodes.append(n)

            for node in stat_nodes:
                if node.data in expanded_statistical:
                    continue
                expanded_statistical.add(node.data)
                guarded = self._tripped_stop_signal(node.data)
                if guarded:
                    saw_crypto = True
                new_statistical = self._expand(node, depth, forced_layers, guarded=guarded, family="statistical")
                for state in new_statistical:
                    if state.rank > node.rank:
                        any_progress = True
                statistical_candidates.extend(new_statistical)

            # Fix 2:
            # protected: outputs of structural layers -- exempt from score-based pruning
            # (cap at beam_width * 3 as safety valve against unbounded growth)
            if len(protected) > self.beam_width * 3:
                protected.sort(key=lambda s: s.rank, reverse=True)
                protected = protected[: self.beam_width * 3]

            # scored_pool: statistical layer outputs + the parent candidates -- ranked and pruned to beam_width
            scored_pool_candidates = statistical_candidates + list(frontier)
            scored_pool_candidates.sort(key=lambda s: s.rank, reverse=True)
            scored_pool = self._select_diverse(scored_pool_candidates)

            # Final frontier = dedup(protected + scored_pool)
            seen: Dict[bytes, _State] = {}
            for state in protected + scored_pool:
                if state.data not in seen or state.rank > seen[state.data].rank:
                    seen[state.data] = state

            frontier = list(seen.values())
            # Fix 4: Frontier must be sorted after merging pools
            frontier.sort(key=lambda s: s.rank, reverse=True)

            if not frontier:
                break

            for candidate in frontier:
                if not candidate.n_statistical and identify_key_requirement(candidate.data):
                    return self._build_result(candidate, data, saw_crypto=True)

            challenger = frontier[0]
            is_better = challenger.rank > best.rank and (
                not _is_finished(best.data, self.flag_prefixes)
                or challenger.rank >= best.rank + MIN_BEST_IMPROVEMENT
            )
            if not is_better and best is root:
                for c in frontier:
                    if c.steps:
                        first_step = c.steps[0]
                        if (
                            not first_step.statistical
                            and first_step.confidence >= STRONG_STRUCTURAL_CONFIDENCE
                            and c.rank >= best.rank - 0.05
                        ):
                            challenger = c
                            is_better = True
                            break

            if is_better:
                best = challenger

            if best.rank >= SOLVED_SCORE and _is_finished(best.data, self.flag_prefixes):
                break

            if not any_progress and depth > 1 and not protected:
                break

        return self._build_result(best, data, saw_crypto)

    def _select_diverse(self, candidates: List[_State]) -> List[_State]:
        """Fill the beam, capped per layer, and always keeping one waypoint.

        Two rules, both load-bearing:

        * **Per-layer cap.** XOR's 255 candidates would otherwise take every slot
          and the correct first step would be pruned before it could pay off.
        * **One unscorable waypoint is always retained.** A candidate that is not
          valid UTF-8 scores 0.0 because `score` measures *text*, not because it
          is worthless -- and it is exactly the state a later layer needs. A
          base64-wrapped XOR payload hits this: base64 decodes to binary, that
          binary scores 0.0, and in a width-3 beam it loses to 0.28-scoring
          brute-force noise and dies before the XOR layer can consume it. The
          chain is not unlikely, it is unreachable.
        """
        chosen: List[_State] = []
        taken: Dict[str, int] = {}

        for state in candidates:
            layer = state.layer_names[-1] if state.layer_names else ""
            if taken.get(layer, 0) >= MAX_SLOTS_PER_LAYER:
                continue
            taken[layer] = taken.get(layer, 0) + 1
            chosen.append(state)
            if len(chosen) >= self.beam_width:
                break

        # Reserve a slot for the best waypoint `score` could not judge.
        if not any(s.unscorable for s in chosen):
            waypoints = [s for s in candidates if s.unscorable]
            if waypoints:
                if len(chosen) >= self.beam_width and chosen[-1].rank < MIN_CANDIDATE_SCORE:
                    chosen[-1] = waypoints[0]
                else:
                    chosen.append(waypoints[0])
        return chosen

    def _expand(
        self,
        node: _State,
        depth: int,
        forced_layers: Dict[int, str],
        guarded: bool = False,
        family: Optional[str] = None,
    ) -> List[_State]:
        forced_name = forced_layers.get(depth)
        if forced_name is not None:
            # A forced layer pins this depth: only that layer runs, and its
            # confidence floor is bypassed -- the user already decided.
            target = self._find_layer(forced_name)
            if target is None:
                return []
            layers: List[Layer] = [target]
        elif family == "structural":
            layers = [l for l in self.decode_layers if not l.family.is_statistical]
        elif family == "statistical":
            layers = [l for l in self.decode_layers if l.family.is_statistical]
        else:
            layers = self.decode_layers

        out: List[_State] = []
        structural_hit = self._strong_structural_match(node.data)
        wrapper_only_unscorable = False
        if structural_hit:
            strong_structural = [
                l
                for l in self.decode_layers
                if (not l.family.is_statistical)
                and l.detect(node.data) >= STRONG_STRUCTURAL_CONFIDENCE
            ]
            wrapper_only_unscorable = bool(strong_structural) and all(
                l.name in ("Base64", "Base64Url", "Hex")
                and all(
                    cached_score(out_bytes, self.flag_prefixes) < MIN_CANDIDATE_SCORE
                    for out_bytes in l.decode(node.data)
                )
                for l in strong_structural
            )

        for layer in layers:
            if forced_name is None and layer.detect(node.data) < self.min_confidence:
                continue
            # Fix 1: Guard against immediate re-application of involutions or repeated statistical layers
            if forced_name is None and node.layer_names:
                last_layer = node.layer_names[-1]
                if layer.name in INVOLUTIONS and last_layer == layer.name:
                    continue
                if layer.family.is_statistical and layer.name in node.layer_names:
                    continue
            # Strong structural match preempts brute force on wrapper text,
            # except Caesar when the wrapper decodes to unscorable binary (Caesar-shifted Base64).
            if (
                forced_name is None
                and layer.family.is_statistical
                and structural_hit
                and not (wrapper_only_unscorable and layer.name == "Caesar")
            ):
                continue
            for produced in layer.decode(node.data):
                state = self._accept(layer, node, produced, forced=forced_name is not None)
                if state is not None:
                    out.append(state)
        return out

    def _strong_structural_match(self, data: bytes) -> bool:
        """True if a structural layer is confident about this node."""
        return any(
            (not layer.family.is_statistical)
            and layer.detect(data) >= STRONG_STRUCTURAL_CONFIDENCE
            for layer in self.decode_layers
        )

    def _accept(
        self,
        layer: Layer,
        node: _State,
        produced: bytes,
        forced: bool,
    ) -> Optional[_State]:
        if not produced or produced == node.data:
            return None
        if produced in node.ancestors:  # cycle guard
            return None

        value = cached_score(produced, self.flag_prefixes)
        if layer.name == "Caesar":
            bonus = _score_caesar_shift(produced, self.flag_prefixes)
            if bonus > value:
                value = bonus

        # `score` cannot judge non-UTF-8, but a gzip wrapper or re-encoded
        # binary is a legitimate waypoint that a later layer can consume.
        unscorable = value < MIN_CANDIDATE_SCORE
        if unscorable and not looks_structured(produced):
            return None

        confidence = 1.0 if forced else layer.detect(node.data)
        evidence = layer.candidate_evidence(node.data, produced)
        if evidence.low_confidence_short_ciphertext:
            confidence = min(confidence, 0.25)
        step = DecodeStep(
            layer_number=len(node.steps) + 1,
            encoding_detected=layer.name,
            confidence=confidence,
            input_value=_render(node.data),
            output_value=_render(produced),
            score_before=node.score,
            score_after=value,
            entropy_before=shannon_entropy(node.data),
            entropy_after=shannon_entropy(produced),
            forced=forced,
            statistical=layer.family.is_statistical,
            key_hex=evidence.key.hex() if evidence.key is not None else None,
            key_length=len(evidence.key) if evidence.key is not None else None,
            low_confidence_short_ciphertext=evidence.low_confidence_short_ciphertext,
        )
        return _State(
            produced,
            value,
            node.layer_names + (layer.name,),
            node.steps + (step,),
            node.ancestors | {produced},
            unscorable=unscorable,
        )

    def _tripped_stop_signal(self, data: bytes) -> bool:
        return any(signal.detect(data) >= 0.8 for signal in self.stop_signals)

    def _find_layer(self, requested_name: str) -> Optional[Layer]:
        normalized = requested_name.lower().replace("_", "").replace("-", "")
        if normalized.startswith("rot") and normalized[3:].isdigit():
            normalized = "caesar"
        normalized = _ALIASES.get(normalized, normalized)
        for layer in self.detectors:
            if layer.name.lower().replace("/", "") == normalized.replace("/", ""):
                return layer
        return None

    # ── result assembly ────────────────────────────────────────────

    def _build_result(
        self, best: _State, original: bytes, saw_crypto: bool
    ) -> DecodeResult:
        # "solved" requires all three: a good raw score, a positive *rank* after
        # the statistical-step discount, and output that looks like a genuine
        # destination. Any one alone is insufficient -- see SOLVED_SCORE,
        # STAT_STEP_PENALTY and _is_finished.
        finished = _is_finished(best.data, self.flag_prefixes)
        if finished and best.rank >= SOLVED_SCORE:
            status = "solved"
        elif best.rank >= PARTIAL_SCORE:
            status = "partial_decode_low_confidence"
        else:
            status = "likely_real_cryptography_or_unrecoverable"
        stopped = "completed" if status == "solved" else "no_improvement"

        # Crypto only wins the verdict if we never found anything worth having.
        if saw_crypto and best.rank < PARTIAL_SCORE:
            status = "likely_real_cryptography_or_unrecoverable"
            stopped = "crypto_detected"

        requirement = identify_key_requirement(best.data)
        if requirement:
            status = "key_required"
            stopped = "key_required"

        return DecodeResult(
            key_required=requirement is not None,
            encryption_format=requirement.format_name if requirement else None,
            key_requirement_evidence=requirement.evidence if requirement else None,
            required_decryption_info=requirement.required_info if requirement else None,
            layers_detected=list(best.layer_names),
            intermediate_values=[_render(original)] + [s.output_value for s in best.steps],
            steps=list(best.steps),
            branches=None,
            final_output=_render(best.data),
            final_bytes=best.data,
            confidence="medium" if requirement else self._confidence_for(best, status),
            stopped_reason=stopped,
            status=status,
        )

    @staticmethod
    def _confidence_for(best: _State, status: str) -> str:
        """Confidence tracks the *evidence*, not just the score.

        A lucky single XOR key on undecodable input can reach 0.5 by stumbling
        onto a real dictionary word, which is not the same as having decoded
        something. A result built entirely from brute-force guesses is reported
        as low confidence unless it is unambiguously solved.
        """
        if status == "solved":
            return "high"
        if status == "partial_decode_low_confidence" and best.n_statistical < len(best.steps):
            # At least one structural (evidenced) step, so the read is grounded.
            return "medium"
        return "low"


_ALIASES = {
    "url": "urlencoding",
    "ascii": "asciidecimal",
    "decimal": "asciidecimal",
    "gzip": "gzipzlib",
    "zlib": "gzipzlib",
    "compression": "gzipzlib",
    "rot13": "caesar",
    "rotn": "caesar",
    "rotation": "caesar",
    "reverse": "reversal",
    "b64": "base64",
    "b32": "base32",
    "noise": "nthcharnoise",
    "nthchar": "nthcharnoise",
    "xor": "xorsinglebyte",
}


def _is_finished(data: bytes, flag_prefixes: Tuple[str, ...] = DEFAULT_FLAG_PREFIXES) -> bool:
    """True when the state looks like a genuine destination, not a waypoint.

    A high score alone is not enough. Three separate failure modes needed
    closing, each found by a real mis-ranking rather than by inspection:

    * Printable junk scores well, so require dictionary or flag content.
    * A *partially* decoded compound clears a token-ratio test -- one 95-char
      garbage token beside a few real words still counts -- so reject any token
      too long to be a word.
    * A flat +0.5 flag bonus ignores the rest of the string, so coincidental
      brace pairs in XOR noise beat real prose. Requiring the text to be
      alphanumeric-dense before believing a flag separates them cleanly:
      real flags measure 0.80-0.82, noise 0.38-0.72.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False

    # A "solved" result is something we will show as text, so it must actually be
    # text. Control characters mean the decode ended on binary we cannot honestly
    # present -- and in practice they only appear in brute-force output, never in
    # a real answer.
    if not all(c.isprintable() or c in "\n\r\t" for c in text):
        return False

    if is_verified_strict_flag(text, flag_prefixes):
        alnum = sum(1 for c in text if c.isalnum())
        return alnum / len(text) >= FLAG_ALNUM_RATIO

    if flag_content_score(text, flag_prefixes) == 1.0:
        alnum = sum(1 for c in text if c.isalnum())
        return alnum / len(text) >= FLAG_ALNUM_RATIO

    tokens = [t for t in text.split() if t]
    if not tokens:
        return False
    if any(len(t) > MAX_PLAUSIBLE_TOKEN for t in tokens):
        return False
    return english_word_ratio(text) >= 0.4


def _as_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return str(value).encode("utf-8")


def _render(data: bytes) -> str:
    """The last step before a human sees anything: bytes to text.

    Undecodable bytes come back as visible escapes rather than mojibake that
    could be mistaken for a real answer.
    """
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="backslashreplace")
