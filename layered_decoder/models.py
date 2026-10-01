from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DecodeStep:
    layer_number: int
    encoding_detected: str
    confidence: float  # 0.0-1.0
    input_value: str
    output_value: str
    score_before: float
    score_after: float
    entropy_before: float
    entropy_after: float
    forced: bool = False  # True if user forced this layer
    statistical: bool = False  # True for family B (brute-force-and-score) layers
    key_hex: Optional[str] = None
    key_length: Optional[int] = None
    low_confidence_short_ciphertext: bool = False


@dataclass
class BranchResult:
    branch_id: str  # e.g. 'odd_chars', 'even_chars'
    steps: List[DecodeStep]
    final_output: str


@dataclass
class DecodeResult:
    layers_detected: List[str]
    intermediate_values: List[str]
    steps: List[DecodeStep]
    branches: Optional[List[BranchResult]]
    final_output: str
    confidence: str  # 'high', 'medium', 'low'

    # The honesty verdict from the termination check:
    #   'solved' | 'best_guess' | 'partial_decode_low_confidence'
    #   | 'likely_real_cryptography_or_unrecoverable' | 'key_required'
    status: str = "likely_real_cryptography_or_unrecoverable"

    # Which of the two reasons the search stopped. `status` says how good the
    # answer is; this says why we stopped looking.
    #   'completed' | 'no_improvement' | 'max_depth' | 'crypto_detected'
    #   | 'invalid_forced_layer' | 'forced_decode_failed'
    #   | 'segmented_completed' | 'branch_recombined' | 'key_required'
    stopped_reason: str = "no_improvement"

    # Raw bytes behind `final_output`. `final_output` is a lossy render (the CLI
    # shows text), so anything that needs to keep decoding must use this.
    final_bytes: bytes = b""

    key_required: bool = False
    encryption_format: Optional[str] = None
    key_requirement_evidence: Optional[str] = None
    required_decryption_info: Optional[str] = None

    partial_match_detected: bool = False
    likely_compound_input: bool = False
    segments: Optional[List["DecodeResult"]] = None
    low_confidence_short_ciphertext: bool = False

    def __post_init__(self) -> None:
        self.low_confidence_short_ciphertext = (
            self.low_confidence_short_ciphertext
            or any(step.low_confidence_short_ciphertext for step in self.steps)
            or any(segment.low_confidence_short_ciphertext for segment in self.segments or [])
            or any(step.low_confidence_short_ciphertext
                   for branch in self.branches or [] for step in branch.steps)
        )
        if self.low_confidence_short_ciphertext and not self.key_required:
            self.status = "best_guess"
            self.confidence = "low"

    def to_dict(self) -> dict:
        def serialize_branch(branch: BranchResult) -> dict:
            return {
                "branch_id": branch.branch_id,
                "steps": [step.__dict__ for step in branch.steps],
                "final_output": branch.final_output,
            }

        d = {
            "layers_detected": self.layers_detected,
            "intermediate_values": self.intermediate_values,
            "steps": [s.__dict__ for s in self.steps],
            "branches": (
                [serialize_branch(b) for b in self.branches] if self.branches else None
            ),
            "final_output": self.final_output,
            "final_bytes_hex": self.final_bytes.hex(),
            "confidence": self.confidence,
            "status": self.status,
            "key_required": self.key_required,
            "encryption_format": self.encryption_format,
            "key_requirement_evidence": self.key_requirement_evidence,
            "required_decryption_info": self.required_decryption_info,
            "stopped_reason": self.stopped_reason,
            "partial_match_detected": self.partial_match_detected,
            "likely_compound_input": self.likely_compound_input,
            "low_confidence_short_ciphertext": self.low_confidence_short_ciphertext,
        }
        if self.segments is not None:
            d["segments"] = [s.to_dict() for s in self.segments]
        return d

    def summary(self) -> str:
        lines = [
            f"Decode Result (Confidence: {self.confidence})",
            f"Status: {self.status}",
            f"Stopped Reason: {self.stopped_reason}",
        ]
        if self.key_required:
            lines.extend([
                f"Decryption key required: {self.encryption_format}",
                f"Evidence: {self.key_requirement_evidence}",
                f"Needed: {self.required_decryption_info}",
            ])
        if self.likely_compound_input:
            lines.append("Compound Input: True")
        if self.low_confidence_short_ciphertext:
            lines.append("low_confidence_short_ciphertext: true")
        lines.append(f"Final Output: {self.final_output}")
        lines.append("-" * 40)
        if self.segments:
            lines.append(f"Compound Segments ({len(self.segments)}):")
            for idx, seg in enumerate(self.segments, 1):
                lines.append(
                    f"  [Segment {idx}] Layers: {' -> '.join(seg.layers_detected)}"
                )
                lines.append(f"    Output: {seg.final_output}")
                lines.append(
                    f"    Status: {seg.status} (conf: {seg.confidence})"
                )
        else:
            lines.append("Steps:")
            for step in self.steps:
                lines.append(
                    f"Layer {step.layer_number}: {step.encoding_detected} "
                    f"(conf: {step.confidence:.2f})"
                )
                lines.append(
                    f"  Score: {step.score_before:.2f} -> {step.score_after:.2f} | "
                    f"Entropy: {step.entropy_before:.2f} -> {step.entropy_after:.2f}"
                )
            if self.branches:
                lines.append("Branches:")
                for branch in self.branches:
                    lines.append(
                        f"  Branch {branch.branch_id}: {len(branch.steps)} steps, "
                        f"output: {branch.final_output[:50]}..."
                    )
        return "\n".join(lines)
