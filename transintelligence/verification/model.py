"""Verification output models for evidence-aware reasoning."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from transintelligence.epistemic import Claim, Evidence

class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

@dataclass(frozen=True)
class VerificationResult:
    claim: Claim
    status: VerificationStatus
    confidence: float
    supporting_evidence: tuple[Evidence, ...] = ()
    issues: tuple[str, ...] = ()

class EvidenceVerifier:
    """Deterministic verifier that distinguishes unsupported assertions from evidence-backed claims."""
    def verify(self, claim: Claim) -> VerificationResult:
        if not claim.evidence:
            return VerificationResult(claim, VerificationStatus.INSUFFICIENT_EVIDENCE, 0.0, issues=("claim has no linked evidence",))
        support = min(e.confidence.value for e in claim.evidence)
        if claim.counter_evidence and max(e.confidence.value for e in claim.counter_evidence) >= support:
            return VerificationResult(claim, VerificationStatus.CONTRADICTED, support, claim.evidence, ("counter-evidence confidence meets or exceeds support",))
        return VerificationResult(claim, VerificationStatus.SUPPORTED, min(claim.confidence.value, support), claim.evidence)
