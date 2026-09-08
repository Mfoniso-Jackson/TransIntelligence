"""Verification output models for evidence-aware reasoning, plus
model-calibration verification (`CalibrationVerifier`, below) --
fills the `Verifier` protocol stub in
`transintelligence/reasoning/interfaces.py`, the last reasoning-protocol
stub from before Phase 4 still unbuilt (`Predictor`, `Planner`, and
`Simulator` were filled in Phase 6).

Unlike `Predictor`/`Planner`/`Simulator`, `Verifier` isn't a first-order
reasoning mechanism that produces an answer -- per the master context's
own architecture (`docs/master-context.md` §24), "Verification" sits
under META-INTELLIGENCE, downstream of REASONING and WORLD MODEL: it
checks the reliability of what those layers already produced.
`EvidenceVerifier` (below) already covers one kind of check -- is a
`Claim` backed by `Evidence`. `CalibrationVerifier` covers a different
one: does a predictive model's own claimed uncertainty match how often
its predictions are actually right -- the operational form of Dawid's
definition of calibration (*The Well-Calibrated Bayesian*, Journal of
the American Statistical Association 77(379), 605-613, 1982): a
well-calibrated forecaster's stated confidence level should equal the
long-run frequency with which outcomes actually fall within it.
"""
from __future__ import annotations
import math
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


class CalibrationStatus(str, Enum):
    WELL_CALIBRATED = "well_calibrated"
    MISCALIBRATED = "miscalibrated"
    INSUFFICIENT_DATA = "insufficient_data"

@dataclass(frozen=True)
class CalibrationResult:
    status: CalibrationStatus
    claimed_coverage: float
    observed_coverage: float | None
    n_observations: int
    n_hits: int | None
    lr_statistic: float | None
    p_value: float | None
    issues: tuple[str, ...] = ()

def _log_or_zero(count: int, prob: float) -> float:
    """count*ln(prob), using the 0*ln(0) := 0 convention for a zero
    count (the standard convention for a likelihood-ratio statistic's
    boundary cases, e.g. all-hit or all-miss samples) -- `prob` is
    clamped away from the domain boundary so a genuinely 0-probability
    event with count>0 contributes a large, finite penalty instead of
    raising a domain error."""
    if count == 0:
        return 0.0
    prob = min(max(prob, 1e-12), 1 - 1e-12)
    return count * math.log(prob)

def _unconditional_coverage_lr(n: int, n_hits: int, claimed_coverage: float) -> float:
    """Kupiec's (1995) unconditional-coverage / "proportion of failures"
    likelihood-ratio statistic: -2*ln(L(claimed_coverage) / L(observed
    hit rate)), comparing the model's claimed hit probability against
    the empirical maximum-likelihood estimate x/n. Always >= 0 (the MLE
    can never have lower likelihood than any other fixed probability),
    so its square root below is always defined."""
    observed_coverage = n_hits / n
    log_likelihood_claimed = (_log_or_zero(n - n_hits, 1 - claimed_coverage)
                               + _log_or_zero(n_hits, claimed_coverage))
    log_likelihood_observed = (_log_or_zero(n - n_hits, 1 - observed_coverage)
                                + _log_or_zero(n_hits, observed_coverage))
    return -2.0 * (log_likelihood_claimed - log_likelihood_observed)

@dataclass(frozen=True)
class CalibrationVerifier:
    """Checks whether a predictive model's claimed uncertainty
    (`claimed_sigma`, assumed Gaussian) matches how often its own
    residuals actually fall within `threshold_sigmas` standard
    deviations -- Kupiec's (1995) unconditional-coverage test, the
    likelihood-ratio "proportion of failures" backtest originally built
    to check whether a risk model's stated confidence intervals are
    honest. The claimed coverage probability at a given `threshold_sigmas`
    is `erf(threshold_sigmas / sqrt(2))` (the fraction of a standard
    normal distribution within that many standard deviations of zero);
    the LR statistic is asymptotically chi-squared with 1 degree of
    freedom, whose CDF has the closed form `erf(sqrt(x/2))` (chi-squared(1)
    is the square of a standard normal) -- computable from `math.erf`
    alone, no numerical library beyond the standard `math` module needed,
    matching this repo's lightweight-dependencies constraint."""

    threshold_sigmas: float = 1.0
    alpha: float = 0.05
    min_observations: int = 10

    def verify(self, residuals: list[float], claimed_sigma: float) -> CalibrationResult:
        if claimed_sigma <= 0:
            raise ValueError("claimed_sigma must be positive")
        claimed_coverage = math.erf(self.threshold_sigmas / math.sqrt(2))
        n = len(residuals)
        if n < self.min_observations:
            return CalibrationResult(CalibrationStatus.INSUFFICIENT_DATA, claimed_coverage, None, n, None, None, None,
                                      issues=(f"fewer than {self.min_observations} observations ({n} given)",))
        n_hits = sum(1 for r in residuals if abs(r) <= self.threshold_sigmas * claimed_sigma)
        observed_coverage = n_hits / n
        lr = _unconditional_coverage_lr(n, n_hits, claimed_coverage)
        p_value = 1.0 - math.erf(math.sqrt(lr / 2.0))
        if p_value >= self.alpha:
            return CalibrationResult(CalibrationStatus.WELL_CALIBRATED, claimed_coverage, observed_coverage,
                                      n, n_hits, lr, p_value)
        issue = f"observed coverage {observed_coverage:.3f} vs. claimed {claimed_coverage:.3f} (p={p_value:.4f})"
        return CalibrationResult(CalibrationStatus.MISCALIBRATED, claimed_coverage, observed_coverage,
                                  n, n_hits, lr, p_value, issues=(issue,))
