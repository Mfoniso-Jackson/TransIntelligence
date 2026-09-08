"""Tests for CalibrationVerifier (transintelligence/verification/model.py,
docs/research-agenda.md #7o) -- fills the Verifier protocol stub, the
last reasoning-protocol stub from before Phase 4 left unbuilt. Verified
against exact hand-computed likelihood-ratio values (Kupiec 1995) before
trusting it on real experiment 16 residual data.
"""
import math

import pytest

from transintelligence.verification import CalibrationStatus, CalibrationVerifier


def test_overstated_uncertainty_is_flagged_miscalibrated_exact_lr():
    """n=10 residuals all exactly 0.0 (claimed_sigma=1.0, threshold=1
    sigma -> claimed coverage ~0.6827): every single residual is a
    "hit," but the model only claimed 68.27% would be -- Kupiec's LR
    statistic for this exact (n=10, x=10, p=0.68268...) case hand-
    computed independently via the formula -2*(x*ln(p) - x*ln(x/n))
    (the n-x=0 terms vanish under the 0*ln(.):=0 convention) is
    7.6343029260425235, giving p=0.005726851401046229 -- well below the
    default alpha=0.05, correctly flagging an overly-conservative model,
    not just an overly-narrow one."""
    verifier = CalibrationVerifier(threshold_sigmas=1.0, alpha=0.05, min_observations=10)
    result = verifier.verify([0.0] * 10, claimed_sigma=1.0)
    assert result.n_observations == 10
    assert result.n_hits == 10
    assert math.isclose(result.lr_statistic, 7.6343029260425235, rel_tol=1e-9)
    assert math.isclose(result.p_value, 0.005726851401046229, rel_tol=1e-9)
    assert result.status == CalibrationStatus.MISCALIBRATED


def test_hit_rate_matching_the_claimed_coverage_is_well_calibrated():
    """n=20, 14 hits within 1 sigma (70%, close to the 68.27% claimed) --
    hand-computed LR=0.0279, p=0.867, comfortably above alpha=0.05."""
    residuals = [0.5] * 14 + [2.0] * 6  # 14 hits, 6 misses at threshold_sigmas=1.0
    verifier = CalibrationVerifier(threshold_sigmas=1.0, alpha=0.05, min_observations=10)
    result = verifier.verify(residuals, claimed_sigma=1.0)
    assert result.n_hits == 14
    assert math.isclose(result.lr_statistic, 0.027945587655608506, rel_tol=1e-9)
    assert math.isclose(result.p_value, 0.8672368094223024, rel_tol=1e-9)
    assert result.status == CalibrationStatus.WELL_CALIBRATED
    assert result.issues == ()


def test_understated_uncertainty_is_flagged_miscalibrated():
    """n=20, only 2 hits within 1 sigma (10%, far below the 68.27%
    claimed) -- a model whose real error is much larger than it claims,
    the classic risk-model-backtest failure mode Kupiec's test was built
    to catch. Hand-computed p=4.675e-8, far below alpha."""
    residuals = [0.5] * 2 + [3.0] * 18
    verifier = CalibrationVerifier(threshold_sigmas=1.0, alpha=0.05, min_observations=10)
    result = verifier.verify(residuals, claimed_sigma=1.0)
    assert result.n_hits == 2
    assert math.isclose(result.p_value, 4.6751434057412666e-08, rel_tol=1e-6)
    assert result.status == CalibrationStatus.MISCALIBRATED
    assert len(result.issues) == 1


def test_claimed_coverage_matches_the_standard_normal_cdf_at_threshold():
    """threshold_sigmas=1.0 must give the textbook ~68.27% one-sigma
    coverage, not an arbitrary or off-by-a-constant value."""
    verifier = CalibrationVerifier(threshold_sigmas=1.0)
    result = verifier.verify([0.0] * 10, claimed_sigma=1.0)
    assert math.isclose(result.claimed_coverage, 0.6826894921370859, rel_tol=1e-9)


def test_two_sigma_threshold_gives_the_textbook_95_percent_coverage():
    verifier = CalibrationVerifier(threshold_sigmas=2.0)
    result = verifier.verify([0.0] * 10, claimed_sigma=1.0)
    assert math.isclose(result.claimed_coverage, 0.9544997361036416, rel_tol=1e-9)


def test_insufficient_data_below_min_observations():
    verifier = CalibrationVerifier(min_observations=10)
    result = verifier.verify([0.0] * 5, claimed_sigma=1.0)
    assert result.status == CalibrationStatus.INSUFFICIENT_DATA
    assert result.observed_coverage is None
    assert result.p_value is None
    assert len(result.issues) == 1


def test_raises_on_non_positive_claimed_sigma():
    verifier = CalibrationVerifier()
    with pytest.raises(ValueError):
        verifier.verify([0.0] * 20, claimed_sigma=0.0)


def test_scales_with_claimed_sigma_not_just_raw_residual_magnitude():
    """The same residuals should be judged well-calibrated or
    miscalibrated purely relative to claimed_sigma -- doubling both the
    residuals and claimed_sigma together must leave the verdict
    unchanged, since only the standardized ratio matters."""
    verifier = CalibrationVerifier(threshold_sigmas=1.0)
    residuals = [0.5] * 14 + [2.0] * 6
    result_a = verifier.verify(residuals, claimed_sigma=1.0)
    result_b = verifier.verify([r * 4 for r in residuals], claimed_sigma=4.0)
    assert result_a.status == result_b.status
    assert result_a.n_hits == result_b.n_hits
    assert math.isclose(result_a.lr_statistic, result_b.lr_statistic, rel_tol=1e-9)
