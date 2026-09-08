"""Regression tests for experiment 19's detection-diagnosis follow-up
(docs/research-agenda.md #7p,
experiments/exp19_nonlinear_regime_shift/RESULTS.md "Follow-up" section).
"""
import statistics

from experiments.exp19_nonlinear_regime_shift.detection_diagnosis import (
    pre_shift_residuals, post_shift_residual_stdev,
)

SEEDS = range(6)


def test_pre_shift_residual_noise_is_statistically_matched():
    """Reproduces RESULTS.md's original refutation of the "noisier
    residuals" hypothesis: steady-state pre-shift residual stdev should
    be nearly identical between the linear and nonlinear models."""
    nl_stdevs = [statistics.pstdev(pre_shift_residuals(s, nonlinear=True)) for s in SEEDS]
    li_stdevs = [statistics.pstdev(pre_shift_residuals(s, nonlinear=False)) for s in SEEDS]
    assert abs(statistics.mean(nl_stdevs) - statistics.mean(li_stdevs)) < 0.05


def test_post_shift_residuals_are_more_variable_for_the_nonlinear_model():
    """The follow-up's actual new finding: even though pre-shift
    behavior is statistically matched, post-shift (before detection),
    the nonlinear model's own residual stream is noisier on average than
    the linear model's -- a real, measured difference that plausibly
    explains slower/less-reliable CUSUM detection, even without a full
    mechanistic account of why it happens."""
    nl_post = [s for s in (post_shift_residual_stdev(seed, True) for seed in SEEDS) if s is not None]
    li_post = [s for s in (post_shift_residual_stdev(seed, False) for seed in SEEDS) if s is not None]
    assert statistics.mean(nl_post) > statistics.mean(li_post)
