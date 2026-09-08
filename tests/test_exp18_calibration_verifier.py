"""Regression tests for Experiment 18 (docs/research-agenda.md #7o,
experiments/exp18_calibration_verifier/RESULTS.md).
"""
from experiments.exp18_calibration_verifier.run import (
    BEAM_WIDTH, LOOKAHEAD, NOISE_SIGMA, THRESHOLD_SIGMAS,
    InstrumentedAgent, pooled_residuals, train_instrumented_agent,
)
from transintelligence.verification import CalibrationVerifier

SEEDS = range(3)


def test_instrumented_agent_logs_residuals_on_both_sides_of_the_shift():
    """The full, untrimmed residual log must actually cover both the
    pre- and post-shift windows -- if it were empty on either side, the
    calibration comparison downstream would be vacuous."""
    agent = train_instrumented_agent(lookahead=1, beam_width=None, seed=0)
    assert isinstance(agent, InstrumentedAgent)
    pre = pooled_residuals([agent], post_shift=False)
    post = pooled_residuals([agent], post_shift=True)
    assert len(pre) > 100
    assert len(post) > 100


def test_greedy_and_mpc_show_no_consistent_calibration_gap_post_shift():
    """Experiment 18's actual finding: neither agent's one-step dynamics
    model is measurably worse-calibrated than the other's, post-shift --
    the observed coverage rates should be close, not one dramatically
    lower than the other. This is a null result and the test checks
    exactly that: the gap stays small (well under half the total
    shortfall from the claimed coverage), not that either agent is
    "well calibrated" in isolation (both show a small universal
    overconfidence present pre-shift too, see RESULTS.md)."""
    greedy_agents = [train_instrumented_agent(lookahead=1, beam_width=None, seed=s) for s in SEEDS]
    mpc_agents = [train_instrumented_agent(lookahead=LOOKAHEAD, beam_width=BEAM_WIDTH, seed=s) for s in SEEDS]
    verifier = CalibrationVerifier(threshold_sigmas=THRESHOLD_SIGMAS)
    greedy_result = verifier.verify(pooled_residuals(greedy_agents, post_shift=True), claimed_sigma=NOISE_SIGMA)
    mpc_result = verifier.verify(pooled_residuals(mpc_agents, post_shift=True), claimed_sigma=NOISE_SIGMA)
    assert abs(greedy_result.observed_coverage - mpc_result.observed_coverage) < 0.03


def test_pre_shift_shows_the_same_small_miscalibration_as_post_shift():
    """The confound control: if pre-shift (stable data, no regime
    confusion) were well-calibrated while post-shift wasn't, that would
    point to a shift-specific cause. It isn't -- both windows show
    roughly the same small overconfidence, meaning it's a baseline
    property of the fitting procedure, not a regime-shift artifact."""
    greedy_agents = [train_instrumented_agent(lookahead=1, beam_width=None, seed=s) for s in SEEDS]
    verifier = CalibrationVerifier(threshold_sigmas=THRESHOLD_SIGMAS)
    pre_result = verifier.verify(pooled_residuals(greedy_agents, post_shift=False), claimed_sigma=NOISE_SIGMA)
    post_result = verifier.verify(pooled_residuals(greedy_agents, post_shift=True), claimed_sigma=NOISE_SIGMA)
    assert abs(pre_result.observed_coverage - post_result.observed_coverage) < 0.03
