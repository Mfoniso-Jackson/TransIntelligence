"""Regression test for experiment 18's threshold-sweep follow-up
(docs/research-agenda.md #7o,
experiments/exp18_calibration_verifier/RESULTS.md "Follow-up" section).
"""
from experiments.exp16_regime_shift_multistep_planning.run import BEAM_WIDTH, LOOKAHEAD, NOISE_SIGMA
from experiments.exp18_calibration_verifier.run import pooled_residuals, train_instrumented_agent
from transintelligence.verification import CalibrationVerifier

SEEDS = range(4)
THRESHOLDS = (0.5, 1.0, 2.0, 3.0)


def test_greedy_vs_mpc_coverage_gap_stays_small_at_every_threshold():
    """The follow-up's core finding: the tiny calibration gap experiment
    18 found at threshold_sigmas=1.0 shouldn't be an artifact of that
    one threshold choice -- it should stay small from central (0.5
    sigma) to tail (3.0 sigma) coverage, not reveal a "fatter tails"
    difference a central-only check would miss."""
    greedy_agents = [train_instrumented_agent(lookahead=1, beam_width=None, seed=s) for s in SEEDS]
    mpc_agents = [train_instrumented_agent(lookahead=LOOKAHEAD, beam_width=BEAM_WIDTH, seed=s) for s in SEEDS]
    greedy_post = pooled_residuals(greedy_agents, post_shift=True)
    mpc_post = pooled_residuals(mpc_agents, post_shift=True)

    for threshold in THRESHOLDS:
        verifier = CalibrationVerifier(threshold_sigmas=threshold)
        greedy_result = verifier.verify(greedy_post, claimed_sigma=NOISE_SIGMA)
        mpc_result = verifier.verify(mpc_post, claimed_sigma=NOISE_SIGMA)
        gap = abs(greedy_result.observed_coverage - mpc_result.observed_coverage)
        assert gap < 0.03, f"threshold={threshold}: gap={gap} unexpectedly large"
