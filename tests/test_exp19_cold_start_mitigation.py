"""Regression tests for experiment 19's cold-start-mitigation follow-up
(docs/research-agenda.md #7p,
experiments/exp19_nonlinear_regime_shift/RESULTS.md "Follow-up" section).
"""
import statistics

from experiments.exp19_nonlinear_regime_shift.cold_start_mitigation import (
    FullCoverageOracleAgent, run_agent,
)
from experiments.exp19_nonlinear_regime_shift.run import REGIME_SHIFT_TRIAL, CONDITIONS, OracleAdaptsAgent

SEEDS = range(5)


def test_shortening_refit_interval_makes_the_mild_shift_reversal_worse_not_better():
    """Attempt 1's negative result, reproduced: a shorter refit_interval
    should INCREASE (not close) the gap between oracle_adapts and
    never_adapts under a mild shift -- premature greedy lock-in on an
    early, unreliable estimate, not faster recovery."""
    never_post = statistics.mean(run_agent(CONDITIONS["never_adapts"], s, 0.4) for s in SEEDS)
    gap_20 = never_post - statistics.mean(
        run_agent(lambda: OracleAdaptsAgent(REGIME_SHIFT_TRIAL, refit_interval=20), s, 0.4) for s in SEEDS
    )
    gap_2 = never_post - statistics.mean(
        run_agent(lambda: OracleAdaptsAgent(REGIME_SHIFT_TRIAL, refit_interval=2), s, 0.4) for s in SEEDS
    )
    assert gap_2 > gap_20


def test_full_coverage_before_greedy_closes_the_mild_shift_reversal():
    """Attempt 2's positive result: requiring every action to be known
    before acting greedily (instead of exploiting the first one) should
    close, not just shrink, the mild-shift reversal -- oracle_adapts
    should no longer underperform never_adapts once this fix is applied."""
    never_post = statistics.mean(run_agent(CONDITIONS["never_adapts"], s, 0.4) for s in SEEDS)
    original_post = statistics.mean(
        run_agent(lambda: OracleAdaptsAgent(REGIME_SHIFT_TRIAL), s, 0.4) for s in SEEDS
    )
    fixed_post = statistics.mean(
        run_agent(lambda: FullCoverageOracleAgent(REGIME_SHIFT_TRIAL), s, 0.4) for s in SEEDS
    )
    assert original_post < never_post  # the original reversal, reproduced
    assert fixed_post >= never_post - 0.1  # the fix closes it (allowing a small margin for seed noise)
