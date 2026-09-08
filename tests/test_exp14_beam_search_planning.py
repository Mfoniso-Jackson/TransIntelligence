"""Regression tests for Experiment 14 (docs/research-agenda.md #7k,
experiments/exp14_beam_search_planning/RESULTS.md).
"""
from experiments.exp14_beam_search_planning.run import (
    BEAM_WIDTHS,
    N_STATES,
    evaluate,
    sample_states,
)


def test_beam_search_matches_or_beats_exhaustive_call_count_reduction():
    states = sample_states(N_STATES, seed=0)
    _, exhaustive_calls, _, _ = evaluate(depth=3, beam_width=None, states=states)
    _, beam_calls, _, _ = evaluate(depth=3, beam_width=2, states=states)
    assert beam_calls < exhaustive_calls / 10  # at least a 10x reduction at depth=3


def test_beam_search_converges_far_more_reliably_than_exhaustive_at_depth_3():
    """The core, quantified finding: exhaustive search's terminal-only
    scoring is vulnerable to a persistent non-converging oscillation
    once only the first action of each plan is executed and replanned,
    while beam search's per-step pruning is far more robust to it --
    checked directly across many starting states, not asserted from one
    traced example."""
    states = sample_states(N_STATES, seed=0)
    _, _, _, exhaustive_converged = evaluate(depth=3, beam_width=None, states=states)
    _, _, _, beam_converged = evaluate(depth=3, beam_width=2, states=states)
    assert beam_converged > exhaustive_converged + 5


def test_beam_search_converges_reliably_at_a_depth_exhaustive_search_never_attempts():
    """depth=6 is where exhaustive search is not even run (extrapolated
    cost is prohibitive) -- beam search should still reliably reach the
    target there, not just be fast."""
    states = sample_states(N_STATES, seed=0)
    for beam_width in BEAM_WIDTHS:
        _, _, _, converged = evaluate(depth=6, beam_width=beam_width, states=states)
        assert converged >= len(states) - 3  # allow a small number of non-convergent states
