"""Regression tests for experiment 14's tie-breaking follow-up
(docs/research-agenda.md #7k,
experiments/exp14_beam_search_planning/RESULTS.md "Follow-up" section).
"""
from experiments.exp14_beam_search_planning.run import (
    DEPTHS_WITH_GROUND_TRUTH, N_STATES, SEED, evaluate, sample_states,
)
from experiments.exp14_beam_search_planning.tie_breaking import evaluate_min_effort


def test_min_effort_tiebreak_improves_convergence_at_depth_3_but_does_not_match_beam_search():
    """The follow-up's core finding: minimal-effort tie-breaking is a
    real, partial fix (more than doubling the convergence count at the
    depth where the oscillation pathology is worst) but does not close
    the gap to beam search -- confirming beam search's robustness
    advantage is a separate mechanism, not just a smarter tie-break in
    disguise."""
    states = sample_states(N_STATES, SEED)
    depth = 3
    assert depth in DEPTHS_WITH_GROUND_TRUTH
    _, _, _, original_converged = evaluate(depth, None, states)
    _, _, _, min_effort_converged = evaluate_min_effort(depth, states)
    _, _, _, beam_converged = evaluate(depth, 2, states)
    assert min_effort_converged > original_converged
    assert min_effort_converged < beam_converged


def test_min_effort_tiebreak_makes_no_difference_at_depth_2():
    """At depth 2 there's essentially nothing to break ties on
    differently -- exhaustive search's original result should be
    unchanged by the min-effort rule."""
    states = sample_states(N_STATES, SEED)
    depth = 2
    original_score, _, _, original_converged = evaluate(depth, None, states)
    min_effort_score, _, _, min_effort_converged = evaluate_min_effort(depth, states)
    assert min_effort_converged == original_converged
    assert abs(min_effort_score - original_score) < 1e-6
