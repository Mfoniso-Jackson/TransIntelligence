"""Regression tests for Experiment 12 (docs/research-agenda.md #7i,
experiments/exp12_multistep_planning/RESULTS.md). Small-scale versions
of the real experiment (fewer episodes/seeds, for test-suite speed).
"""
import statistics

from experiments.exp12_multistep_planning.run import LOOKAHEAD, run_condition


def test_mpc_beats_greedy_with_learned_dynamics():
    seeds = range(6)
    greedy = statistics.mean(run_condition(s, 1, False) for s in seeds)
    mpc = statistics.mean(run_condition(s, LOOKAHEAD, False) for s in seeds)
    assert mpc > greedy


def test_mpc_beats_greedy_with_oracle_dynamics_too():
    """Same qualitative gap with the true dynamics -- confirms the
    advantage is about planning horizon, not about the learned model
    happening to favor one condition."""
    seeds = range(6)
    greedy = statistics.mean(run_condition(s, 1, True) for s in seeds)
    mpc = statistics.mean(run_condition(s, LOOKAHEAD, True) for s in seeds)
    assert mpc > greedy


def test_learned_and_oracle_dynamics_perform_similarly():
    """The learned dynamics model should cost little relative to knowing
    the true dynamics exactly -- consistent with LinearDynamicsModel-style
    models recovering simple linear dynamics almost exactly given enough
    data (tests/test_world_models.py)."""
    seeds = range(6)
    greedy_learned = statistics.mean(run_condition(s, 1, False) for s in seeds)
    greedy_oracle = statistics.mean(run_condition(s, 1, True) for s in seeds)
    assert abs(greedy_learned - greedy_oracle) < 0.01


def test_capping_lookahead_at_remaining_steps_makes_mpc_reduce_to_greedy_at_the_last_step():
    from experiments.exp12_multistep_planning.run import ACTIONS, NUDGES, choose_action
    import random

    def true_predict(p, pend, a):
        return p + 0.5 * NUDGES[a] + 0.5 * pend

    rng = random.Random(0)
    greedy = choose_action(0.0, 2.0, true_predict, set(ACTIONS), lookahead=1, remaining_steps=1, rng=rng)
    mpc_capped = choose_action(0.0, 2.0, true_predict, set(ACTIONS), lookahead=LOOKAHEAD, remaining_steps=1, rng=rng)
    assert greedy == mpc_capped
