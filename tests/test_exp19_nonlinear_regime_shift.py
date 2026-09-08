"""Regression tests for Experiment 19 (docs/research-agenda.md #7p,
experiments/exp19_nonlinear_regime_shift/RESULTS.md).
"""
import random
import statistics

from experiments.exp19_nonlinear_regime_shift.run import (
    ACTIONS, CONDITIONS, REGIME_SHIFT_TRIAL, OracleAdaptsAgent, run_condition,
)

SEEDS = range(4)


def test_oracle_agents_model_is_empty_for_a_stretch_right_after_the_shift():
    """The mechanism behind this experiment's central finding, checked
    directly: `_training_transitions()` starts returning only post-shift
    data the instant the shift trial is reached, but `NonlinearDynamicsModel`
    needs 3 fresh observations per action to fit at all -- one more than
    `LinearDynamicsModel`'s 2 -- and the model itself only updates on the
    next scheduled refit boundary (every REFIT_INTERVAL=20 observations).
    Starting an agent with an empty history exactly at the shift trial
    (equivalent to the moment its stale data is discarded) isolates this:
    known_actions() must stay empty for the entire first refit window,
    since zero post-shift observations exist yet to fit from."""
    agent = OracleAdaptsAgent(REGIME_SHIFT_TRIAL)
    rng = random.Random(0)
    known_actions_seen_empty = []
    for offset in range(19):  # REFIT_INTERVAL - 1: one short of the first post-shift refit
        state = rng.uniform(-5.0, 5.0)
        action = rng.choice(ACTIONS)
        agent.observe(REGIME_SHIFT_TRIAL + offset, state, action, state + 0.1)
        known_actions_seen_empty.append(agent.model.known_actions() == set())
    assert all(known_actions_seen_empty)


def test_mild_shift_oracle_adaptation_is_worse_than_never_adapting():
    """The counterintuitive headline finding: under a MILD shift, the
    cold-start cost of a full reset outweighs the (small) benefit of
    correct post-shift dynamics -- unlike experiment 13's linear-model
    version, where the two were merely indistinguishable, not reversed."""
    oracle_post = statistics.mean(run_condition(CONDITIONS["oracle_adapts"], s, post_shift_scale=0.4)[1] for s in SEEDS)
    never_post = statistics.mean(run_condition(CONDITIONS["never_adapts"], s, post_shift_scale=0.4)[1] for s in SEEDS)
    assert oracle_post < never_post


def test_severe_shift_adaptation_still_beats_never_adapting():
    """Even with the cold-start tax, a severe enough shift still makes
    adaptation worth it -- the stale model's harm outweighs the
    relearning cost, mirroring experiment 13's own sign_flip result."""
    oracle_post = statistics.mean(run_condition(CONDITIONS["oracle_adapts"], s, post_shift_scale=-1.0)[1] for s in SEEDS)
    never_post = statistics.mean(run_condition(CONDITIONS["never_adapts"], s, post_shift_scale=-1.0)[1] for s in SEEDS)
    assert oracle_post > never_post
