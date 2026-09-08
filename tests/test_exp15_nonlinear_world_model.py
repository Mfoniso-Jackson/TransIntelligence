"""Regression tests for Experiment 15 (docs/research-agenda.md #7l,
experiments/exp15_nonlinear_world_model/RESULTS.md). Small-scale
versions of the real experiment (fewer seeds, for test-suite speed).
"""
import statistics

from experiments.exp15_nonlinear_world_model.run import CONDITIONS, run_condition


def test_nonlinear_world_model_matches_the_oracle_ceiling():
    seeds = range(8)
    rewards, ceilings = [], []
    for seed in seeds:
        r, c = run_condition(CONDITIONS["nonlinear_world_model"], seed)
        rewards.append(r)
        ceilings.append(c)
    regret = statistics.mean(ceilings) - statistics.mean(rewards)
    assert abs(regret) < 0.05


def test_linear_world_model_is_substantially_worse_than_nonlinear_under_true_nonlinearity():
    """The core claim: reusing LinearDynamicsModel unchanged under a
    genuinely nonlinear environment should show a real, large regret
    gap relative to the correctly-specified nonlinear model -- both
    using the identical ordinary_least_squares tool and identical state
    access, differing only in which features are fit."""
    seeds = range(8)

    def mean_regret(label: str) -> float:
        rewards, ceilings = [], []
        for seed in seeds:
            r, c = run_condition(CONDITIONS[label], seed)
            rewards.append(r)
            ceilings.append(c)
        return statistics.mean(ceilings) - statistics.mean(rewards)

    linear_regret = mean_regret("linear_world_model")
    nonlinear_regret = mean_regret("nonlinear_world_model")
    assert linear_regret > nonlinear_regret * 10


def test_linear_world_model_still_beats_the_state_blind_floor():
    """The linear model is misspecified but not useless -- it still
    captures the dominant linear trend, so it should beat ignoring state
    entirely, the same pattern experiment 11 found for its model-free
    linear baseline."""
    seeds = range(8)
    linear_rewards = [run_condition(CONDITIONS["linear_world_model"], s)[0] for s in seeds]
    blind_rewards = [run_condition(CONDITIONS["state_blind"], s)[0] for s in seeds]
    assert statistics.mean(linear_rewards) > statistics.mean(blind_rewards)
