"""Regression test for Experiment 11's quadratic-baseline follow-up
(docs/research-agenda.md #7h,
experiments/exp11_world_model_planning/RESULTS.md "Follow-up" section).
"""
import statistics

from experiments.exp11_world_model_planning.quadratic_baseline import ModelFreeQuadraticQAgent
from experiments.exp11_world_model_planning.run import CONDITIONS, run_condition


def _mean_regret(agent_factory, seeds: range) -> float:
    rewards, ceilings = [], []
    for seed in seeds:
        r, c = run_condition(agent_factory, seed)
        rewards.append(r)
        ceilings.append(c)
    return statistics.mean(ceilings) - statistics.mean(rewards)


def test_quadratic_model_free_baseline_closes_the_gap_with_the_world_model():
    """The core prediction: a function class that CAN represent the true
    (quadratic) reward surface should close the gap experiment 11 found
    for a linear one -- confirming the original explanation (linear
    cannot represent a peak) was the real cause, not an unaccounted-for
    confound between the two conditions."""
    quadratic_regret = _mean_regret(lambda env: ModelFreeQuadraticQAgent(), range(8))
    world_model_regret = _mean_regret(CONDITIONS["world_model"], range(8))
    assert abs(quadratic_regret - world_model_regret) < 0.2


def test_quadratic_model_free_baseline_beats_the_linear_one_decisively():
    quadratic_regret = _mean_regret(lambda env: ModelFreeQuadraticQAgent(), range(8))
    linear_regret = _mean_regret(CONDITIONS["model_free_linear_q"], range(8))
    assert linear_regret > quadratic_regret + 1.0
