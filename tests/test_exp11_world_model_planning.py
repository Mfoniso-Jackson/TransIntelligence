"""Regression tests for Experiment 11 (docs/research-agenda.md #7h,
experiments/exp11_world_model_planning/RESULTS.md). Small-scale versions
of the real experiment (fewer trials/seeds, for test-suite speed).
"""
import statistics

from experiments.exp11_world_model_planning.run import CONDITIONS, run_condition


def _mean_regret(label: str, seeds: range) -> float:
    rewards, ceilings = [], []
    for seed in seeds:
        r, c = run_condition(CONDITIONS[label], seed)
        rewards.append(r)
        ceilings.append(c)
    return statistics.mean(ceilings) - statistics.mean(rewards)


def test_world_model_agent_matches_the_oracle_dynamics_ceiling():
    world_model_regret = _mean_regret("world_model", range(8))
    assert abs(world_model_regret) < 0.1


def test_state_blind_agent_has_the_largest_regret():
    """The floor: ignoring state entirely should be clearly worse than
    every state-aware condition."""
    state_blind_regret = _mean_regret("state_blind", range(8))
    world_model_regret = _mean_regret("world_model", range(8))
    assert state_blind_regret > world_model_regret + 1.0


def test_world_model_beats_model_free_linear_q_despite_identical_state_access_and_tooling():
    """The core claim: decomposing 'learn the (linear) dynamics, apply
    the (known, quadratic) reward formula' beats directly fitting a
    linear model of the (quadratic) reward, even though both conditions
    see the same state and use the same OLS tool -- the same
    linear-cannot-represent-curvature lesson experiment 10 established
    for effect estimation, now shown in a planning setting."""
    model_free_regret = _mean_regret("model_free_linear_q", range(8))
    world_model_regret = _mean_regret("world_model", range(8))
    assert model_free_regret > world_model_regret + 1.0


def test_model_free_linear_q_still_beats_the_state_blind_floor():
    """Even a misspecified linear value model should still beat ignoring
    state entirely -- it's worse than the world model, not worthless."""
    model_free_regret = _mean_regret("model_free_linear_q", range(8))
    state_blind_regret = _mean_regret("state_blind", range(8))
    assert state_blind_regret > model_free_regret
