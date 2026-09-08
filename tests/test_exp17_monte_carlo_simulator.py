"""Regression tests for Experiment 17 (docs/research-agenda.md #7n,
experiments/exp17_monte_carlo_simulator/RESULTS.md).
"""
import random

from experiments.exp17_monte_carlo_simulator.run import (
    HORIZON, INIT_RANGE, LOOKAHEAD, BEAM_WIDTH,
    compare_at_rollout_count, make_transition_fn, policy_from_agent, score_fn, train_agent,
    DelayedRegimeShiftControlEnv, TARGET, LAG_WEIGHT, NOISE_SIGMA, REGIME_SHIFT_STEP, POST_SHIFT_SCALE,
)
from transintelligence.simulation import MonteCarloSimulator

SEEDS = range(3)


def _train_for_seeds(seeds) -> list:
    trained = []
    for seed in seeds:
        greedy_agent = train_agent(lookahead=1, beam_width=None, seed=seed)
        mpc_agent = train_agent(lookahead=LOOKAHEAD, beam_width=BEAM_WIDTH, seed=seed)
        env = DelayedRegimeShiftControlEnv(target=TARGET, lag_weight=LAG_WEIGHT, horizon=HORIZON,
                                            init_range=INIT_RANGE, noise_sigma=NOISE_SIGMA,
                                            regime_shift_step=REGIME_SHIFT_STEP, post_shift_scale=POST_SHIFT_SCALE, seed=seed)
        start_rng = random.Random(seed + 90_000)
        start_states = [(start_rng.uniform(*INIT_RANGE), 0.0) for _ in range(3)]
        trained.append((seed, greedy_agent, mpc_agent, env, start_states))
    return trained


def test_monte_carlo_comparison_recovers_experiment_16s_ranking():
    """The core validation: MonteCarloSimulator, using nothing but
    stochastic rollout comparison of the two already-trained, frozen
    policies, should find greedy_cusum_adapts beats mpc_beam_cusum_adapts
    on most start states -- the same ranking experiment 16 established
    via full multi-seed environment rollouts averaged over 400 episodes."""
    trained = _train_for_seeds(SEEDS)
    wins, total, greedy_means, mpc_means = compare_at_rollout_count(200, trained, verbose=False)
    assert wins / total >= 0.8
    assert sum(greedy_means) / len(greedy_means) > sum(mpc_means) / len(mpc_means)


def test_more_rollouts_does_not_make_the_verdict_worse():
    """Sensitivity control: a far smaller n_rollouts should not produce a
    HIGHER win rate for greedy than the full n_rollouts -- if it did,
    that would suggest N_ROLLOUTS=200's margin is an artifact of sample
    size rather than a genuine, stable effect."""
    trained = _train_for_seeds(SEEDS)
    small_wins, small_total, _, _ = compare_at_rollout_count(5, trained, verbose=False)
    large_wins, large_total, _, _ = compare_at_rollout_count(200, trained, verbose=False)
    assert large_wins / large_total >= small_wins / small_total - 0.2


def test_swapping_policy_argument_order_flips_the_reported_winner():
    """Re-verifies order-invariance (already checked at the kernel level
    in test_simulation.py) with this experiment's own real trained
    agents and real post-shift dynamics -- rules out a bug specific to
    this particular combination of closures."""
    trained = _train_for_seeds(range(1))
    _, greedy_agent, mpc_agent, env, start_states = trained[0]
    simulator = MonteCarloSimulator(n_rollouts=200)
    transition_fn = make_transition_fn(env)
    rng_forward = random.Random(0)
    rng_reversed = random.Random(0)
    forward = simulator.compare_policies(start_states[0], policy_from_agent(greedy_agent), policy_from_agent(mpc_agent),
                                          transition_fn, score_fn, horizon=HORIZON, rng=rng_forward)
    reversed_ = simulator.compare_policies(start_states[0], policy_from_agent(mpc_agent), policy_from_agent(greedy_agent),
                                            transition_fn, score_fn, horizon=HORIZON, rng=rng_reversed)
    assert forward["winner"] == "a"
    assert reversed_["winner"] == "b"
