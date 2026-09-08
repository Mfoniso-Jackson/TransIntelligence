"""Experiment 17 -- does `MonteCarloSimulator`, using nothing but
stochastic rollout comparison, independently recover a finding this
program already established by direct multi-seed experimentation?
(docs/research-agenda.md #7n, Phase 6, closing)

This is the validating experiment for `transintelligence/simulation/`'s
`MonteCarloSimulator` -- the last of Phase 6's three remaining gaps
(nonlinear dynamics: experiment 15; regime-adaptation + multi-step
planning: experiment 16; `Simulator`: here). Rather than building a new
environment from scratch, this reuses experiment 16's own trained agents
and environment (`Agent`, `DelayedRegimeShiftControlEnv`, all constants,
imported directly -- not reimplemented, so there's no room for a subtly
different setup to produce an artificially clean match) and asks a
narrower question with a genuinely different methodology: experiment 16
found, via full multi-seed environment rollouts averaged over 400
episodes each, that `greedy_cusum_adapts` beats `mpc_beam_cusum_adapts`
post-shift (-4.52 vs. -33.06). Does `MonteCarloSimulator.compare_policies`
-- given only the two agents' already-trained, FROZEN policies (learning
switched off; `agent.observe()` is never called again after training)
and the environment's true post-shift dynamics formula plus its real
observation noise -- recover the same ranking from short, single-state
Monte Carlo rollouts alone?

This is a genuine, pre-registered check: the answer is already known
from experiment 16, so there's no room to quietly pick a favorable case,
and a mismatch would be a real, reportable finding (either a bug in
`MonteCarloSimulator`, or evidence that experiment 16's full-environment
result doesn't hold from arbitrary starting states).

Run: PYTHONPATH=. python experiments/exp17_monte_carlo_simulator/run.py
"""
from __future__ import annotations

import random
import statistics

from environments.transworld import NUDGES
from experiments.exp16_regime_shift_multistep_planning.run import (
    ACTIONS, BEAM_WIDTH, HORIZON, INIT_RANGE, LAG_WEIGHT, LOOKAHEAD, N_EPISODES,
    NOISE_SIGMA, POST_SHIFT_SCALE, REGIME_SHIFT_STEP, TARGET, WARMUP_EPISODES,
    Agent, DelayedRegimeShiftControlEnv, run_episode,
)
from transintelligence.simulation import MonteCarloSimulator

SEEDS = list(range(12))  # same seeds experiment 16 used, for direct comparability
N_ROLLOUTS = 200
N_START_STATES = 5
# Any global_step strictly greater than REGIME_SHIFT_STEP gives the same
# scale factor (env._scale_at is a step function of the threshold alone),
# so a single fixed post-shift step is exact, not an approximation.
POST_SHIFT_STEP = REGIME_SHIFT_STEP + 1


def train_agent(lookahead: int, beam_width: int | None, seed: int) -> Agent:
    """Reproduces exactly the training loop experiment 16's `run_condition`
    used for `greedy_cusum_adapts` (lookahead=1) and `mpc_beam_cusum_adapts`
    (lookahead=LOOKAHEAD, beam_width=BEAM_WIDTH) -- same env, same seed,
    same episode count -- but returns the trained `Agent` itself instead
    of only its aggregate reward, since this experiment needs the agent's
    own `choose_action` as a policy, not a summary statistic."""
    env = DelayedRegimeShiftControlEnv(target=TARGET, lag_weight=LAG_WEIGHT, horizon=HORIZON,
                                        init_range=INIT_RANGE, noise_sigma=NOISE_SIGMA,
                                        regime_shift_step=REGIME_SHIFT_STEP, post_shift_scale=POST_SHIFT_SCALE, seed=seed)
    rng = random.Random(seed + 50_000)
    agent = Agent(lookahead=lookahead, adapts=True, beam_width=beam_width)
    for episode_idx in range(N_EPISODES):
        run_episode(env, agent, episode_idx, rng)
    return agent


def make_transition_fn(env: DelayedRegimeShiftControlEnv):
    """The environment's real post-shift dynamics formula
    (`true_next_position`) plus its real observation noise
    (`NOISE_SIGMA`) -- a stochastic transition_fn, matching
    `env.step()`'s own formula exactly, evaluated at a fixed post-shift
    global_step rather than by mutating a real environment's internal
    counters."""
    def transition_fn(state: tuple[float, float], action: str, rng: random.Random) -> tuple[float, float]:
        position, pending = state
        next_position = (env.true_next_position(position, pending, action, POST_SHIFT_STEP)
                          + rng.gauss(0.0, NOISE_SIGMA))
        next_pending = POST_SHIFT_SCALE * NUDGES[action]
        return next_position, next_pending
    return transition_fn


def score_fn(state: tuple[float, float]) -> float:
    position, _ = state
    return -((position - TARGET) ** 2)


def policy_from_agent(agent: Agent):
    return lambda state, remaining, rng: agent.choose_action(state[0], state[1], remaining, rng)


SMALL_N_ROLLOUTS = 5  # a sensitivity control: does the verdict get noisier with far fewer samples?


def compare_at_rollout_count(n_rollouts: int, trained: list[tuple[Agent, Agent, DelayedRegimeShiftControlEnv, list[tuple[float, float]]]],
                              verbose: bool) -> tuple[int, int, list[float], list[float]]:
    """Runs compare_policies for every (seed, start_state) pair at a
    fixed n_rollouts, reusing the SAME trained agents and start states
    passed in via `trained` -- so results at different n_rollouts are
    directly comparable, not confounded by different training runs."""
    simulator = MonteCarloSimulator(n_rollouts=n_rollouts)
    wins_for_greedy = 0
    total_comparisons = 0
    greedy_means, mpc_means = [], []
    if verbose:
        print(f"{'seed':>4} {'start_position':>14} {'greedy_mean':>12} {'mpc_mean':>12} {'winner':>8}")
    for seed, greedy_agent, mpc_agent, env, start_states in trained:
        transition_fn = make_transition_fn(env)
        rollout_rng = random.Random(seed + 90_500 + n_rollouts)
        for start_state in start_states:
            result = simulator.compare_policies(start_state, policy_from_agent(greedy_agent), policy_from_agent(mpc_agent),
                                                 transition_fn, score_fn, horizon=HORIZON, rng=rollout_rng)
            total_comparisons += 1
            wins_for_greedy += result["winner"] == "a"
            greedy_means.append(result["mean_a"])
            mpc_means.append(result["mean_b"])
            if verbose:
                print(f"{seed:>4} {start_state[0]:>14.4f} {result['mean_a']:>12.4f} {result['mean_b']:>12.4f} "
                      f"{'greedy' if result['winner'] == 'a' else 'mpc':>8}")
    return wins_for_greedy, total_comparisons, greedy_means, mpc_means


def main() -> None:
    trained = []
    for seed in SEEDS:
        greedy_agent = train_agent(lookahead=1, beam_width=None, seed=seed)
        mpc_agent = train_agent(lookahead=LOOKAHEAD, beam_width=BEAM_WIDTH, seed=seed)
        env = DelayedRegimeShiftControlEnv(target=TARGET, lag_weight=LAG_WEIGHT, horizon=HORIZON,
                                            init_range=INIT_RANGE, noise_sigma=NOISE_SIGMA,
                                            regime_shift_step=REGIME_SHIFT_STEP, post_shift_scale=POST_SHIFT_SCALE, seed=seed)
        start_rng = random.Random(seed + 90_000)
        start_states = [(start_rng.uniform(*INIT_RANGE), 0.0) for _ in range(N_START_STATES)]
        trained.append((seed, greedy_agent, mpc_agent, env, start_states))

    wins, total, greedy_means, mpc_means = compare_at_rollout_count(N_ROLLOUTS, trained, verbose=True)
    print()
    print(f"n_rollouts={N_ROLLOUTS}: greedy_cusum_adapts wins {wins}/{total} comparisons ({100 * wins / total:.1f}%)")
    print(f"mean(greedy_mean) = {statistics.mean(greedy_means):.4f}, mean(mpc_mean) = {statistics.mean(mpc_means):.4f}")

    small_wins, small_total, _, _ = compare_at_rollout_count(SMALL_N_ROLLOUTS, trained, verbose=False)
    print(f"n_rollouts={SMALL_N_ROLLOUTS}: greedy_cusum_adapts wins {small_wins}/{small_total} comparisons "
          f"({100 * small_wins / small_total:.1f}%) -- sensitivity control")


if __name__ == "__main__":
    main()
