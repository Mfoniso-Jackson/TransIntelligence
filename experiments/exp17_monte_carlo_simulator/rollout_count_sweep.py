"""Follow-up to experiment 17 -- where is the point at which
`MonteCarloSimulator`'s verdict becomes unreliable, if it exists at all
for this effect size? Not a new numbered experiment -- fills a gap
experiment 17's own RESULTS.md flagged explicitly: "`n_rollouts`
sensitivity was checked at only two values (5 and 200) — the point at
which the verdict becomes unreliable... was not located."

Reuses experiment 17's own `train_agent`/`compare_at_rollout_count`
unchanged -- agents are trained ONCE (the expensive part), then compared
at every `n_rollouts` value in `N_ROLLOUTS_SWEEP` against the SAME
trained policies and starting states, so results across the sweep are
directly comparable and not confounded by different training runs.

Run: PYTHONPATH=. python experiments/exp17_monte_carlo_simulator/rollout_count_sweep.py
"""
from __future__ import annotations

from experiments.exp16_regime_shift_multistep_planning.run import BEAM_WIDTH, LOOKAHEAD
from experiments.exp17_monte_carlo_simulator.run import (
    HORIZON, INIT_RANGE, LAG_WEIGHT, N_START_STATES, NOISE_SIGMA, POST_SHIFT_SCALE,
    REGIME_SHIFT_STEP, SEEDS, TARGET,
    DelayedRegimeShiftControlEnv, compare_at_rollout_count, train_agent,
)
import random

N_ROLLOUTS_SWEEP = [1, 2, 3, 5, 10, 20, 50, 100, 200]


def build_trained():
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
    return trained


def main() -> None:
    trained = build_trained()
    print(f"{'n_rollouts':>10} {'wins_for_greedy':>16} {'total':>6} {'win_rate':>9}")
    for n_rollouts in N_ROLLOUTS_SWEEP:
        wins, total, _, _ = compare_at_rollout_count(n_rollouts, trained, verbose=False)
        print(f"{n_rollouts:>10} {wins:>16} {total:>6} {100 * wins / total:>8.1f}%")


if __name__ == "__main__":
    main()
