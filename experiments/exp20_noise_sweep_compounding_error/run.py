"""Experiment 20 -- does experiment 16's central hypothesis (multi-step
lookahead chains two predictions from the same learned model, and the
gap comes from each prediction's estimation error, which never fully
vanishes because the environment has real observation noise) actually
scale with that noise, or is the persistent `mpc_beam_cusum_adapts` vs.
`greedy_cusum_adapts` gap structural -- present even with little or no
environment noise? (docs/research-agenda.md #7q, Phase 6, continued)

Experiment 16 stated this explicitly as its best-supported but
UNCONFIRMED explanation; experiment 18 tested a specific ALTERNATIVE
explanation (self-steered training data degrading the model) and ruled
it out, narrowing the field but never manipulating the chaining
mechanism itself. This experiment does that directly, the most literal
operationalization of the hypothesis's own wording ("... because the
environment has real observation noise, NOISE_SIGMA"): sweep
`NOISE_SIGMA` itself and check whether the post-shift reward gap between
`greedy_cusum_adapts` and `mpc_beam_cusum_adapts` shrinks toward zero as
`NOISE_SIGMA` shrinks toward zero.

Nothing new is implemented -- `Agent`, `DelayedRegimeShiftControlEnv`,
and `run_episode` are reused exactly as experiment 16 (and its
successors, 17/18) already verified, imported directly rather than
reimplemented. The only change is that `noise_sigma` becomes a swept
parameter instead of a fixed constant.

## The prediction each hypothesis makes

- **If the compounding-estimation-error hypothesis is right:** the gap
  should shrink substantially, ideally toward ~0, as `NOISE_SIGMA` -> 0
  -- with a deterministic (or near-deterministic) environment, a
  correctly-specified linear model given enough data should fit
  essentially exactly, leaving no estimation error left to compound.
- **If the gap is structural instead** (e.g. an artifact of how beam
  search's own intermediate-state scoring interacts with multi-step
  replanning, independent of model uncertainty): the gap should stay
  roughly constant across the sweep, present even near `NOISE_SIGMA=0`.

Run: PYTHONPATH=. python experiments/exp20_noise_sweep_compounding_error/run.py
"""
from __future__ import annotations

import random
import statistics

from experiments.exp16_regime_shift_multistep_planning.run import (
    BEAM_WIDTH, HORIZON, INIT_RANGE, LAG_WEIGHT, LOOKAHEAD, N_EPISODES,
    POST_SHIFT_SCALE, REGIME_SHIFT_STEP, TARGET, WARMUP_EPISODES,
    Agent, DelayedRegimeShiftControlEnv, run_episode,
)

SEEDS = list(range(20))  # 10 seeds first showed a noise_sigma=0.2 dip that didn't replicate at 20 -- see RESULTS.md
NOISE_SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]  # 0.1 is experiment 16's original value


def run_condition(agent_factory, seed: int, noise_sigma: float) -> tuple[float, float, list[int]]:
    """Returns (pre_shift_reward, post_shift_reward, reset_steps) --
    `reset_steps` (the CUSUM-detected reset points, an attribute
    `Agent` already tracks) lets the sweep check whether the reward gap
    reflects estimation-error compounding specifically, or is confounded
    by detection reliability itself varying across noise levels -- a
    real possibility experiment 19 already found detection sensitivity
    to a model's residual characteristics, so it isn't assumed away here."""
    env = DelayedRegimeShiftControlEnv(target=TARGET, lag_weight=LAG_WEIGHT, horizon=HORIZON,
                                        init_range=INIT_RANGE, noise_sigma=noise_sigma,
                                        regime_shift_step=REGIME_SHIFT_STEP, post_shift_scale=POST_SHIFT_SCALE, seed=seed)
    rng = random.Random(seed + 70_000)
    agent = agent_factory()
    pre_rewards, post_rewards = [], []
    for episode_idx in range(N_EPISODES):
        episode_start_step = env.global_step
        reward = run_episode(env, agent, episode_idx, rng)
        if episode_idx >= WARMUP_EPISODES:
            (pre_rewards if episode_start_step < REGIME_SHIFT_STEP else post_rewards).append(reward)
    return statistics.mean(pre_rewards), statistics.mean(post_rewards), agent.reset_steps


CONDITIONS = {
    "greedy_cusum_adapts": lambda: Agent(lookahead=1, adapts=True),
    "mpc_beam_cusum_adapts": lambda: Agent(lookahead=LOOKAHEAD, adapts=True, beam_width=BEAM_WIDTH),
}


def _detection_summary(reset_steps_per_seed: list[list[int]]) -> str:
    detected = sum(1 for r in reset_steps_per_seed if any(t >= REGIME_SHIFT_STEP for t in r))
    false_positives = sum(1 for r in reset_steps_per_seed if any(t < REGIME_SHIFT_STEP for t in r))
    latencies = [min(t for t in r if t >= REGIME_SHIFT_STEP) - REGIME_SHIFT_STEP
                 for r in reset_steps_per_seed if any(t >= REGIME_SHIFT_STEP for t in r)]
    latency_str = f"{statistics.mean(latencies):.1f}" if latencies else "n/a"
    return f"detected={detected}/{len(reset_steps_per_seed)} false_pos={false_positives}/{len(reset_steps_per_seed)} mean_latency={latency_str}"


def main() -> None:
    print(f"{'noise_sigma':>11} {'greedy_post':>13} {'mpc_post':>13} {'gap (greedy-mpc)':>17}")
    gaps = []
    for noise_sigma in NOISE_SIGMAS:
        greedy_posts, mpc_posts = [], []
        greedy_resets, mpc_resets = [], []
        for seed in SEEDS:
            g_pre, g_post, g_resets = run_condition(CONDITIONS["greedy_cusum_adapts"], seed, noise_sigma)
            m_pre, m_post, m_resets = run_condition(CONDITIONS["mpc_beam_cusum_adapts"], seed, noise_sigma)
            greedy_posts.append(g_post)
            mpc_posts.append(m_post)
            greedy_resets.append(g_resets)
            mpc_resets.append(m_resets)
        greedy_mean = statistics.mean(greedy_posts)
        mpc_mean = statistics.mean(mpc_posts)
        gap = greedy_mean - mpc_mean
        gaps.append((noise_sigma, gap))
        print(f"{noise_sigma:>11.2f} {greedy_mean:>13.4f} {mpc_mean:>13.4f} {gap:>17.4f}")
        print(f"    greedy detection: {_detection_summary(greedy_resets)}")
        print(f"    mpc    detection: {_detection_summary(mpc_resets)}")

    print()
    print("gap by noise_sigma:", gaps)


if __name__ == "__main__":
    main()
