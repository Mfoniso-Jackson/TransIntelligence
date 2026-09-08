"""Follow-up to Experiment 13: RESULTS.md flagged a real, unexplained-
beyond-diagnosis limitation -- a 20% false-positive detection rate,
notably higher than experiment 5's ~6% baseline, traced to monitoring a
periodically-refit model's own residuals rather than a stationary raw
signal. This checks the most direct, mechanistically-motivated lever:
does refitting the dynamics model MORE often (smaller `refit_interval`)
reduce the false-positive rate, by keeping the model's residuals closer
to the true dynamics continuously instead of letting small-sample
staleness accumulate between infrequent refits?
(docs/research-agenda.md #7j)

Two things are measured, not just one, because a naive "just refit more
often" fix could easily trade one problem for another:

1. **False-positive rate** under a purely stationary environment (the
   regime shift is set beyond the trial horizon, so it never happens) --
   the direct measurement of the limitation experiment 13 found.
2. **True-detection rate and latency** under the severe (`sign_flip`)
   shift from experiment 13 -- refitting more often changes the
   residual stream's statistical character everywhere, not just during
   stationary periods, so a fix that reduces false positives but also
   cripples true detection would not actually be a fix.

Run: PYTHONPATH=. python experiments/exp13_regime_shift_world_model/refit_interval_calibration.py
"""
from __future__ import annotations

import random
import statistics

from environments.transworld import RegimeShiftControlEnv

from experiments.exp13_regime_shift_world_model.run import (
    ACTIONS,
    CUSUMAdaptiveAgent,
    N_TRIALS,
    NOISE_SIGMA,
    REGIME_SHIFT_TRIAL,
    SEEDS,
    STATE_RANGE,
    TARGET,
    WARMUP_TRIALS,
)

REFIT_INTERVALS_TO_TEST = [10, 20]  # 20 is experiment 13's original


def run_agent(seed: int, refit_interval: int, regime_shift_trial: int, post_shift_scale: float) -> list[int]:
    env = RegimeShiftControlEnv(target=TARGET, state_range=STATE_RANGE, noise_sigma=NOISE_SIGMA,
                                 regime_shift_trial=regime_shift_trial, post_shift_scale=post_shift_scale, seed=seed)
    rng = random.Random(seed + 30_000)
    agent = CUSUMAdaptiveAgent(refit_interval=refit_interval)
    for trial in range(N_TRIALS):
        info = env.observe()
        if trial < WARMUP_TRIALS:
            action = rng.choice(ACTIONS)
        else:
            action = agent.choose_action(info.state, rng)
        info = env.step(action)
        agent.observe(trial, info.state, action, info.next_state)
    return agent.reset_trials


def main() -> None:
    print(f"{'refit_interval':>14} {'stationary_false_positive_rate':>31} {'true_detection_rate':>20} {'mean_detection_latency':>23}")
    for refit_interval in REFIT_INTERVALS_TO_TEST:
        # 1. False-positive rate: environment never shifts.
        fp_seeds = 0
        for seed in SEEDS:
            resets = run_agent(seed, refit_interval, regime_shift_trial=N_TRIALS + 1, post_shift_scale=1.0)
            if resets:
                fp_seeds += 1

        # 2. True-detection rate and latency: the severe sign-flip shift from experiment 13.
        detected, latencies = 0, []
        for seed in SEEDS:
            resets = run_agent(seed, refit_interval, regime_shift_trial=REGIME_SHIFT_TRIAL, post_shift_scale=-1.0)
            post_shift_resets = [t for t in resets if t >= REGIME_SHIFT_TRIAL]
            if post_shift_resets:
                detected += 1
                latencies.append(post_shift_resets[0] - REGIME_SHIFT_TRIAL)

        mean_latency = statistics.mean(latencies) if latencies else float("nan")
        print(f"{refit_interval:>14} {f'{fp_seeds}/{len(SEEDS)}':>31} {f'{detected}/{len(SEEDS)}':>20} {mean_latency:>23.1f}")


if __name__ == "__main__":
    main()
