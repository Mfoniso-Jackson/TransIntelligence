"""Experiment 4, swept across observation noise -- mirrors experiments 1
and 2's noise sweeps rather than reporting a single fixed configuration
(docs/research-agenda.md #7a).

Critical point this sweep tests: the trigger's null accuracy
(`min_accuracy=0.85`) was calibrated specifically for noise_sigma=0.05,
using experiment 1's observed steady-state accuracy at that noise level
(0.906) as the reference. It is kept FIXED across this sweep on purpose --
the question is whether a trigger calibrated for one noise level stays
well-behaved (specific and sensitive) outside it, not whether some
per-noise-level recalibration can be found that works everywhere.

Run: PYTHONPATH=. python experiments/exp04_frame_discovery/sweep_noise.py
"""
from __future__ import annotations

import statistics

from environments.transworld import FrameSwitchEnv
from experiments.exp01_frame_conditioning.held_out_frame import ALL_FRAMES, HELD_OUT_INDEX, TRAIN_FRAMES
from experiments.exp04_frame_discovery.agents import DiscoveringRFAgent, RandomDiscoveryAgent

SWITCH_PERIOD = 40
JITTER = 10
N_STEPS = 3000
SEEDS = list(range(10))
WINDOW = 40
MIN_ACCURACY = 0.85  # fixed across the sweep, deliberately not recalibrated per noise level
ALPHA = 0.01
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]  # same levels as experiments/exp02_frame_dependence/run.py


def run_seed(agent_cls, seed: int, noise_sigma: float) -> dict:
    env = FrameSwitchEnv(ALL_FRAMES, switch_period=SWITCH_PERIOD, jitter=JITTER, noise_sigma=noise_sigma, seed=seed)
    agent = agent_cls(TRAIN_FRAMES, window=WINDOW, min_accuracy=MIN_ACCURACY, alpha=ALPHA, seed=seed)

    before, after = [], []
    first_step = None
    for step in range(N_STEPS):
        info = env.observe()
        is_held_out = info.active_frame_index == HELD_OUT_INDEX
        reward = env.feedback(agent.act(info.raw))
        agent.update(reward, true_is_held_out=is_held_out)
        if first_step is None:
            first_step = agent.first_acted_discovery_step
        if is_held_out:
            (after if first_step is not None and step >= first_step else before).append(reward)

    false_triggers = [t for t in agent.trigger_log if t["held_out_fraction"] < 0.5]
    return {
        "discovered": first_step is not None,
        "acc_before": sum(before) / len(before) if before else float("nan"),
        "acc_after": sum(after) / len(after) if after else float("nan"),
        "n_triggers": len(agent.trigger_log),
        "n_false_triggers": len(false_triggers),
    }


def main() -> None:
    print(f"{'noise':<7} {'agent':<18} {'disc_rate':>10} {'acc_before':>11} {'acc_after':>11} {'n_triggers':>11} {'false_rate':>11}")
    for sigma in NOISE_LEVELS:
        for label, agent_cls in (("discovering_rf", DiscoveringRFAgent), ("random_discovery", RandomDiscoveryAgent)):
            rows = [run_seed(agent_cls, seed, sigma) for seed in SEEDS]
            disc_rate = sum(r["discovered"] for r in rows) / len(rows)
            before_vals = [r["acc_before"] for r in rows if r["acc_before"] == r["acc_before"]]
            after_vals = [r["acc_after"] for r in rows if r["acc_after"] == r["acc_after"]]
            total_triggers = sum(r["n_triggers"] for r in rows)
            total_false = sum(r["n_false_triggers"] for r in rows)
            false_rate = total_false / total_triggers if total_triggers else float("nan")
            print(f"{sigma:<7} {label:<18} {disc_rate:>10.2f} "
                  f"{statistics.mean(before_vals) if before_vals else float('nan'):>11.3f} "
                  f"{statistics.mean(after_vals) if after_vals else float('nan'):>11.3f} "
                  f"{total_triggers:>11} {false_rate:>11.3f}")


if __name__ == "__main__":
    main()
