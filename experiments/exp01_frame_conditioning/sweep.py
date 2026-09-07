"""Experiment 1, swept across environment configurations -- mirrors how
Experiment 2 swept observation noise (docs/research-agenda.md #6) rather
than reporting a single fixed configuration. Two 1-D sweeps, each holding
the other dimension at run.py's default:

1. Noise sweep: same noise levels as Experiment 2 (0.0 .. 0.4), switch
   period fixed at 40+/-10.
2. Switch-period sweep: 20, 40, 80 (jitter scaled to period/4), noise fixed
   at 0.05 (run.py's default).

Reuses run_agent_on_seed/recovery_curve/calibration from run.py so this
sweep and the single-configuration run in RESULTS.md can never silently
drift apart in how they define "recovery" or "steady state".

Run: PYTHONPATH=. python experiments/exp01_frame_conditioning/sweep.py
"""
from __future__ import annotations

import statistics

from experiments.exp01_frame_conditioning.run import AGENT_KINDS, calibration, recovery_curve, run_agent_on_seed

SEEDS = list(range(10))
N_STEPS = 3000

NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]   # same levels as experiments/exp02_frame_dependence/run.py
SWITCH_PERIODS = [20, 40, 80]


def run_grid_point(switch_period: int, jitter: int, noise_sigma: float) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    per_agent_overall: dict[str, list[float]] = {}
    for agent_kind in AGENT_KINDS:
        overall = []
        for seed in SEEDS:
            log = run_agent_on_seed(agent_kind, seed, switch_period=switch_period, jitter=jitter,
                                     noise_sigma=noise_sigma, n_steps=N_STEPS)
            overall.append(sum(log.rewards) / len(log.rewards))
        per_agent_overall[agent_kind] = overall
        results[agent_kind] = {"overall_acc": statistics.mean(overall), "stdev": statistics.pstdev(overall)}

    for a, b in (("rf_aware", "flat"), ("rf_aware", "learned_embedding")):
        diffs = [x - y for x, y in zip(per_agent_overall[a], per_agent_overall[b])]
        results[f"{a}-{b}"] = {"overall_acc": statistics.mean(diffs), "stdev": statistics.pstdev(diffs)}
    return results


def print_row(label: str, results: dict[str, dict[str, float]]) -> None:
    cells = " ".join(f"{results[k]['overall_acc']:>8.3f}" for k in
                      ("flat", "learned_embedding", "rf_aware", "flat_oracle", "true_oracle",
                       "rf_aware-flat", "rf_aware-learned_embedding"))
    print(f"{label:<18} {cells}")


def main() -> None:
    header = f"{'config':<18} {'flat':>8} {'l_embed':>8} {'rf_aware':>8} {'f_oracle':>8} {'t_oracle':>8} {'rf-flat':>8} {'rf-lemb':>8}"

    print("=== noise sweep (switch_period=40, jitter=10) ===")
    print(header)
    for sigma in NOISE_LEVELS:
        results = run_grid_point(switch_period=40, jitter=10, noise_sigma=sigma)
        print_row(f"sigma={sigma}", results)

    print()
    print("=== switch-period sweep (noise_sigma=0.05, jitter=period//4) ===")
    print(header)
    for period in SWITCH_PERIODS:
        results = run_grid_point(switch_period=period, jitter=max(1, period // 4), noise_sigma=0.05)
        print_row(f"period={period}", results)


if __name__ == "__main__":
    main()
