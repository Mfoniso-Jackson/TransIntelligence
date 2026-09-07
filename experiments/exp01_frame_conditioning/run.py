"""Experiment 1 -- does explicit frame structure beat a flat baseline under
matched (zero) information about which frame is active, and does that
advantage survive against a baseline given the true frame id directly?

See docs/research-agenda.md #5 for the hypothesis and falsification
criterion, and docs/related-work.md #2 for why the flat-vs-structured
question alone isn't enough to claim novelty (a learned-embedding condition
B is not yet implemented here -- see "Not yet built" in RESULTS.md).

Run: PYTHONPATH=. python experiments/exp01_frame_conditioning/run.py
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from transintelligence import ReferenceFrame

from environments.transworld import FrameSwitchEnv
from experiments.exp01_frame_conditioning.agents import FlatBaselineAgent, FlatOracleAgent, RFAwareAgent

FRAMES = [
    ReferenceFrame("f1", baseline=0.5, metadata={"direction": "higher_is_better"}),
    ReferenceFrame("f2", baseline=0.3, metadata={"direction": "higher_is_better"}),
    ReferenceFrame("f3", baseline=0.5, metadata={"direction": "lower_is_better"}),
    ReferenceFrame("f4", baseline=0.7, metadata={"direction": "lower_is_better"}),
]
N_STEPS = 3000
SWITCH_PERIOD = 40
JITTER = 10
NOISE_SIGMA = 0.05
SEEDS = list(range(10))
RECOVERY_WINDOW = 10   # steps immediately after a switch
STEADY_WINDOW = (30, 40)  # steps [30,40) after a switch, before the next one is likely


@dataclass
class SeedLog:
    rewards: list[int] = field(default_factory=list)
    switched: list[bool] = field(default_factory=list)
    belief_in_true: list[float | None] = field(default_factory=list)


def run_agent_on_seed(agent_kind: str, seed: int) -> SeedLog:
    env = FrameSwitchEnv(FRAMES, switch_period=SWITCH_PERIOD, jitter=JITTER, noise_sigma=NOISE_SIGMA, seed=seed)
    if agent_kind == "rf_aware":
        agent = RFAwareAgent(FRAMES)
    elif agent_kind == "flat":
        agent = FlatBaselineAgent()
    elif agent_kind == "flat_oracle":
        agent = FlatOracleAgent(len(FRAMES))
    else:
        raise ValueError(agent_kind)

    log = SeedLog()
    for _ in range(N_STEPS):
        info = env.observe()
        if agent_kind == "flat_oracle":
            predict = agent.act(info.raw, info.active_frame_index)
        else:
            predict = agent.act(info.raw)
        reward = env.feedback(predict)
        agent.update(reward)

        log.rewards.append(reward)
        log.switched.append(info.switched)
        log.belief_in_true.append(agent.belief_in(info.active_frame_index) if agent_kind == "rf_aware" else None)
    return log


def recovery_curve(log: SeedLog) -> tuple[float, float]:
    """Mean accuracy in the RECOVERY_WINDOW right after a switch vs. the
    STEADY_WINDOW later, averaged across all switches in this seed's run."""
    switch_indices = [i for i, s in enumerate(log.switched) if s]
    recovery_accs, steady_accs = [], []
    for idx in switch_indices:
        rec = log.rewards[idx: idx + RECOVERY_WINDOW]
        steady = log.rewards[idx + STEADY_WINDOW[0]: idx + STEADY_WINDOW[1]]
        if rec:
            recovery_accs.append(sum(rec) / len(rec))
        if steady:
            steady_accs.append(sum(steady) / len(steady))
    return (
        statistics.mean(recovery_accs) if recovery_accs else float("nan"),
        statistics.mean(steady_accs) if steady_accs else float("nan"),
    )


def calibration(log: SeedLog) -> float:
    """Mean belief mass placed on the true active frame during steady windows (RF-aware only)."""
    switch_indices = [i for i, s in enumerate(log.switched) if s]
    vals = []
    for idx in switch_indices:
        window = log.belief_in_true[idx + STEADY_WINDOW[0]: idx + STEADY_WINDOW[1]]
        vals.extend(v for v in window if v is not None)
    return statistics.mean(vals) if vals else float("nan")


def main() -> None:
    print(f"{'agent':<12} {'overall_acc':>12} {'(stdev)':>9} {'recovery_acc':>13} {'steady_acc':>11} {'belief_in_true':>15}")
    per_agent_overall: dict[str, list[float]] = {}
    for agent_kind in ("flat", "rf_aware", "flat_oracle"):
        overall, recovery, steady, calib = [], [], [], []
        for seed in SEEDS:
            log = run_agent_on_seed(agent_kind, seed)
            overall.append(sum(log.rewards) / len(log.rewards))
            r, s = recovery_curve(log)
            recovery.append(r)
            steady.append(s)
            if agent_kind == "rf_aware":
                calib.append(calibration(log))
        per_agent_overall[agent_kind] = overall
        calib_str = f"{statistics.mean(calib):.3f}" if calib else "n/a"
        print(f"{agent_kind:<12} {statistics.mean(overall):>12.3f} {statistics.pstdev(overall):>9.3f} "
              f"{statistics.mean(recovery):>13.3f} {statistics.mean(steady):>11.3f} {calib_str:>15}")

    print()
    for a, b in (("rf_aware", "flat"), ("rf_aware", "flat_oracle")):
        diffs = [x - y for x, y in zip(per_agent_overall[a], per_agent_overall[b])]
        wins = sum(1 for d in diffs if d > 0)
        print(f"paired overall_acc({a}) - overall_acc({b}): mean={statistics.mean(diffs):+.3f} "
              f"stdev={statistics.pstdev(diffs):.3f} wins={wins}/{len(diffs)} seeds")


if __name__ == "__main__":
    main()
