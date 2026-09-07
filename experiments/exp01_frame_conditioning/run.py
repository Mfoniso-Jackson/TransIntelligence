"""Experiment 1 -- does explicit frame structure beat a flat baseline under
matched (zero) information about which frame is active, and does that
advantage survive against a baseline given the true frame id directly, a
learned-embedding baseline, and a clean inference-cost oracle?

See docs/research-agenda.md #5 for the hypothesis and falsification
criterion, and docs/related-work.md #2 for why the flat-vs-structured
question alone isn't enough to claim novelty. Full results and the fixes
that got the numbers here are in RESULTS.md -- this single-configuration
run is one data point; sweep.py in this directory sweeps noise and switch
frequency around it, reusing the helpers defined here.

Run: PYTHONPATH=. python experiments/exp01_frame_conditioning/run.py
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from transintelligence import ReferenceFrame

from environments.transworld import FrameSwitchEnv
from experiments.exp01_frame_conditioning.agents import (
    FlatBaselineAgent,
    FlatOracleAgent,
    LearnedEmbeddingAgent,
    RFAwareAgent,
    TrueOracleAgent,
)

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
# Windows are defined as fractions of switch_period so recovery_curve() and
# calibration() stay meaningful when sweep.py varies switch_period: at the
# default period=40 these reduce to exactly (0,10) and (30,40), matching the
# original fixed-window design.
RECOVERY_FRACTION = 0.25          # first quarter of the period
STEADY_FRACTION = (0.75, 1.0)     # last quarter of the period


@dataclass
class SeedLog:
    rewards: list[int] = field(default_factory=list)
    switched: list[bool] = field(default_factory=list)
    belief_in_true: list[float | None] = field(default_factory=list)


def run_agent_on_seed(agent_kind: str, seed: int, switch_period: int = SWITCH_PERIOD,
                       jitter: int = JITTER, noise_sigma: float = NOISE_SIGMA, n_steps: int = N_STEPS) -> SeedLog:
    env = FrameSwitchEnv(FRAMES, switch_period=switch_period, jitter=jitter, noise_sigma=noise_sigma, seed=seed)
    if agent_kind == "rf_aware":
        agent = RFAwareAgent(FRAMES)
    elif agent_kind == "flat":
        agent = FlatBaselineAgent()
    elif agent_kind == "learned_embedding":
        agent = LearnedEmbeddingAgent(len(FRAMES), seed=seed)
    elif agent_kind == "flat_oracle":
        agent = FlatOracleAgent(len(FRAMES))
    elif agent_kind == "true_oracle":
        agent = TrueOracleAgent(FRAMES)
    else:
        raise ValueError(agent_kind)

    needs_frame_id = agent_kind in ("flat_oracle", "true_oracle")
    log = SeedLog()
    for _ in range(n_steps):
        info = env.observe()
        if needs_frame_id:
            predict = agent.act(info.raw, info.active_frame_index)
        else:
            predict = agent.act(info.raw)
        reward = env.feedback(predict)
        agent.update(reward)

        log.rewards.append(reward)
        log.switched.append(info.switched)
        log.belief_in_true.append(agent.belief_in(info.active_frame_index) if agent_kind == "rf_aware" else None)
    return log


def _windows(switch_period: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Recovery and steady windows as step-offsets after a switch, scaled to
    switch_period (see RECOVERY_FRACTION/STEADY_FRACTION) so these stay
    meaningful when sweep.py varies the switch frequency."""
    recovery = (0, max(1, round(switch_period * RECOVERY_FRACTION)))
    steady = (round(switch_period * STEADY_FRACTION[0]), max(round(switch_period * STEADY_FRACTION[0]) + 1, round(switch_period * STEADY_FRACTION[1])))
    return recovery, steady


def recovery_curve(log: SeedLog, switch_period: int = SWITCH_PERIOD) -> tuple[float, float]:
    """Mean accuracy in the recovery window right after a switch vs. the
    steady window later, averaged across all switches in this seed's run."""
    (rec_start, rec_end), (steady_start, steady_end) = _windows(switch_period)
    switch_indices = [i for i, s in enumerate(log.switched) if s]
    recovery_accs, steady_accs = [], []
    for idx in switch_indices:
        rec = log.rewards[idx + rec_start: idx + rec_end]
        steady = log.rewards[idx + steady_start: idx + steady_end]
        if rec:
            recovery_accs.append(sum(rec) / len(rec))
        if steady:
            steady_accs.append(sum(steady) / len(steady))
    return (
        statistics.mean(recovery_accs) if recovery_accs else float("nan"),
        statistics.mean(steady_accs) if steady_accs else float("nan"),
    )


def calibration(log: SeedLog, switch_period: int = SWITCH_PERIOD) -> float:
    """Mean belief mass placed on the true active frame during steady windows (RF-aware only)."""
    _, (steady_start, steady_end) = _windows(switch_period)
    switch_indices = [i for i, s in enumerate(log.switched) if s]
    vals = []
    for idx in switch_indices:
        window = log.belief_in_true[idx + steady_start: idx + steady_end]
        vals.extend(v for v in window if v is not None)
    return statistics.mean(vals) if vals else float("nan")


AGENT_KINDS = ("flat", "learned_embedding", "rf_aware", "flat_oracle", "true_oracle")


def main() -> None:
    print(f"{'agent':<18} {'overall_acc':>12} {'(stdev)':>9} {'recovery_acc':>13} {'steady_acc':>11} {'belief_in_true':>15}")
    per_agent_overall: dict[str, list[float]] = {}
    for agent_kind in AGENT_KINDS:
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
        print(f"{agent_kind:<18} {statistics.mean(overall):>12.3f} {statistics.pstdev(overall):>9.3f} "
              f"{statistics.mean(recovery):>13.3f} {statistics.mean(steady):>11.3f} {calib_str:>15}")

    print()
    for a, b in (
        ("rf_aware", "flat"),
        ("rf_aware", "learned_embedding"),
        ("learned_embedding", "flat"),
        ("rf_aware", "flat_oracle"),
        ("true_oracle", "rf_aware"),
    ):
        diffs = [x - y for x, y in zip(per_agent_overall[a], per_agent_overall[b])]
        wins = sum(1 for d in diffs if d > 0)
        print(f"paired overall_acc({a}) - overall_acc({b}): mean={statistics.mean(diffs):+.3f} "
              f"stdev={statistics.pstdev(diffs):.3f} wins={wins}/{len(diffs)} seeds")


if __name__ == "__main__":
    main()
