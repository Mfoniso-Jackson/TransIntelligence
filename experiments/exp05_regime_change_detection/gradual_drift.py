"""Experiment 5, gradual drift (docs/research-agenda.md #7b) -- CUSUM is
designed to detect abrupt mean shifts (Page 1954); does it fail, or fail
to detect at all, when the true regime change is a gradual ramp instead?

Working hypothesis going in: since `change_points()` recalibrates mu0/
sigma from a burn-in window at the *start* of each detection cycle and
then holds them fixed while monitoring, a sufficiently slow drift might
never accumulate enough deviation from that fixed reference to cross
threshold within a realistic series length -- i.e. the detector could go
"blind" to slow enough drift. That hypothesis was WRONG, and this script
is what showed it: recall stays at 1.00 even when the ramp spans 1000
steps in an 1100-step series (the drift never even completes), because
CUSUM's calibration is fixed once per cycle, not continuously re-chased --
so any persistent drift, however slow, eventually accumulates past
threshold. Detection delay grows with ramp length, but sub-linearly, not
catastrophically.

Run: PYTHONPATH=. python experiments/exp05_regime_change_detection/gradual_drift.py
"""
from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta, timezone

from transintelligence import State, StateHistory
from transintelligence.reasoning.temporal import CUSUMTemporalReasoner

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
NOISE_SIGMA = 0.05
START, END = 0.3, 0.9
RAMP_START = 60
TOLERANCE = 15
SEEDS = list(range(30))
RAMP_LENGTHS_AND_TOTAL_LENGTHS = [
    (1, 150), (5, 150), (10, 150), (20, 150), (40, 150), (80, 150), (150, 150),
    (300, 400), (600, 700), (1000, 1100),
]


def make_history(seed: int, ramp_len: int, total_len: int) -> StateHistory:
    rng = random.Random(seed)
    states = []
    for i in range(total_len):
        if i < RAMP_START:
            mu = START
        elif i < RAMP_START + ramp_len:
            frac = (i - RAMP_START) / ramp_len
            mu = START + frac * (END - START)
        else:
            mu = END
        states.append(State("x", {"v": mu + rng.gauss(0, NOISE_SIGMA)}, T0 + timedelta(minutes=i)))
    return StateHistory(states)


def run_trial(seed: int, ramp_len: int, total_len: int) -> tuple[bool, float]:
    reasoner = CUSUMTemporalReasoner()
    history = make_history(seed, ramp_len, total_len)
    changes = reasoner.change_points(history, "v")
    steps = [round((c - T0).total_seconds() / 60) for c in changes]
    hits = [s for s in steps if s >= RAMP_START - TOLERANCE]
    return (len(hits) > 0, min(hits) - RAMP_START if hits else float("nan"))


def main() -> None:
    print(f"{'ramp_length':<12} {'total_length':>13} {'recall':>8} {'mean_delay':>11}")
    for ramp_len, total_len in RAMP_LENGTHS_AND_TOTAL_LENGTHS:
        rows = [run_trial(seed, ramp_len, total_len) for seed in SEEDS]
        recalls = [r for r, _ in rows]
        delays = [d for _, d in rows if d == d]
        print(f"{ramp_len:<12} {total_len:>13} {statistics.mean(recalls):>8.2f} "
              f"{statistics.mean(delays) if delays else float('nan'):>11.1f}")


if __name__ == "__main__":
    main()
