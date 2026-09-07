"""Experiment 5 -- does CUSUMTemporalReasoner detect regime changes in a
StateHistory, and how does that degrade under noise? (docs/research-agenda.md
#7b). Mirrors the noise-sweep discipline from experiments 1/2/4: CUSUM
hyperparameters are held FIXED across noise levels (the class defaults,
`burn_in=30`, `h_sigma=8.0`), not recalibrated per level, to test whether a
detector tuned once degrades gracefully or breaks sharply outside its
calibrated regime -- the same question asked of the frame-discovery
trigger in experiment 4, applied here to a genuine kernel primitive
instead of an RL research script.

Ground truth: synthetic State sequences built from K known regimes of
fixed length L, each with a distinct mean and Gaussian noise. A detection
counts as a true positive if it falls within TOLERANCE steps of a true
change point (each true change point matched to at most one detection).

Run: PYTHONPATH=. python experiments/exp05_regime_change_detection/run.py
"""
from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta, timezone

from transintelligence import State, StateHistory
from transintelligence.reasoning.temporal import CUSUMTemporalReasoner

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
REGIME_MEANS = [0.3, 0.7, 0.3, 0.6, 0.4]
REGIME_LENGTH = 60
TOLERANCE = 15  # steps
NOISE_LEVELS = [0.01, 0.02, 0.05, 0.1, 0.2]
SEEDS = list(range(20))


def make_history(seed: int, noise_sigma: float, regime_means: list[float] = REGIME_MEANS,
                  regime_length: int = REGIME_LENGTH) -> tuple[StateHistory, list[int]]:
    rng = random.Random(seed)
    states = []
    step = 0
    for mu in regime_means:
        for _ in range(regime_length):
            states.append(State("x", {"v": mu + rng.gauss(0, noise_sigma)}, T0 + timedelta(minutes=step)))
            step += 1
    true_change_steps = [i * regime_length for i in range(1, len(regime_means))]
    return StateHistory(states), true_change_steps


def evaluate(detected_steps: list[int], true_change_steps: list[int]) -> dict:
    matched_true = set()
    matched_detected = set()
    delays = []
    for d in detected_steps:
        for t in true_change_steps:
            if t in matched_true:
                continue
            if abs(d - t) <= TOLERANCE:
                matched_true.add(t)
                matched_detected.add(d)
                delays.append(d - t)
                break

    recall = len(matched_true) / len(true_change_steps) if true_change_steps else float("nan")
    precision = len(matched_detected) / len(detected_steps) if detected_steps else float("nan")
    mean_delay = statistics.mean(delays) if delays else float("nan")
    return {"recall": recall, "precision": precision, "mean_delay": mean_delay, "n_detected": len(detected_steps)}


def run_trial(seed: int, noise_sigma: float) -> dict:
    history, true_change_steps = make_history(seed, noise_sigma)
    reasoner = CUSUMTemporalReasoner()  # class defaults, fixed across the whole sweep
    changes = reasoner.change_points(history, "v")
    detected_steps = [round((c - T0).total_seconds() / 60) for c in changes]
    return evaluate(detected_steps, true_change_steps)


def main() -> None:
    print(f"{'noise_sigma':<12} {'recall':>8} {'precision':>10} {'mean_delay':>11} {'mean_n_detected':>16}")
    for sigma in NOISE_LEVELS:
        rows = [run_trial(seed, sigma) for seed in SEEDS]
        recalls = [r["recall"] for r in rows if r["recall"] == r["recall"]]
        precisions = [r["precision"] for r in rows if r["precision"] == r["precision"]]
        delays = [r["mean_delay"] for r in rows if r["mean_delay"] == r["mean_delay"]]
        n_detected = [r["n_detected"] for r in rows]
        print(f"{sigma:<12} {statistics.mean(recalls):>8.3f} "
              f"{statistics.mean(precisions) if precisions else float('nan'):>10.3f} "
              f"{statistics.mean(delays) if delays else float('nan'):>11.2f} "
              f"{statistics.mean(n_detected):>16.2f}")


if __name__ == "__main__":
    main()
