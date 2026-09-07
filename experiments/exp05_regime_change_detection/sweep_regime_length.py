"""Experiment 5, regime-length sweep (docs/research-agenda.md #7b) --
CUSUMTemporalReasoner's burn_in window couples its minimum usable regime
length: a regime shorter than burn_in can never be calibrated on before
the next change happens. This sweep finds exactly where that transition
occurs, holding noise fixed at a moderate, well-behaved level (see
sweep_noise via run.py for the noise dimension instead).

Run: PYTHONPATH=. python experiments/exp05_regime_change_detection/sweep_regime_length.py
"""
from __future__ import annotations

import statistics

from transintelligence.reasoning.temporal import CUSUMTemporalReasoner

from experiments.exp05_regime_change_detection.run import T0, evaluate, make_history

NOISE_SIGMA = 0.05
REGIME_LENGTHS = [10, 20, 30, 40, 60, 100]
SEEDS = list(range(20))


def run_trial(seed: int, regime_length: int) -> dict:
    history, true_change_steps = make_history(seed, NOISE_SIGMA, regime_length=regime_length)
    reasoner = CUSUMTemporalReasoner()  # class defaults: burn_in=30
    changes = reasoner.change_points(history, "v")
    detected_steps = [round((c - T0).total_seconds() / 60) for c in changes]
    return evaluate(detected_steps, true_change_steps)


def main() -> None:
    print(f"{'regime_length':<15} {'recall':>8} {'precision':>10}")
    for regime_length in REGIME_LENGTHS:
        rows = [run_trial(seed, regime_length) for seed in SEEDS]
        recalls = [r["recall"] for r in rows]
        precisions = [r["precision"] for r in rows if r["precision"] == r["precision"]]
        precision_str = f"{statistics.mean(precisions):.3f}" if precisions else "n/a"
        print(f"{regime_length:<15} {statistics.mean(recalls):>8.3f} {precision_str:>10}")


if __name__ == "__main__":
    main()
