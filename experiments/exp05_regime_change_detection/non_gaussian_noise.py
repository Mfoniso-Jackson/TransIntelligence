"""Experiment 5, non-Gaussian noise (docs/research-agenda.md #7b) -- CUSUM's
calibration (mean/sigma from a burn-in window, threshold as a sigma
multiple) implicitly assumes light-tailed, roughly Gaussian noise. Page's
original 1954 formulation and this class's own empirical calibration
(see model.py docstring) were both derived/tested against Gaussian noise.
This tests what happens when that assumption is violated -- deliberately,
not as an afterthought -- via a standard contaminated-Gaussian mixture: 5%
of observations drawn from a much wider Gaussian ("outliers"), 95% from
the normal one, at matched overall structure but genuinely heavy-tailed.

No single canonical citation is adopted for "CUSUM's non-normal
robustness" here -- the SPC literature on this is a family of results
(robust/nonparametric CUSUM variants exist specifically because standard
CUSUM is not automatically robust), not one seminal paper the way Page
1954 is for CUSUM itself. Tested empirically instead of asserted from
authority.

Run: PYTHONPATH=. python experiments/exp05_regime_change_detection/non_gaussian_noise.py
"""
from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta, timezone

from transintelligence import State, StateHistory
from transintelligence.reasoning.temporal import CUSUMTemporalReasoner

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
SIGMA = 0.02
P_OUTLIER = 0.05
OUTLIER_SCALE = 5.0
SEEDS = list(range(200))
SHIFT_SEEDS = list(range(30))


def contaminated_noise(rng: random.Random, sigma: float = SIGMA) -> float:
    if rng.random() < P_OUTLIER:
        return rng.gauss(0, sigma * OUTLIER_SCALE)
    return rng.gauss(0, sigma)


def stable_history(seed: int, heavy_tailed: bool, n: int = 100) -> StateHistory:
    rng = random.Random(seed)
    noise_fn = contaminated_noise if heavy_tailed else (lambda r: r.gauss(0, SIGMA))
    return StateHistory([State("x", {"v": 0.5 + noise_fn(rng)}, T0 + timedelta(minutes=i)) for i in range(n)])


def shift_history(seed: int, heavy_tailed: bool, true_step: int = 60, n: int = 120) -> StateHistory:
    rng = random.Random(seed)
    noise_fn = contaminated_noise if heavy_tailed else (lambda r: r.gauss(0, 0.05))
    states = []
    for i in range(n):
        mu = 0.3 if i < true_step else 0.9
        states.append(State("x", {"v": mu + noise_fn(rng)}, T0 + timedelta(minutes=i)))
    return StateHistory(states)


def false_positive_rate(heavy_tailed: bool) -> float:
    reasoner = CUSUMTemporalReasoner()
    hits = sum(1 for seed in SEEDS if reasoner.change_points(stable_history(seed, heavy_tailed), "v"))
    return hits / len(SEEDS)


def recall(heavy_tailed: bool, true_step: int = 60, tolerance: int = 15) -> float:
    reasoner = CUSUMTemporalReasoner()
    hits = []
    for seed in SHIFT_SEEDS:
        history = shift_history(seed, heavy_tailed, true_step)
        changes = reasoner.change_points(history, "v")
        steps = [round((c - T0).total_seconds() / 60) for c in changes]
        hits.append(any(abs(s - true_step) <= tolerance for s in steps))
    return statistics.mean(hits)


def main() -> None:
    print(f"{'condition':<14} {'false_positive_rate':>20} {'recall_on_real_shift':>22}")
    for heavy in (False, True):
        label = "heavy_tailed" if heavy else "gaussian"
        print(f"{label:<14} {false_positive_rate(heavy):>20.3f} {recall(heavy):>22.3f}")


if __name__ == "__main__":
    main()
