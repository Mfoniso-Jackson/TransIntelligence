"""Experiment 5, per-segment accuracy sweep (docs/research-agenda.md #7b) --
the noise sweep in run.py benchmarks change_points() (did we detect the
right steps), not regime_segments()'s actual output (are the resulting
segment mean estimates any good). A detector can have decent recall/
precision while still producing poor segment-mean estimates if boundaries
are off by a few steps, or good estimates could survive occasional missed/
spurious detections if the resulting merged/split segments still average
out close to the truth -- these are different questions, and the first
doesn't answer the second.

Per state, the "estimated" value is whatever RegimeSegment.mean_value its
containing segment reports; the "true" value is the mean of the actual
generating regime that state belongs to (REGIME_MEANS[step // regime_length]).
Mean absolute error (MAE) over all states is the headline metric, with two
reference points: a "no segmentation" floor (single global mean) and a
"ground-truth segmentation" ceiling (means computed from the TRUE regime
boundaries, not detected ones) -- CUSUMTemporalReasoner's number should
land between these two, and how close to the ceiling it gets is the
actual question, not recall/precision alone.

Run: PYTHONPATH=. python experiments/exp05_regime_change_detection/sweep_segment_accuracy.py
"""
from __future__ import annotations

import statistics

from transintelligence import StateHistory
from transintelligence.reasoning.temporal import CUSUMTemporalReasoner

from experiments.exp05_regime_change_detection.run import NOISE_LEVELS, REGIME_LENGTH, REGIME_MEANS, SEEDS, make_history


def _true_regime_index(step: int, regime_length: int = REGIME_LENGTH) -> int:
    return min(step // regime_length, len(REGIME_MEANS) - 1)


def _states_with_steps(history: StateHistory, key: str = "v") -> list[tuple[int, float]]:
    t0 = history.states[0].timestamp
    return [(round((s.timestamp - t0).total_seconds() / 60), s.values[key])
            for s in history.states if key in s.values]


def detected_segmentation_mae(history: StateHistory) -> float:
    reasoner = CUSUMTemporalReasoner()
    segments = reasoner.regime_segments(history, "v")
    t0 = history.states[0].timestamp
    errors = []
    for seg in segments:
        true_means = [REGIME_MEANS[_true_regime_index(round((s.timestamp - t0).total_seconds() / 60))] for s in seg.states]
        errors.extend(abs(seg.mean_value - tm) for tm in true_means)
    return statistics.mean(errors) if errors else float("nan")


def no_segmentation_mae(history: StateHistory) -> float:
    states_steps = _states_with_steps(history)
    global_mean = statistics.mean(v for _, v in states_steps)
    errors = [abs(global_mean - REGIME_MEANS[_true_regime_index(step)]) for step, _ in states_steps]
    return statistics.mean(errors)


def ground_truth_segmentation_mae(history: StateHistory) -> float:
    states_steps = _states_with_steps(history)
    by_regime: dict[int, list[float]] = {}
    for step, v in states_steps:
        by_regime.setdefault(_true_regime_index(step), []).append(v)
    regime_means = {idx: statistics.mean(vals) for idx, vals in by_regime.items()}
    errors = [abs(regime_means[_true_regime_index(step)] - REGIME_MEANS[_true_regime_index(step)]) for step, _ in states_steps]
    return statistics.mean(errors)


def main() -> None:
    print(f"{'noise_sigma':<12} {'no_segmentation':>16} {'detected_segments':>18} {'ground_truth_segments':>22}")
    for sigma in NOISE_LEVELS:
        no_seg, detected, gt = [], [], []
        for seed in SEEDS:
            history, _ = make_history(seed, sigma)
            no_seg.append(no_segmentation_mae(history))
            detected.append(detected_segmentation_mae(history))
            gt.append(ground_truth_segmentation_mae(history))
        print(f"{sigma:<12} {statistics.mean(no_seg):>16.4f} {statistics.mean(detected):>18.4f} {statistics.mean(gt):>22.4f}")


if __name__ == "__main__":
    main()
