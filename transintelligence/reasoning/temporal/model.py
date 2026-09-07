"""Baseline temporal reasoning: regime-change detection and comparison over
a StateHistory's numeric value streams (Phase 4, docs/research-agenda.md #7b).

`state_at`/`trajectory` already exist on StateHistory itself (Phase 1).
What was missing, per the master context's Phase 4 list and
docs/reasoning.md's description of this module, is genuinely derived
temporal reasoning: detecting *when* a tracked value's regime changed, and
segmenting a history into contiguous regimes -- not just looking values up
by time.

CUSUMTemporalReasoner bundles two established mechanisms for this,
mirroring how BaselineRelativeReasoner bundles evaluate/compare/rank/
sensitivity under one Phase-1 baseline rather than one class per method:

- Change detection: a two-sided CUSUM control chart (Page, *Continuous
  Inspection Schemes*, Biometrika 41(1-2), 1954, see
  docs/related-work.md §9a) -- a simple, well-established, self-
  calibrating method for detecting a shift in a series' mean, chosen over
  a full Bayesian changepoint or HMM treatment (Adams & MacKay 2007;
  Rabiner 1989) for the same reason BaselineRelativeReasoner started with
  a location-shift formula rather than a probabilistic model: smallest
  mechanism that could produce a falsifiable result. It restarts and
  recalibrates after each detected change to find multiple regime changes
  across a long series, a practical adaptation of Page's original
  single-changepoint formulation, not part of the original method.
  `joint_change_points()` extends this to multiple simultaneously-tracked
  keys (Crosier, *Multivariate Generalizations of Cumulative Sum
  Quality-Control Schemes*, Technometrics 30(3), 1988 -- his simpler
  "reduce to a scalar, then CUSUM" variant, assuming independence across
  keys rather than modeling cross-covariance).
- Temporal comparison: Dynamic Time Warping (Sakoe, Chiba, *Dynamic
  Programming Algorithm Optimization for Spoken Word Recognition*, IEEE
  Trans. Acoustics, Speech, Signal Processing 26(1), 1978) -- the basic
  symmetric form, with no slope constraint (the paper's own refinement,
  not implemented here), for comparing two trajectories that may be
  time-shifted or of different lengths, where naive pointwise comparison
  fails.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from transintelligence.core.common import TimeWindow
from transintelligence.core.states import State, StateHistory


@dataclass(frozen=True)
class RegimeSegment:
    window: TimeWindow
    states: tuple[State, ...]
    mean_value: float


def dynamic_time_warp(series_a: list[float], series_b: list[float]) -> float:
    """Basic symmetric-form DTW distance (Sakoe & Chiba 1978, no slope
    constraint) between two numeric sequences, which may differ in length
    or be time-shifted relative to each other. 0.0 for identical
    sequences; grows with genuine shape difference, not with length or
    alignment offset the way naive pointwise (same-index) comparison
    does. O(n*m) time and space -- fine at the scale this repo operates
    at, not intended for long sequences."""
    n, m = len(series_a), len(series_b)
    if n == 0 or m == 0:
        raise ValueError("cannot compute DTW distance for an empty series")
    inf = float("inf")
    d = [[inf] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(series_a[i - 1] - series_b[j - 1])
            d[i][j] = cost + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[n][m]


class CUSUMTemporalReasoner:
    """Self-calibrating two-sided CUSUM change detector over a
    StateHistory's `values[key]` numeric stream.

    Calibrates target mean and drift/threshold (`k`, `h`) from a rolling
    burn-in window of `burn_in` states rather than requiring them as fixed
    external parameters.

    Defaults (`burn_in=30`, `h_sigma=8.0`) were empirically calibrated, not
    taken from textbook convention: the "conventional" statistical-process-
    control starting points (k=0.5σ, h=5σ, small burn-in) assume σ is
    known or well-estimated. With a short self-calibrated burn-in window
    that assumption breaks down -- the σ estimate itself is too noisy, and
    since the CUSUM statistic is tested at *every* subsequent step (not
    once), even a nominal "5-sigma" threshold produced a **47% false-positive
    rate on genuinely stationary data** at `burn_in=10` (measured directly,
    200 trials). Increasing `burn_in` to 30 and `h_sigma` to 8.0 brings that
    down to ~6% -- still not negligible (this is a real, stated limitation,
    not a solved problem), but far better than the untested "conventional"
    defaults would have been. See
    experiments/exp05_regime_change_detection/RESULTS.md for the full
    calibration sweep and the corresponding detection-power/delay tradeoff.
    """

    def __init__(self, burn_in: int = 30, k_sigma: float = 0.5, h_sigma: float = 8.0, min_sigma: float = 1e-9):
        if burn_in < 2:
            raise ValueError("burn_in must be at least 2 to estimate a variance")
        self.burn_in = burn_in
        self.k_sigma = k_sigma
        self.h_sigma = h_sigma
        self.min_sigma = min_sigma

    def _numeric_series(self, history: StateHistory, key: str) -> list[tuple[datetime, float]]:
        series = []
        for s in history.states:
            if key in s.values and isinstance(s.values[key], (int, float)) and not isinstance(s.values[key], bool):
                series.append((s.timestamp, float(s.values[key])))
        return series

    def change_points(self, history: StateHistory, key: str) -> list[datetime]:
        series = self._numeric_series(history, key)
        changes: list[datetime] = []
        idx = 0
        while idx + self.burn_in <= len(series):
            window_vals = [v for _, v in series[idx: idx + self.burn_in]]
            mu0 = statistics.mean(window_vals)
            sigma = max(statistics.pstdev(window_vals), self.min_sigma)
            k = self.k_sigma * sigma
            h = self.h_sigma * sigma

            s_pos = s_neg = 0.0
            changed_at_offset = None
            for offset, (_, x) in enumerate(series[idx + self.burn_in:]):
                s_pos = max(0.0, s_pos + (x - mu0) - k)
                s_neg = max(0.0, s_neg - (x - mu0) - k)
                if s_pos > h or s_neg > h:
                    changed_at_offset = offset
                    break

            if changed_at_offset is None:
                break
            change_index = idx + self.burn_in + changed_at_offset
            changes.append(series[change_index][0])
            idx = change_index
        return changes

    def _numeric_series_multi(self, history: StateHistory, keys: list[str]) -> list[tuple[datetime, list[float]]]:
        """States where ALL of `keys` are present and numeric -- states
        missing even one tracked key are dropped entirely, not partially
        included. A real requirement of this method, not an edge case."""
        aligned = []
        for s in history.states:
            if all(k in s.values and isinstance(s.values[k], (int, float)) and not isinstance(s.values[k], bool)
                   for k in keys):
                aligned.append((s.timestamp, [float(s.values[k]) for k in keys]))
        return aligned

    def joint_change_points(self, history: StateHistory, keys: list[str]) -> list[datetime]:
        """Multi-key regime-change detection: reduces each multivariate
        observation to a scalar by summing per-key standardized (z-scored)
        deviations from their own burn-in means, then runs the same
        two-sided CUSUM on that combined scalar (Crosier, *Multivariate
        Generalizations of Cumulative Sum Quality-Control Schemes*,
        Technometrics 30(3), 1988 -- his "reduce to scalar first" variant,
        not his alternative direct-vector-CUSUM procedure).

        Assumes independence across keys (no cross-covariance term) --
        a real simplification relative to a full multivariate treatment,
        and the reason this sums per-key z-scores rather than using a
        Mahalanobis-style statistic. `h_sigma`/`k_sigma` are reused
        unchanged from the single-key calibration; see
        experiments/exp05_regime_change_detection/RESULTS.md for why that
        held up (or didn't) empirically rather than being assumed."""
        aligned = self._numeric_series_multi(history, keys)
        n_keys = len(keys)
        changes: list[datetime] = []
        idx = 0
        while idx + self.burn_in <= len(aligned):
            window = aligned[idx: idx + self.burn_in]
            mus = [statistics.mean(w[1][i] for w in window) for i in range(n_keys)]
            sigmas = [max(statistics.pstdev([w[1][i] for w in window]), self.min_sigma) for i in range(n_keys)]

            s_pos = s_neg = 0.0
            changed_at_offset = None
            for offset, (_, vals) in enumerate(aligned[idx + self.burn_in:]):
                combined = sum((vals[i] - mus[i]) / sigmas[i] for i in range(n_keys)) / (n_keys ** 0.5)
                s_pos = max(0.0, s_pos + combined - self.k_sigma)
                s_neg = max(0.0, s_neg - combined - self.k_sigma)
                if s_pos > self.h_sigma or s_neg > self.h_sigma:
                    changed_at_offset = offset
                    break

            if changed_at_offset is None:
                break
            change_index = idx + self.burn_in + changed_at_offset
            changes.append(aligned[change_index][0])
            idx = change_index
        return changes

    def regime_segments(self, history: StateHistory, key: str) -> list[RegimeSegment]:
        series = self._numeric_series(history, key)
        if not series:
            return []
        change_ts = self.change_points(history, key)

        boundaries = [series[0][0]] + change_ts + [None]  # None sentinel = open-ended final segment
        states_by_time = [s for s in history.states if key in s.values
                           and isinstance(s.values[key], (int, float)) and not isinstance(s.values[key], bool)]

        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            if end is None:
                seg_states = [s for s in states_by_time if s.timestamp >= start]
            else:
                seg_states = [s for s in states_by_time if start <= s.timestamp < end]
            if not seg_states:
                continue
            values = [s.values[key] for s in seg_states]
            segments.append(RegimeSegment(
                window=TimeWindow(start=seg_states[0].timestamp, end=seg_states[-1].timestamp),
                states=tuple(seg_states),
                mean_value=statistics.mean(values),
            ))
        return segments

    def compare(self, a: State, b: State) -> dict[str, tuple[Any, Any]]:
        """Keys present in either state where the values differ -- same
        shape as ReferenceFrame.differences()."""
        keys = set(a.values) | set(b.values)
        return {k: (a.values.get(k), b.values.get(k)) for k in keys if a.values.get(k) != b.values.get(k)}

    def trajectory_distance(self, a: StateHistory, b: StateHistory, key: str) -> float:
        """DTW distance between two StateHistories' `values[key]` numeric
        streams -- a genuine trajectory comparison, unlike `compare()`
        (a single-state pointwise diff). Ignores each history's actual
        timestamps beyond ordering: DTW aligns by shape, not by wall-clock
        time, which is the point of using it over a same-index comparison."""
        series_a = [v for _, v in self._numeric_series(a, key)]
        series_b = [v for _, v in self._numeric_series(b, key)]
        return dynamic_time_warp(series_a, series_b)
