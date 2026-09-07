"""Frame-discovery agents for Experiment 4 (docs/research-agenda.md #7a).

Both agents wrap an RFAwareAgent by composition (not subclassing) and
mutate its public `frames`/`belief` lists directly when a discovery
fires -- this reuses RFAwareAgent's Bayesian belief-update mechanics
verbatim rather than reimplementing them, so any future change to that
mechanism doesn't need to be duplicated here.

Both use identical trigger logic (non-overlapping windows, one exact
binomial tail test per window -- see `_should_discover`) so any accuracy
difference between them is attributable to *what* gets appended on
trigger, not to *when*: DiscoveringRFAgent fits a `(baseline, direction)`
pair from the window's reward-implied labels; RandomDiscoveryAgent appends
a randomly generated one instead. This is the confound control
docs/research-agenda.md #7a specifies must exist from the start, following
the lesson from experiment 3 (docs/experiments/exp03_cross_domain_transfer).
"""
from __future__ import annotations

import math
import random

from transintelligence import ReferenceFrame

from experiments.exp01_frame_conditioning.agents import RFAwareAgent


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), computed exactly."""
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(0, k + 1))


def _fit_frame(raws: list[float], implied_labels: list[int], name: str, grid_step: float = 0.05) -> ReferenceFrame:
    """Grid search over (baseline, direction) for the pair that best matches
    the window's reward-implied labels -- the same reward-implies-label
    trick LearnedEmbeddingAgent uses, applied to a one-shot fit rather than
    an online gradient step."""
    best_correct = -1
    best_baseline = 0.5
    best_direction = "higher_is_better"
    n_steps = round(1.0 / grid_step)
    for direction in ("higher_is_better", "lower_is_better"):
        for i in range(n_steps + 1):
            baseline = i * grid_step
            correct = 0
            for raw, label in zip(raws, implied_labels):
                score = raw - baseline
                if direction == "lower_is_better":
                    score = -score
                predicted = 1 if score > 0 else -1
                correct += int(predicted == label)
            if correct > best_correct:
                best_correct, best_baseline, best_direction = correct, baseline, direction
    return ReferenceFrame(name, baseline=best_baseline, metadata={"direction": best_direction})


class _DiscoveringWrapperBase:
    """Shared windowing/trigger/bookkeeping. Subclasses implement
    `_propose_frame` for what gets appended when the trigger fires."""

    def __init__(self, initial_frames: list[ReferenceFrame], window: int = 40, min_accuracy: float = 0.85,
                 alpha: float = 0.01, max_frames: int | None = None, error_rate: float = 0.05, seed: int = 0):
        self.inner = RFAwareAgent(list(initial_frames), error_rate=error_rate)
        self.window = window
        self.min_accuracy = min_accuracy
        self.alpha = alpha
        self.max_frames = max_frames if max_frames is not None else len(initial_frames) + 2
        self.rng = random.Random(seed)

        self._window_rewards: list[int] = []
        self._window_raws: list[float] = []
        self._window_labels: list[int] = []
        self._window_true_frame_indices: list[int] = []
        self._last_raw = 0.0
        self._last_prediction = 1
        self._discovery_count = 0

        # Logging for metrics (docs/research-agenda.md #7a): every trigger,
        # whether or not it was acted on (capped), and whether the window it
        # fired on was genuinely a novel (held-out) regime or a noisy dip in
        # a known one -- populated by the caller via `note_true_frame`.
        self.trigger_log: list[dict] = []
        self._step_count = 0

    def act(self, raw: float) -> int:
        self._last_raw = raw
        self._last_prediction = self.inner.act(raw)
        return self._last_prediction

    def update(self, reward: int, true_is_held_out: bool = False) -> None:
        """`true_is_held_out` is ground truth supplied by the caller, used
        only for the trigger_log's held-out-fraction metric -- never for the
        agent's own decision-making (that would defeat the point of testing
        discovery). Optional/defaults to False so this agent still works as
        a drop-in RFAwareAgent-shaped agent when ground truth isn't
        available or the metric isn't needed."""
        self.inner.update(reward)
        implied = self._last_prediction if reward == 1 else -self._last_prediction
        self._window_rewards.append(reward)
        self._window_raws.append(self._last_raw)
        self._window_labels.append(implied)
        self._window_true_frame_indices.append(int(true_is_held_out))
        self._step_count += 1

        if len(self._window_rewards) >= self.window:
            self._check_window()
            self._window_rewards.clear()
            self._window_raws.clear()
            self._window_labels.clear()
            self._window_true_frame_indices.clear()

    def _check_window(self) -> None:
        successes = sum(self._window_rewards)
        p_value = _binom_cdf(successes, self.window, self.min_accuracy)
        if p_value >= self.alpha:
            return  # window's performance is consistent with known frames; no trigger

        held_out_fraction = sum(self._window_true_frame_indices) / len(self._window_true_frame_indices)
        acted = len(self.inner.frames) < self.max_frames
        self.trigger_log.append({
            "step": self._step_count,
            "p_value": p_value,
            "held_out_fraction": held_out_fraction,
            "acted": acted,
        })
        if not acted:
            return

        new_frame = self._propose_frame(list(self._window_raws), list(self._window_labels))
        self.inner.frames = self.inner.frames + [new_frame]
        n = len(self.inner.frames)
        self.inner.belief = [1.0 / n] * n
        self._discovery_count += 1

    def _propose_frame(self, raws: list[float], labels: list[int]) -> ReferenceFrame:
        raise NotImplementedError

    @property
    def frame_count(self) -> int:
        return len(self.inner.frames)

    @property
    def first_acted_discovery_step(self) -> int | None:
        acted = [t for t in self.trigger_log if t["acted"]]
        return acted[0]["step"] if acted else None


class DiscoveringRFAgent(_DiscoveringWrapperBase):
    """Fits a new (baseline, direction) frame from the triggering window's
    reward-implied labels via grid search -- the treatment condition."""

    name = "discovering_rf"

    def _propose_frame(self, raws: list[float], labels: list[int]) -> ReferenceFrame:
        return _fit_frame(raws, labels, name=f"discovered_{self._discovery_count}")


class RandomDiscoveryAgent(_DiscoveringWrapperBase):
    """Appends a randomly generated frame on the identical trigger --
    the confound control. If this recovers nearly as much accuracy as
    DiscoveringRFAgent, the result is "a growable candidate list helps",
    not "discovery is finding anything real"."""

    name = "random_discovery"

    def _propose_frame(self, raws: list[float], labels: list[int]) -> ReferenceFrame:
        baseline = self.rng.uniform(0.0, 1.0)
        direction = self.rng.choice(("higher_is_better", "lower_is_better"))
        return ReferenceFrame(f"random_{self._discovery_count}", baseline=baseline, metadata={"direction": direction})
