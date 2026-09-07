"""Minimal synthetic environment for Experiment 1 (docs/research-agenda.md #5).

At each step, the environment shows the agent one entity's raw observed
value (a single, possibly noisy scalar). The agent predicts a binary
"conclusion" for it (+1/-1). Reward is 1 if that prediction matches the
*currently active* reference frame's true conclusion -- `sign(evaluate(raw,
frame))`, reusing the same definition Experiment 2 established and the
fixed `sensitivity()` relies on -- else 0. The active frame is drawn from a
small, known candidate list and switches, unannounced, every
`switch_period` steps (+/- jitter). The agent is never told which frame is
active; only `active_frame_index` (exposed for logging/oracle-control
agents, not for agents under test) reveals it directly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from transintelligence import ReferenceFrame


def true_conclusion(raw: float, frame: ReferenceFrame) -> int:
    score = raw - frame.baseline
    if frame.metadata.get("direction", "higher_is_better") == "lower_is_better":
        score = -score
    return 1 if score > 0 else -1


@dataclass
class StepInfo:
    raw: float
    active_frame_index: int
    switched: bool
    reward: int | None = None


class FrameSwitchEnv:
    def __init__(self, frames: list[ReferenceFrame], switch_period: int = 40, jitter: int = 10,
                 noise_sigma: float = 0.05, seed: int = 0):
        if len(frames) < 2:
            raise ValueError("need at least two candidate frames")
        self.frames = frames
        self.switch_period = switch_period
        self.jitter = jitter
        self.noise_sigma = noise_sigma
        self.rng = random.Random(seed)
        self._active_index = self.rng.randrange(len(frames))
        self._steps_until_switch = self._sample_period()
        self._mu: float | None = None
        self._pending: StepInfo | None = None

    def _sample_period(self) -> int:
        return max(1, self.switch_period + self.rng.randint(-self.jitter, self.jitter))

    def observe(self) -> StepInfo:
        """Advance to the next entity (possibly switching the active frame first)."""
        switched = False
        self._steps_until_switch -= 1
        if self._steps_until_switch <= 0:
            new_index = self.rng.randrange(len(self.frames))
            switched = new_index != self._active_index
            self._active_index = new_index
            self._steps_until_switch = self._sample_period()

        self._mu = self.rng.uniform(0.0, 1.0)
        observed = self._mu + self.rng.gauss(0.0, self.noise_sigma)
        self._pending = StepInfo(observed, self._active_index, switched)
        return self._pending

    def feedback(self, predict: int) -> int:
        """Submit the agent's prediction for the last observe() call; returns reward (0/1)."""
        assert self._mu is not None and self._pending is not None, "call observe() first"
        active_frame = self.frames[self._active_index]
        true_label = true_conclusion(self._mu, active_frame)
        reward = 1 if predict == true_label else 0
        self._pending.reward = reward
        return reward
