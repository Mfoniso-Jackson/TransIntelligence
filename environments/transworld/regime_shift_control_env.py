"""Non-stationary version of `ResourceControlEnv` (Experiment 11) for
Experiment 13 (docs/research-agenda.md #7j, Phase 6, continued): the
nudge magnitudes silently rescale at an unknown trial -- "the actuator
recalibrates" -- so a dynamics model fit on pre-shift data becomes
systematically wrong afterward, without the agent being told this
happened. Otherwise identical in structure to `ResourceControlEnv`
(fresh state per trial, single-step decisions).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .resource_control_env import NUDGES


@dataclass
class TrialInfo:
    trial: int
    state: float
    next_state: float | None = None
    reward: float | None = None


class RegimeShiftControlEnv:
    def __init__(self, target: float = 0.0, state_range: tuple[float, float] = (-5.0, 5.0),
                 noise_sigma: float = 0.3, regime_shift_trial: int = 1500,
                 post_shift_scale: float = 0.4, seed: int = 0):
        self.target = target
        self.state_range = state_range
        self.noise_sigma = noise_sigma
        self.regime_shift_trial = regime_shift_trial
        self.post_shift_scale = post_shift_scale
        self.rng = random.Random(seed)
        self._trial = 0
        self._pending: TrialInfo | None = None

    def _scale_at(self, trial: int) -> float:
        return 1.0 if trial < self.regime_shift_trial else self.post_shift_scale

    def observe(self) -> TrialInfo:
        state = self.rng.uniform(*self.state_range)
        self._pending = TrialInfo(trial=self._trial, state=state)
        return self._pending

    def step(self, action: str) -> TrialInfo:
        assert self._pending is not None, "call observe() first"
        scale = self._scale_at(self._trial)
        nudge = NUDGES[action]
        next_state = self._pending.state + scale * nudge + self.rng.gauss(0.0, self.noise_sigma)
        reward = -((next_state - self.target) ** 2)
        self._pending.next_state = next_state
        self._pending.reward = reward
        self._trial += 1
        return self._pending

    def true_expected_reward(self, trial: int, state: float, action: str) -> float:
        """E[reward | trial, state, action] under the known true
        dynamics AT THAT TRIAL -- explicit `trial` argument rather than
        reading internal state, so callers can compute this for a past
        trial without depending on timing relative to `step()`'s own
        counter. Used only for oracle policies and the performance
        ceiling, never by a learning agent."""
        scale = self._scale_at(trial)
        nudge = NUDGES[action]
        return -((state + scale * nudge - self.target) ** 2) - self.noise_sigma ** 2

    def optimal_action(self, trial: int, state: float) -> str:
        return max(NUDGES, key=lambda a: self.true_expected_reward(trial, state, a))
