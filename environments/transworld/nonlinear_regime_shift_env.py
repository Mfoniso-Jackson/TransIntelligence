"""Combines `NonlinearControlEnv`'s (experiment 15) quadratic restoring
force with `RegimeShiftControlEnv`'s (experiment 13) silent actuator
rescaling, for experiment 19 (docs/research-agenda.md #7p, Phase 6,
continued): does regime-change detection compose with a nonlinear
dynamics model, the way experiment 13 showed it composes with a linear
one? Experiment 16 already tested "regime detection + multi-step
planning"; this is the other combination Phase 6's remaining-gaps list
named -- "regime detection + nonlinear dynamics" -- still untested.

`next_state = state + scale(t)*nudge(action) - GAMMA*state*abs(state) +
noise`, where `scale(t)` is 1.0 before `regime_shift_trial` and
`post_shift_scale` afterward -- the actuator recalibrates, exactly as in
experiment 13, but now on top of the same restoring-force nonlinearity
experiment 15 introduced. `GAMMA` itself does not shift -- only the
actuator's scale does, isolating the regime-shift mechanism from the
nonlinearity rather than shifting both at once. Otherwise identical in
structure to both parent environments: fresh state per trial, single-
step decisions (unlike experiment 16's persistent multi-step episodes).
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


class NonlinearRegimeShiftControlEnv:
    def __init__(self, target: float = 0.0, state_range: tuple[float, float] = (-5.0, 5.0),
                 noise_sigma: float = 0.3, gamma: float = 0.3,
                 regime_shift_trial: int = 1500, post_shift_scale: float = 0.4, seed: int = 0):
        self.target = target
        self.state_range = state_range
        self.noise_sigma = noise_sigma
        self.gamma = gamma
        self.regime_shift_trial = regime_shift_trial
        self.post_shift_scale = post_shift_scale
        self.rng = random.Random(seed)
        self._trial = 0
        self._pending: TrialInfo | None = None

    def _scale_at(self, trial: int) -> float:
        return 1.0 if trial < self.regime_shift_trial else self.post_shift_scale

    def _restoring_force(self, state: float) -> float:
        return -self.gamma * state * abs(state)

    def observe(self) -> TrialInfo:
        state = self.rng.uniform(*self.state_range)
        self._pending = TrialInfo(trial=self._trial, state=state)
        return self._pending

    def step(self, action: str) -> TrialInfo:
        assert self._pending is not None, "call observe() first"
        scale = self._scale_at(self._trial)
        nudge = NUDGES[action]
        next_state = (self._pending.state + scale * nudge + self._restoring_force(self._pending.state)
                       + self.rng.gauss(0.0, self.noise_sigma))
        reward = -((next_state - self.target) ** 2)
        self._pending.next_state = next_state
        self._pending.reward = reward
        self._trial += 1
        return self._pending

    def true_expected_reward(self, trial: int, state: float, action: str) -> float:
        """E[reward | trial, state, action] under the known true dynamics
        AT THAT TRIAL -- explicit `trial` argument, matching
        `RegimeShiftControlEnv`'s same design, so callers can compute
        this without depending on timing relative to `step()`'s own
        counter. Used only for oracle policies and the ceiling."""
        scale = self._scale_at(trial)
        nudge = NUDGES[action]
        predicted = state + scale * nudge + self._restoring_force(state)
        return -((predicted - self.target) ** 2) - self.noise_sigma ** 2

    def optimal_action(self, trial: int, state: float) -> str:
        return max(NUDGES, key=lambda a: self.true_expected_reward(trial, state, a))
