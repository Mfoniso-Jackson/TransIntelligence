"""Combines `DelayedControlEnv`'s (experiment 12) persistent multi-step
episodes with `RegimeShiftControlEnv`'s (experiment 13) silent
mid-experiment dynamics shift, for experiment 16 (docs/research-agenda.md
#7m, Phase 6, continued): does regime-change detection compose with
multi-step planning, or does testing them separately (experiments 12/14
for planning, 13 for detection) miss something about how they interact?

`next_position = position + lag_weight*scale(t)*nudge(action) +
(1-lag_weight)*pending + noise`, `next_pending = scale(t)*nudge(action)`,
where `scale(t)` is 1.0 before `regime_shift_step` and `post_shift_scale`
afterward -- `regime_shift_step` is a cumulative step count across the
WHOLE run (all episodes), not reset per episode, so the shift can land
at any point within an episode, exactly like experiment 13's trial-based
shift. `post_shift_scale=-1.0` (the actuator's effect reverses direction
entirely) is experiment 13's dramatic "sign_flip" severity, reused here
rather than re-deriving a new severity from scratch.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .resource_control_env import NUDGES


@dataclass
class StepResult:
    position: float
    pending: float
    action: str
    next_position: float
    next_pending: float
    reward: float
    done: bool


class DelayedRegimeShiftControlEnv:
    def __init__(self, target: float = 0.0, lag_weight: float = 0.5, horizon: int = 5,
                 init_range: tuple[float, float] = (-5.0, 5.0), noise_sigma: float = 0.1,
                 regime_shift_step: int = 1000, post_shift_scale: float = -1.0, seed: int = 0):
        self.target = target
        self.lag_weight = lag_weight
        self.horizon = horizon
        self.init_range = init_range
        self.noise_sigma = noise_sigma
        self.regime_shift_step = regime_shift_step
        self.post_shift_scale = post_shift_scale
        self.rng = random.Random(seed)
        self.position = 0.0
        self.pending = 0.0
        self.episode_step = 0
        self.global_step = 0

    def _scale_at(self, global_step: int) -> float:
        return 1.0 if global_step < self.regime_shift_step else self.post_shift_scale

    def reset(self) -> tuple[float, float]:
        self.position = self.rng.uniform(*self.init_range)
        self.pending = 0.0
        self.episode_step = 0
        return self.position, self.pending

    def step(self, action: str) -> StepResult:
        scale = self._scale_at(self.global_step)
        nudge = NUDGES[action]
        next_position = (self.position + self.lag_weight * scale * nudge
                          + (1 - self.lag_weight) * self.pending + self.rng.gauss(0.0, self.noise_sigma))
        next_pending = scale * nudge
        self.episode_step += 1
        self.global_step += 1
        done = self.episode_step >= self.horizon
        reward = -((next_position - self.target) ** 2) if done else 0.0
        result = StepResult(position=self.position, pending=self.pending, action=action,
                             next_position=next_position, next_pending=next_pending,
                             reward=reward, done=done)
        self.position, self.pending = next_position, next_pending
        return result

    def true_next_position(self, position: float, pending: float, action: str, global_step: int) -> float:
        """Exact expected next position at a given global step -- an
        explicit `global_step` argument (not read from internal state)
        so callers can compute this without depending on timing relative
        to `step()`'s own counter, the same design already used in
        `RegimeShiftControlEnv.true_expected_reward`."""
        scale = self._scale_at(global_step)
        nudge = NUDGES[action]
        return position + self.lag_weight * scale * nudge + (1 - self.lag_weight) * pending
