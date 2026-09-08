"""Nonlinear-dynamics version of `ResourceControlEnv` (Experiment 11)
for Experiment 15 (docs/research-agenda.md #7l, Phase 6, continued).

Every world model built so far in this program (`LinearDynamicsModel`,
experiments 11-14) assumes the true transition is linear in state given
the action. This environment adds a genuinely nonlinear self-dynamics
term -- a quadratic restoring force, `-GAMMA * state * abs(state)`,
pulling the state back toward zero with a strength that grows with the
square of how far away it already is (the same structural role
experiment 10's `gamma2 * X**2` term played for a causal effect, now for
a state's own evolution instead of a treatment's effect on an outcome).
`abs(state)` (not a bare square) keeps the force's *sign* opposed to the
state's own sign at every point -- a real restoring force, not a
one-sided drift.

`next_state = state + nudge(action) - GAMMA * state * abs(state) + noise`
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


class NonlinearControlEnv:
    def __init__(self, target: float = 0.0, state_range: tuple[float, float] = (-5.0, 5.0),
                 noise_sigma: float = 0.3, gamma: float = 0.05, seed: int = 0):
        self.target = target
        self.state_range = state_range
        self.noise_sigma = noise_sigma
        self.gamma = gamma
        self.rng = random.Random(seed)
        self._trial = 0
        self._pending: TrialInfo | None = None

    def _restoring_force(self, state: float) -> float:
        return -self.gamma * state * abs(state)

    def observe(self) -> TrialInfo:
        state = self.rng.uniform(*self.state_range)
        self._pending = TrialInfo(trial=self._trial, state=state)
        return self._pending

    def step(self, action: str) -> TrialInfo:
        assert self._pending is not None, "call observe() first"
        nudge = NUDGES[action]
        next_state = (self._pending.state + nudge + self._restoring_force(self._pending.state)
                       + self.rng.gauss(0.0, self.noise_sigma))
        reward = -((next_state - self.target) ** 2)
        self._pending.next_state = next_state
        self._pending.reward = reward
        self._trial += 1
        return self._pending

    def true_expected_reward(self, state: float, action: str) -> float:
        """Exact E[reward | state, action] under the known true
        dynamics -- used only for the oracle policy and the ceiling,
        never by a learning agent."""
        nudge = NUDGES[action]
        predicted = state + nudge + self._restoring_force(state)
        return -((predicted - self.target) ** 2) - self.noise_sigma ** 2

    def optimal_action(self, state: float) -> str:
        return max(NUDGES, key=lambda a: self.true_expected_reward(state, a))
