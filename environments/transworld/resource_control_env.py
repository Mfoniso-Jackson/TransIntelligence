"""Minimal synthetic environment for Experiment 11 (docs/research-agenda.md
#7h, docs/master-context.md §13/§19, Phase 6).

At each trial, the environment draws a fresh current state `s` (uniform
over a fixed range) and shows it to the agent. The agent picks one of a
small set of discrete "nudge" actions. The environment computes
`next_state = s + nudge(action) + noise`, and `reward = -(next_state -
target)**2`. Unlike `FrameSwitchEnv` (Experiment 1's environment, a
stateless bandit where the agent's choice never affects a persistent
world state), here the transition genuinely depends on the chosen
action -- `state(t), action(t) -> state(t+1)`, the master context's own
formalization of what a world model is for.

Each trial's state is drawn fresh rather than chaining into a multi-step
rollout -- a deliberate scope limit stated up front: this environment
tests whether a learned one-step dynamics model helps *single-step*
action selection, not multi-step planning (`Simulator`/`Planner` remain
unbuilt, the natural next Phase 6/7 gap).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

NUDGES: dict[str, float] = {
    "large_down": -2.0,
    "small_down": -1.0,
    "tiny_down": -0.3,
    "tiny_up": 0.3,
    "small_up": 1.0,
    "large_up": 2.0,
}


@dataclass
class TrialInfo:
    state: float
    next_state: float | None = None
    reward: float | None = None


class ResourceControlEnv:
    def __init__(self, target: float = 0.0, state_range: tuple[float, float] = (-5.0, 5.0),
                 noise_sigma: float = 0.3, seed: int = 0):
        self.target = target
        self.state_range = state_range
        self.noise_sigma = noise_sigma
        self.rng = random.Random(seed)
        self._pending: TrialInfo | None = None

    def observe(self) -> TrialInfo:
        """Draw a fresh current state for the next trial."""
        state = self.rng.uniform(*self.state_range)
        self._pending = TrialInfo(state=state)
        return self._pending

    def step(self, action: str) -> TrialInfo:
        """Apply `action` to the pending state; returns the completed
        TrialInfo (state, next_state, reward)."""
        assert self._pending is not None, "call observe() first"
        nudge = NUDGES[action]
        next_state = self._pending.state + nudge + self.rng.gauss(0.0, self.noise_sigma)
        reward = -((next_state - self.target) ** 2)
        self._pending.next_state = next_state
        self._pending.reward = reward
        return self._pending

    def true_expected_reward(self, state: float, action: str) -> float:
        """E[reward | state, action], computed from the known true
        dynamics -- exact because reward = -(state + nudge + noise -
        target)**2 and E[(X + noise)**2] = X**2 + Var(noise) for
        zero-mean noise. Used only to compute the oracle policy and the
        theoretical performance ceiling, never by any learning agent."""
        nudge = NUDGES[action]
        return -((state + nudge - self.target) ** 2) - self.noise_sigma ** 2

    def optimal_action(self, state: float) -> str:
        return max(NUDGES, key=lambda a: self.true_expected_reward(state, a))
