"""Multi-step planning environment for Experiment 12 (docs/research-agenda.md
#7i, Phase 6, continued).

Unlike `ResourceControlEnv` (Experiment 11 -- each trial's state resets
fresh, so a greedy 1-step lookahead policy is already globally optimal),
this environment gives each action a DELAYED, PARTIAL effect: only a
fraction `lag_weight` of a nudge's magnitude lands on the immediate next
position; the rest carries over as `pending` and lands on the position
after that. A greedy agent that only ever looks one step ahead can see
how an action affects the immediate next position, but is structurally
blind to how that same action's delayed component affects the position
after that.

Episodes persist for `horizon` steps -- the state is NOT redrawn fresh
every trial -- and reward is the negative squared distance to target at
the FINAL step only, making this a genuine multi-step credit-assignment
problem, not a sequence of independent single-step decisions.
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


class DelayedControlEnv:
    def __init__(self, target: float = 0.0, lag_weight: float = 0.5, horizon: int = 5,
                 init_range: tuple[float, float] = (-5.0, 5.0), noise_sigma: float = 0.1, seed: int = 0):
        self.target = target
        self.lag_weight = lag_weight  # fraction of a nudge landing THIS step; rest lands next step
        self.horizon = horizon
        self.init_range = init_range
        self.noise_sigma = noise_sigma
        self.rng = random.Random(seed)
        self.position = 0.0
        self.pending = 0.0
        self.step_count = 0

    def reset(self) -> tuple[float, float]:
        self.position = self.rng.uniform(*self.init_range)
        self.pending = 0.0
        self.step_count = 0
        return self.position, self.pending

    def step(self, action: str) -> StepResult:
        nudge = NUDGES[action]
        next_position = (self.position + self.lag_weight * nudge
                          + (1 - self.lag_weight) * self.pending + self.rng.gauss(0.0, self.noise_sigma))
        next_pending = nudge
        self.step_count += 1
        done = self.step_count >= self.horizon
        reward = -((next_position - self.target) ** 2) if done else 0.0
        result = StepResult(position=self.position, pending=self.pending, action=action,
                             next_position=next_position, next_pending=next_pending,
                             reward=reward, done=done)
        self.position, self.pending = next_position, next_pending
        return result

    def true_next_position(self, position: float, pending: float, action: str) -> float:
        """Exact expected next position under the known true dynamics --
        used only by the oracle conditions and to define the environment
        itself, never by a learning agent."""
        nudge = NUDGES[action]
        return position + self.lag_weight * nudge + (1 - self.lag_weight) * pending
