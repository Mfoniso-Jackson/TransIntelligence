"""Follow-up to Experiment 11: RESULTS.md's own "what this does not
establish" flagged that a *nonlinear* model-free baseline (one that CAN
represent the true reward surface) was untested and might close the gap
with `world_model` -- this checks that directly, rather than leaving it
as an unverified caveat. (docs/research-agenda.md #7h)

`model_free_linear_q` lost to `world_model` specifically because the
true reward is a downward parabola (quadratic) in state, and a linear
fit cannot represent a peak. `ModelFreeQuadraticQAgent` fits reward as a
QUADRATIC function of state per action (`reward ~ intercept(a) +
b1(a)*state + b2(a)*state**2`, via the same `ordinary_least_squares`,
just with an extra feature column) -- a function class that CAN
represent the true parabola exactly.

The prediction this follow-up tests: if the quadratic baseline closes
the gap with `world_model`, that confirms the mechanism identified in
the main experiment (linear-cannot-represent-a-peak) was the actual
cause of `model_free_linear_q`'s shortfall, not some other unaccounted-
for difference between the two conditions. If it does NOT close the
gap, that would mean something else was going on and the original
explanation needs revisiting.

Run: PYTHONPATH=. python experiments/exp11_world_model_planning/quadratic_baseline.py
"""
from __future__ import annotations

import random
import statistics

from transintelligence.reasoning.causal import ordinary_least_squares

from experiments.exp11_world_model_planning.run import (
    ACTIONS,
    CONDITIONS,
    REFIT_INTERVAL,
    SEEDS,
    run_condition,
)


class ModelFreeQuadraticQAgent:
    """Fits reward as a QUADRATIC function of state, per action -- a
    function class that CAN represent the true (parabolic) reward
    surface exactly, unlike `model_free_linear_q`."""

    def __init__(self) -> None:
        self.transitions: list[tuple[float, str, float]] = []  # (state, action, reward)
        self.coefficients: dict[str, tuple[float, float, float]] = {}

    def _refit(self) -> None:
        by_action: dict[str, list[tuple[float, float]]] = {}
        for s, a, r in self.transitions:
            by_action.setdefault(a, []).append((s, r))
        coefficients: dict[str, tuple[float, float, float]] = {}
        for a, pairs in by_action.items():
            if len(pairs) < 3:
                continue
            features = [[1.0, s, s * s] for s, _ in pairs]
            targets = [r for _, r in pairs]
            try:
                intercept, b1, b2 = ordinary_least_squares(features, targets)
            except ValueError:
                continue
            coefficients[a] = (intercept, b1, b2)
        self.coefficients = coefficients

    def choose_action(self, state: float, rng: random.Random) -> str:
        known = list(self.coefficients.keys())
        if not known:
            return rng.choice(ACTIONS)
        return max(known, key=lambda a: self.coefficients[a][0] + self.coefficients[a][1] * state + self.coefficients[a][2] * state * state)

    def observe(self, state: float, action: str, next_state: float, reward: float) -> None:
        self.transitions.append((state, action, reward))
        if len(self.transitions) % REFIT_INTERVAL == 0:
            self._refit()


ALL_CONDITIONS = dict(CONDITIONS)
ALL_CONDITIONS["model_free_quadratic_q"] = lambda env: ModelFreeQuadraticQAgent()


def main() -> None:
    print(f"{'condition':<22} {'mean_reward':>13} {'mean_ceiling':>13} {'mean_regret':>13}")
    for label, factory in ALL_CONDITIONS.items():
        rewards, ceilings = [], []
        for seed in SEEDS:
            r, c = run_condition(factory, seed)
            rewards.append(r)
            ceilings.append(c)
        mean_reward = statistics.mean(rewards)
        mean_ceiling = statistics.mean(ceilings)
        mean_regret = mean_ceiling - mean_reward
        print(f"{label:<22} {mean_reward:>13.4f} {mean_ceiling:>13.4f} {mean_regret:>13.4f}")


if __name__ == "__main__":
    main()
