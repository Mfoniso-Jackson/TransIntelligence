"""Experiment 15 -- does the "learned dynamics are cheap to learn
exactly" advantage experiments 11-14 all relied on survive when the true
dynamics are genuinely nonlinear, and does a correctly-specified
nonlinear dynamics model recover it? (docs/research-agenda.md #7l,
Phase 6, continued)

`NonlinearControlEnv` (`environments/transworld/nonlinear_control_env.py`)
adds a quadratic restoring force to `ResourceControlEnv`'s dynamics:
`next_state = state + nudge(action) - GAMMA*state*abs(state) + noise` --
the same structural role experiment 10's `gamma2*X**2` term played for a
causal effect, now for a state's own evolution. `abs(state)` keeps the
force's sign opposed to the state's own sign everywhere (a genuine
restoring force, not a one-sided drift) -- deliberately NOT a bare
square, so a naively-squared feature (`state**2`, blind to sign) would
still be a genuine misspecification, not an accidental match.

## The confound this needed to control for

Showing a nonlinear world model beats a linear one only proves nonlinear
features can help *somewhere* -- not that they capture the *actual*
functional form. `nonlinear_world_model` uses `state*abs(state)` as its
quadratic feature, matching the true dynamics' actual shape exactly (an
odd, sign-aware function, not `state**2`'s even one). This is checked
directly against the true dynamics, and the comparison is fair in the
same sense experiment 11's quadratic follow-up was: identical state
access, identical `ordinary_least_squares` tool, differing only in
which features are fit.

## A boundary condition this experiment's first run found

The initial `GAMMA=0.05` (the term's magnitude at the state range's
edge, ~1.25, comparable to a single mid-sized nudge) showed almost no
gap between `linear_world_model` and `nonlinear_world_model` -- the
same pattern experiment 13 found for a mild regime shift: a nonlinearity
too small to change *which discrete action ranks best* doesn't produce a
measurable gap, even though the underlying model is still technically
misspecified. Swept `GAMMA` up rather than reporting a null result
without checking whether a stronger version of the same claim held:
`GAMMA=0.3` produces a clean, dramatic gap (linear ~5.6x worse than
nonlinear/oracle); `GAMMA=0.3` is the value used below. Both findings are
reported in RESULTS.md.

Four conditions, matching experiment 11's structure:

1. `state_blind` -- ignores state, tracks per-action average reward.
2. `linear_world_model` -- reuses `LinearDynamicsModel`
   (`transintelligence/world_models/`) unchanged, from experiment 11 --
   deliberately misspecified here, to see how it performs when its
   linearity assumption is genuinely wrong.
3. `nonlinear_world_model` -- fits `next_state ~ intercept(a) +
   b1(a)*state + b2(a)*state*abs(state)` per action, matching the true
   functional form.
4. `oracle_dynamics` -- given the true dynamics exactly, isolating the
   cost of learning (the `true_oracle`/`oracle_dynamics` pattern
   experiments 1, 11, and 13 all used).

Run: PYTHONPATH=. python experiments/exp15_nonlinear_world_model/run.py
"""
from __future__ import annotations

import random
import statistics

from environments.transworld import NUDGES, NonlinearControlEnv
from transintelligence.reasoning.causal import ordinary_least_squares
from transintelligence.world_models import LinearDynamicsModel

ACTIONS = list(NUDGES.keys())
TARGET = 0.0
STATE_RANGE = (-5.0, 5.0)
NOISE_SIGMA = 0.3
GAMMA = 0.3  # small values (e.g. 0.05) don't change which discrete action is optimal -- see RESULTS.md
WARMUP_TRIALS = 60
REFIT_INTERVAL = 20
N_TRIALS = 2000
SEEDS = list(range(20))


class StateBlindAgent:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {a: 0.0 for a in ACTIONS}
        self.counts: dict[str, int] = {a: 0 for a in ACTIONS}

    def choose_action(self, state: float, rng: random.Random) -> str:
        known = [a for a in ACTIONS if self.counts[a] > 0]
        if not known:
            return rng.choice(ACTIONS)
        return max(known, key=lambda a: self.sums[a] / self.counts[a])

    def observe(self, state: float, action: str, next_state: float, reward: float) -> None:
        self.sums[action] += reward
        self.counts[action] += 1


class LinearWorldModelAgent:
    """Reuses `LinearDynamicsModel` (experiment 11) unchanged -- its
    linearity assumption is genuinely wrong in this environment."""

    def __init__(self) -> None:
        self.transitions: list[tuple[float, str, float]] = []
        self.model = LinearDynamicsModel()

    def choose_action(self, state: float, rng: random.Random) -> str:
        known = self.model.known_actions()
        if not known:
            return rng.choice(ACTIONS)
        return min(known, key=lambda a: (self.model.predict(state, a) - TARGET) ** 2)

    def observe(self, state: float, action: str, next_state: float, reward: float) -> None:
        self.transitions.append((state, action, next_state))
        if len(self.transitions) % REFIT_INTERVAL == 0:
            self.model = LinearDynamicsModel.fit(self.transitions)


class NonlinearWorldModelAgent:
    """Fits next_state ~ intercept(a) + b1(a)*state + b2(a)*state*|state|
    per action, via ordinary_least_squares -- a function class that CAN
    represent the true restoring-force dynamics exactly."""

    def __init__(self) -> None:
        self.transitions: list[tuple[float, str, float]] = []
        self.coefficients: dict[str, tuple[float, float, float]] = {}

    def _refit(self) -> None:
        by_action: dict[str, list[tuple[float, float]]] = {}
        for s, a, ns in self.transitions:
            by_action.setdefault(a, []).append((s, ns))
        coefficients: dict[str, tuple[float, float, float]] = {}
        for a, pairs in by_action.items():
            if len(pairs) < 3:
                continue
            features = [[1.0, s, s * abs(s)] for s, _ in pairs]
            targets = [ns for _, ns in pairs]
            try:
                intercept, b1, b2 = ordinary_least_squares(features, targets)
            except ValueError:
                continue
            coefficients[a] = (intercept, b1, b2)
        self.coefficients = coefficients

    def _predict(self, state: float, action: str) -> float:
        intercept, b1, b2 = self.coefficients[action]
        return intercept + b1 * state + b2 * state * abs(state)

    def choose_action(self, state: float, rng: random.Random) -> str:
        known = list(self.coefficients.keys())
        if not known:
            return rng.choice(ACTIONS)
        return min(known, key=lambda a: (self._predict(state, a) - TARGET) ** 2)

    def observe(self, state: float, action: str, next_state: float, reward: float) -> None:
        self.transitions.append((state, action, next_state))
        if len(self.transitions) % REFIT_INTERVAL == 0:
            self._refit()


class OracleDynamicsAgent:
    def __init__(self, env: NonlinearControlEnv) -> None:
        self.env = env

    def choose_action(self, state: float, rng: random.Random) -> str:
        return self.env.optimal_action(state)

    def observe(self, state: float, action: str, next_state: float, reward: float) -> None:
        pass


CONDITIONS = {
    "state_blind": lambda env: StateBlindAgent(),
    "linear_world_model": lambda env: LinearWorldModelAgent(),
    "nonlinear_world_model": lambda env: NonlinearWorldModelAgent(),
    "oracle_dynamics": lambda env: OracleDynamicsAgent(env),
}


def run_condition(agent_factory, seed: int) -> tuple[float, float]:
    env = NonlinearControlEnv(target=TARGET, state_range=STATE_RANGE, noise_sigma=NOISE_SIGMA, gamma=GAMMA, seed=seed)
    rng = random.Random(seed + 40_000)
    agent = agent_factory(env)
    rewards, ceilings = [], []
    for trial in range(N_TRIALS):
        info = env.observe()
        if trial < WARMUP_TRIALS:
            action = rng.choice(ACTIONS)
        else:
            action = agent.choose_action(info.state, rng)
        info = env.step(action)
        agent.observe(info.state, action, info.next_state, info.reward)
        if trial >= WARMUP_TRIALS:
            rewards.append(info.reward)
            ceilings.append(env.true_expected_reward(info.state, env.optimal_action(info.state)))
    return statistics.mean(rewards), statistics.mean(ceilings)


def main() -> None:
    print(f"{'condition':<24} {'mean_reward':>13} {'mean_ceiling':>13} {'mean_regret':>13}")
    for label, factory in CONDITIONS.items():
        rewards, ceilings = [], []
        for seed in SEEDS:
            r, c = run_condition(factory, seed)
            rewards.append(r)
            ceilings.append(c)
        mean_reward = statistics.mean(rewards)
        mean_ceiling = statistics.mean(ceilings)
        mean_regret = mean_ceiling - mean_reward
        print(f"{label:<24} {mean_reward:>13.4f} {mean_ceiling:>13.4f} {mean_regret:>13.4f}")


if __name__ == "__main__":
    main()
