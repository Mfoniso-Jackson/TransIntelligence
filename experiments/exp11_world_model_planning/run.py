"""Experiment 11 -- does an agent that learns explicit forward dynamics
and plans by simulating candidate actions beat a model-free baseline
with the same state access but no dynamics model? (docs/research-agenda.md
#7h, Phase 6, docs/master-context.md §13/§19)

Environment: `environments/transworld/resource_control_env.py`'s
`ResourceControlEnv` -- at each trial, a fresh state `s` is drawn, the
agent picks a discrete "nudge" action, and `reward = -(s + nudge(action) +
noise - target)**2`. The true dynamics are LINEAR in state given the
action (`next_state = s + nudge(action) + noise`), but the reward is a
QUADRATIC function of the resulting state -- this is the specific
structural feature that makes decomposing "learn the (easy, linear)
dynamics, then apply the (known, exact) reward formula" different from
"directly fit the (harder, quadratic) reward as a function of state" with
the same linear-regression tool.

## The confound this needed to control for

Showing a "state-aware" agent beat a "state-blind" one would only
demonstrate that using context helps at all -- trivially true, and not
the master context's actual claim (`M(S_t, A_t) -> S_{t+1}`, a
*transition* model, useful because predicted outcomes of different
actions can be compared). Two state-aware conditions isolate the real
claim instead:

- `model_free_linear_q`: fits reward directly as a linear function of
  state per action (`reward ~ intercept(a) + slope(a)*state`, via the
  same `ordinary_least_squares` the world-model agent uses) -- a
  standard model-free value-function approximator (Sutton & Barto's
  linear function approximation). It sees the same state, uses the same
  regression tool, and gets the same warmup data as the world-model
  agent. It just never models the *transition* -- it goes straight from
  (state, action) to a value estimate.
- `world_model`: fits the transition `next_state ~ intercept(a) +
  slope(a)*state` per action (`LinearDynamicsModel`,
  `transintelligence/world_models/model.py`), then applies the KNOWN
  reward formula `-(predicted_next_state - target)**2` to score each
  candidate action.

Both conditions are equally "state-aware" and use an equally linear
model -- the only difference is *what* they model. Because the true
reward is quadratic in state while the true dynamics are linear, a
correctly-specified linear model of the dynamics is exact, while a
linear model of the reward is a genuine misspecification of a quadratic
function -- directly reusing experiment 10's finding (a linear model
cannot represent a relationship that depends on where a unit starts) in
a planning setting instead of an effect-estimation one.

Four conditions total:

1. `state_blind` -- tracks each action's running average reward,
   ignores current state entirely (a floor, not a serious baseline).
2. `model_free_linear_q` -- described above.
3. `world_model` -- described above.
4. `oracle_dynamics` -- given the true dynamics exactly (no estimation),
   plans the same way `world_model` does. Isolates "cost of learning the
   dynamics" the same way experiment 1's `true_oracle` did.

All four agents act uniformly at random for the first `WARMUP_TRIALS`
trials (shared across conditions, same environment seed and warmup
random-choice sequence, so the comparison starts from identical data) to
bootstrap the two learned models; reported metrics are computed only over
trials after warmup.

Run: PYTHONPATH=. python experiments/exp11_world_model_planning/run.py
"""
from __future__ import annotations

import random
import statistics

from environments.transworld import NUDGES, ResourceControlEnv
from transintelligence.reasoning.causal import ordinary_least_squares
from transintelligence.world_models import LinearDynamicsModel

ACTIONS = list(NUDGES.keys())
TARGET = 0.0
STATE_RANGE = (-5.0, 5.0)
NOISE_SIGMA = 0.3
WARMUP_TRIALS = 60
REFIT_INTERVAL = 20  # refit the learned models periodically, not every trial -- both realistic and tractable
N_TRIALS = 2000
SEEDS = list(range(20))


class StateBlindAgent:
    """Tracks each action's running average reward; ignores current
    state entirely -- the floor, not a serious model-free baseline."""

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


class ModelFreeLinearQAgent:
    """Fits reward directly as a linear function of state, per action --
    a state-aware value-function approximator with no transition model.
    Refits every `REFIT_INTERVAL` trials rather than every single trial:
    both more tractable (refitting from scratch on all accumulated data
    every trial is O(n_trials^2) work) and more realistic -- real
    model-based/model-free agents periodically retrain, they don't
    literally refit after every single environment step."""

    def __init__(self) -> None:
        self.transitions: list[tuple[float, str, float]] = []  # (state, action, reward)
        self.coefficients: dict[str, tuple[float, float]] = {}

    def _refit(self) -> None:
        by_action: dict[str, list[tuple[float, float]]] = {}
        for s, a, r in self.transitions:
            by_action.setdefault(a, []).append((s, r))
        coefficients: dict[str, tuple[float, float]] = {}
        for a, pairs in by_action.items():
            if len(pairs) < 2:
                continue
            features = [[1.0, s] for s, _ in pairs]
            targets = [r for _, r in pairs]
            try:
                intercept, slope = ordinary_least_squares(features, targets)
            except ValueError:
                continue
            coefficients[a] = (intercept, slope)
        self.coefficients = coefficients

    def choose_action(self, state: float, rng: random.Random) -> str:
        known = list(self.coefficients.keys())
        if not known:
            return rng.choice(ACTIONS)
        return max(known, key=lambda a: self.coefficients[a][0] + self.coefficients[a][1] * state)

    def observe(self, state: float, action: str, next_state: float, reward: float) -> None:
        self.transitions.append((state, action, reward))
        if len(self.transitions) % REFIT_INTERVAL == 0:
            self._refit()


class WorldModelAgent:
    """Fits the state TRANSITION as a linear function of state, per
    action, then applies the known reward formula to score candidates.
    Refits every `REFIT_INTERVAL` trials, same reasoning as
    `ModelFreeLinearQAgent`."""

    def __init__(self) -> None:
        self.transitions: list[tuple[float, str, float]] = []  # (state, action, next_state)
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


class OracleDynamicsAgent:
    """Given the true dynamics exactly -- isolates the cost of LEARNING
    the dynamics, the way experiment 1's true_oracle did."""

    def __init__(self, env: ResourceControlEnv) -> None:
        self.env = env

    def choose_action(self, state: float, rng: random.Random) -> str:
        return self.env.optimal_action(state)

    def observe(self, state: float, action: str, next_state: float, reward: float) -> None:
        pass


CONDITIONS = {
    "state_blind": lambda env: StateBlindAgent(),
    "model_free_linear_q": lambda env: ModelFreeLinearQAgent(),
    "world_model": lambda env: WorldModelAgent(),
    "oracle_dynamics": lambda env: OracleDynamicsAgent(env),
}


def run_condition(agent_factory, seed: int) -> tuple[float, float]:
    env = ResourceControlEnv(target=TARGET, state_range=STATE_RANGE, noise_sigma=NOISE_SIGMA, seed=seed)
    rng = random.Random(seed + 10_000)
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
    print(f"{'condition':<22} {'mean_reward':>13} {'mean_ceiling':>13} {'mean_regret':>13}")
    for label, factory in CONDITIONS.items():
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
