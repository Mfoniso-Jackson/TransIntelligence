"""Experiment 12 -- does multi-step planning (receding-horizon control,
Richalet, Rault, Testud, Papon, *Model Predictive Heuristic Control:
Applications to Industrial Processes*, Automatica 14(5), 1978) beat
greedy 1-step lookahead (experiment 11's `world_model` agent) when
actions have a DELAYED effect that a 1-step model is structurally blind
to? (docs/research-agenda.md #7i, Phase 6, continued)

Experiment 11's environment (`ResourceControlEnv`) has pure translation
dynamics with no delayed effects -- for that environment, greedy 1-step
lookahead already IS the globally optimal policy, so there was nothing
for multi-step planning to improve on. `DelayedControlEnv`
(`environments/transworld/delayed_control_env.py`) is genuinely
different: each nudge splits between an immediate effect (fraction
`lag_weight`) and a delayed one (the rest, landing on the position AFTER
next), and episodes now persist for `HORIZON` steps with reward only at
the final step -- a real multi-step credit-assignment problem, not a
sequence of independent single-step decisions.

## The confound this needed to control for

A 2x2 factorial design isolates "does planning horizon matter" from
"does model quality matter," rather than conflating them into one
number:

- **planning horizon**: `greedy` (1-step lookahead, capped at the
  episode's remaining steps) vs. `mpc` (`LOOKAHEAD`-step receding-horizon
  lookahead, replanned every step, same cap).
- **model source**: `learned` (fits `next_position ~ intercept(a) +
  b_pos(a)*position + b_pend(a)*pending` per action via
  `ordinary_least_squares`, refit after every episode) vs. `oracle`
  (given the true dynamics exactly, the same `true_oracle`/
  `oracle_dynamics` pattern experiments 1 and 11 used).

`greedy_learned` and `mpc_learned` use the IDENTICAL learned dynamics
model at any given episode -- the only difference is how many steps
ahead each simulates before acting. This isolates the planning-horizon
question cleanly: if `mpc_learned` beats `greedy_learned` by roughly the
same margin `mpc_oracle` beats `greedy_oracle`, the advantage is really
about horizon, not about the learned model happening to be better in one
condition than another.

Run: PYTHONPATH=. python experiments/exp12_multistep_planning/run.py
"""
from __future__ import annotations

import itertools
import random
import statistics

from environments.transworld import DelayedControlEnv, NUDGES
from transintelligence.reasoning.causal import ordinary_least_squares

ACTIONS = list(NUDGES.keys())
TARGET = 0.0
LAG_WEIGHT = 0.5
HORIZON = 5
INIT_RANGE = (-5.0, 5.0)
NOISE_SIGMA = 0.1
LOOKAHEAD = 2  # sufficient to see a nudge's full effect: immediate (this step) + delayed (next step)
WARMUP_EPISODES = 20
N_EPISODES = 300
SEEDS = list(range(15))


class LearnedDynamics:
    """Fits next_position ~ intercept(a) + b_pos(a)*position +
    b_pend(a)*pending per action, via ordinary_least_squares -- refit
    once per episode (cheap at this scale, and matches "periodic, not
    per-step, refitting" from experiment 11)."""

    def __init__(self) -> None:
        self.transitions: list[tuple[float, float, str, float]] = []  # (position, pending, action, next_position)
        self.coefficients: dict[str, tuple[float, float, float]] = {}

    def observe(self, position: float, pending: float, action: str, next_position: float) -> None:
        self.transitions.append((position, pending, action, next_position))

    def refit(self) -> None:
        by_action: dict[str, list[tuple[float, float, float]]] = {}
        for p, pend, a, np_ in self.transitions:
            by_action.setdefault(a, []).append((p, pend, np_))
        coefficients: dict[str, tuple[float, float, float]] = {}
        for a, rows in by_action.items():
            if len(rows) < 3:
                continue
            features = [[1.0, p, pend] for p, pend, _ in rows]
            targets = [np_ for _, _, np_ in rows]
            try:
                intercept, b_pos, b_pend = ordinary_least_squares(features, targets)
            except ValueError:
                continue
            coefficients[a] = (intercept, b_pos, b_pend)
        self.coefficients = coefficients

    def known_actions(self) -> set[str]:
        return set(self.coefficients.keys())

    def predict(self, position: float, pending: float, action: str) -> float:
        intercept, b_pos, b_pend = self.coefficients[action]
        return intercept + b_pos * position + b_pend * pending


def choose_action(position: float, pending: float, predict_fn, known_actions: set[str],
                   lookahead: int, remaining_steps: int, rng: random.Random) -> str:
    """Receding-horizon action selection: enumerate all action sequences
    of depth `min(lookahead, remaining_steps)`, simulate each via
    `predict_fn` chained forward (pending's own evolution is always
    exactly known -- it's just the last action taken, deterministic by
    construction, no learning needed for that part), score by predicted
    final-position squared distance to target, execute only the first
    action of the best sequence. `lookahead=1` reduces to pure greedy
    1-step lookahead (experiment 11's `world_model` policy)."""
    known = [a for a in ACTIONS if a in known_actions]
    if not known:
        return rng.choice(ACTIONS)
    depth = max(1, min(lookahead, remaining_steps))
    best_action, best_score = None, None
    for sequence in itertools.product(known, repeat=depth):
        p, pend = position, pending
        for a in sequence:
            p = predict_fn(p, pend, a)
            pend = NUDGES[a]
        score = (p - TARGET) ** 2
        if best_score is None or score < best_score:
            best_score = score
            best_action = sequence[0]
    return best_action


def run_episode(env: DelayedControlEnv, dynamics: LearnedDynamics | None, lookahead: int,
                 episode_idx: int, rng: random.Random) -> float:
    is_oracle = dynamics is None
    position, pending = env.reset()
    remaining = env.horizon
    for _ in range(env.horizon):
        if is_oracle:
            predict_fn = env.true_next_position
            known_actions = set(ACTIONS)
        else:
            predict_fn = dynamics.predict
            known_actions = dynamics.known_actions()
        if episode_idx < WARMUP_EPISODES:
            action = rng.choice(ACTIONS)
        else:
            action = choose_action(position, pending, predict_fn, known_actions, lookahead, remaining, rng)
        result = env.step(action)
        if not is_oracle:
            dynamics.observe(result.position, result.pending, result.action, result.next_position)
        position, pending = result.next_position, result.next_pending
        remaining -= 1
        if result.done:
            return result.reward
    raise AssertionError("episode did not terminate")  # pragma: no cover


def run_condition(seed: int, lookahead: int, is_oracle: bool) -> float:
    env = DelayedControlEnv(target=TARGET, lag_weight=LAG_WEIGHT, horizon=HORIZON,
                             init_range=INIT_RANGE, noise_sigma=NOISE_SIGMA, seed=seed)
    rng = random.Random(seed + 20_000)
    dynamics = None if is_oracle else LearnedDynamics()
    rewards = []
    for episode_idx in range(N_EPISODES):
        reward = run_episode(env, dynamics, lookahead, episode_idx, rng)
        if not is_oracle:
            dynamics.refit()
        if episode_idx >= WARMUP_EPISODES:
            rewards.append(reward)
    return statistics.mean(rewards)


CONDITIONS = {
    "greedy_learned": (1, False),
    "greedy_oracle": (1, True),
    "mpc_learned": (LOOKAHEAD, False),
    "mpc_oracle": (LOOKAHEAD, True),
}


def main() -> None:
    print(f"{'condition':<18} {'mean_final_reward':>18}")
    for label, (lookahead, is_oracle) in CONDITIONS.items():
        rewards = [run_condition(seed, lookahead, is_oracle) for seed in SEEDS]
        print(f"{label:<18} {statistics.mean(rewards):>18.4f}")


if __name__ == "__main__":
    main()
