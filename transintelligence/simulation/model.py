"""Baseline simulation: Monte Carlo rollout comparison between given
policies (Phase 6, docs/research-agenda.md #7n, docs/master-context.md
§19: `Simulator`).

`transintelligence/simulation/` was, before this, an empty stub
(`__init__.py` with a one-line placeholder docstring, no `model.py` at
all) -- the same starting point every other empty-stub reasoning
protocol had before its phase started. `Predictor` (`LinearDynamicsModel`)
answers "what's the next state." `Planner` (`RecedingHorizonPlanner`)
answers "which single action is best, searched for internally." Neither
answers the master context's own framing for this layer (§13: "S_t ->A_1->
S_t+1 can be compared against S_t ->A_2-> S'_t+1") as a standalone
capability: comparing the full multi-step outcome of two *given*
candidate policies under the environment's real stochasticity, rather
than deterministically optimizing for one internally. `Planner` also
takes a deterministic `transition_fn(state, action) -> next_state` and
returns one point estimate implicitly (the best-scoring sequence);
`MonteCarloSimulator` takes a stochastic `transition_fn(state, action,
rng) -> next_state` and returns the outcome *distribution* over many
independent rollouts, so a caller sees the spread, not just a mean.

Grounded in Sutton's Dyna architecture (*Integrated Architectures for
Learning, Planning, and Reacting Based on Approximating Dynamic
Programming*, ICML 1990; *Dyna, an Integrated Architecture for Learning,
Planning, and Reacting*, SIGART Bulletin 2(4), 1991) -- plan by
simulating candidate actions through a model rather than only acting in
the real environment -- generalized here from "simulate to search for
the best action" (`Planner`'s job) to "simulate to compare two already-
chosen candidates," the narrower, standalone capability this layer is
about.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Hashable


@dataclass(frozen=True)
class MonteCarloSimulator:
    """Runs `n_rollouts` independent rollouts of a given `policy` from a
    given starting `state`, for `horizon` steps, via a *stochastic*
    `transition_fn(state, action, rng) -> next_state` -- unlike
    `RecedingHorizonPlanner`'s `transition_fn(state, action) ->
    next_state`, this one takes an `rng` so each rollout can sample the
    environment's real noise independently, the whole point of running
    more than one. `policy(state, remaining_steps, rng) -> action` has
    the same signature experiment 13/16's `Agent.choose_action` already
    uses, so a trained agent's own method can be passed directly with no
    wrapping beyond unpacking its state representation.

    `simulate` returns the raw list of `n_rollouts` final scores, not
    just their mean -- the outcome *distribution*, since the spread is
    itself the signal a single deterministic search (`Planner`) cannot
    give. `compare_policies` runs `simulate` for two candidates from the
    identical starting state and reports which one wins, with each
    side's spread alongside the means."""

    n_rollouts: int

    def simulate(self, state: Any, policy: Callable[[Any, int, Any], Hashable],
                 transition_fn: Callable[[Any, Hashable, Any], Any],
                 score_fn: Callable[[Any], float], horizon: int, rng: Any) -> list[float]:
        if self.n_rollouts < 1:
            raise ValueError("n_rollouts must be at least 1")
        horizon = max(1, horizon)
        scores: list[float] = []
        for _ in range(self.n_rollouts):
            simulated_state = state
            for step in range(horizon):
                remaining = horizon - step
                action = policy(simulated_state, remaining, rng)
                simulated_state = transition_fn(simulated_state, action, rng)
            scores.append(score_fn(simulated_state))
        return scores

    def compare_policies(self, state: Any, policy_a: Callable[[Any, int, Any], Hashable],
                          policy_b: Callable[[Any, int, Any], Hashable],
                          transition_fn: Callable[[Any, Hashable, Any], Any],
                          score_fn: Callable[[Any], float], horizon: int, rng: Any) -> dict[str, Any]:
        scores_a = self.simulate(state, policy_a, transition_fn, score_fn, horizon, rng)
        scores_b = self.simulate(state, policy_b, transition_fn, score_fn, horizon, rng)
        mean_a, mean_b = statistics.mean(scores_a), statistics.mean(scores_b)
        return {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "stdev_a": statistics.pstdev(scores_a),
            "stdev_b": statistics.pstdev(scores_b),
            "winner": "a" if mean_a > mean_b else "b",
        }
