"""Baseline planning: receding-horizon (Model Predictive) control over an
arbitrary transition model (Phase 6, docs/research-agenda.md #7i,
docs/master-context.md §19: `Planner`).

`transintelligence/planning/` was, before this, a directory that didn't
exist at all, and `Planner` in `reasoning/interfaces.py` declared no
methods -- the same starting point every other empty-stub reasoning
protocol had before its phase started. `RecedingHorizonPlanner`
generalizes the multi-step planning logic experiment 12 first wrote as a
one-off, environment-specific function
(`experiments/exp12_multistep_planning/run.py`'s original `choose_action`)
into a domain-agnostic kernel primitive: it needs a `transition_fn(state,
action) -> next_state` and a `score_fn(state) -> float` to maximize, not
any particular state representation or action vocabulary. Everything
about experiment 12's specific environment (positions, pending nudges,
squared distance to target) now lives in that experiment's own code,
passed in as closures -- the planner itself knows nothing about it.

Grounded in the founding Model Predictive Control paper: Richalet, Rault,
Testud, Papon, *Model Predictive Heuristic Control: Applications to
Industrial Processes*, Automatica 14(5), 429-445, 1978 -- simulate
several steps ahead, execute only the first action, replan from the
newly observed state at every step. `RecedingHorizonPlanner.choose_action`
itself only computes one decision (the "simulate and pick the first
action" half); the "replan every step" half is the caller's
responsibility (call it again next step with the newly observed state),
exactly as experiment 12 already did.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Hashable


@dataclass(frozen=True)
class RecedingHorizonPlanner:
    """Exhaustively enumerates action sequences of the given depth,
    simulates each via `transition_fn` chained forward, scores the
    resulting final state via `score_fn`, and returns only the first
    action of the best-scoring sequence -- the standard receding-horizon
    protocol: execute one action, observe the true outcome, call again.

    `actions^depth` sequences are enumerated -- fine for the small
    action sets and shallow lookaheads this repo's experiments use
    (matching `CausalGraph`'s "simplicity over efficiency at this
    repo's scale" precedent), not a scalable search strategy for large
    action spaces or long horizons."""

    actions: tuple[Hashable, ...]

    def choose_action(self, state: Any, transition_fn: Callable[[Any, Hashable], Any],
                       score_fn: Callable[[Any], float], depth: int) -> Hashable:
        if not self.actions:
            raise ValueError("no actions to choose from")
        best_action: Hashable | None = None
        best_score: float | None = None
        for sequence in itertools.product(self.actions, repeat=max(1, depth)):
            simulated_state = state
            for action in sequence:
                simulated_state = transition_fn(simulated_state, action)
            score = score_fn(simulated_state)
            if best_score is None or score > best_score:
                best_score = score
                best_action = sequence[0]
        return best_action
