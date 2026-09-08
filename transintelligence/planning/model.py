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

The exhaustive `actions^depth` search this class started with (still the
default when `beam_width` is omitted) was flagged from the start as not
scaling to larger action sets or longer horizons -- experiments 12 and
13 never needed it to, at 6 actions and depth 2. `beam_width` adds the
standard fix: beam search (Lowerre, *The Harpy Speech Recognition
System*, PhD thesis, Carnegie Mellon University, 1976) keeps only the
`beam_width` best-scoring partial sequences at each depth step instead
of expanding every one, trading a small, measurable quality gap for a
large, measurable reduction in `transition_fn` calls -- verified
directly in `experiments/exp14_beam_search_planning/`, not just assumed
from the algorithm's textbook reputation.
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

    `actions^depth` sequences are enumerated by default -- fine for the
    small action sets and shallow lookaheads experiments 12 and 13 use
    (matching `CausalGraph`'s "simplicity over efficiency at this
    repo's scale" precedent), not a scalable search strategy for large
    action spaces or long horizons. Pass `beam_width` to switch to beam
    search instead, trading search completeness for a bounded, much
    smaller number of `transition_fn` calls -- see
    `experiments/exp14_beam_search_planning/` for exactly how much of
    each."""

    actions: tuple[Hashable, ...]

    def choose_action(self, state: Any, transition_fn: Callable[[Any, Hashable], Any],
                       score_fn: Callable[[Any], float], depth: int,
                       beam_width: int | None = None) -> Hashable:
        if not self.actions:
            raise ValueError("no actions to choose from")
        depth = max(1, depth)
        if beam_width is None:
            return self._choose_action_exhaustive(state, transition_fn, score_fn, depth)
        return self._choose_action_beam_search(state, transition_fn, score_fn, depth, beam_width)

    def _choose_action_exhaustive(self, state: Any, transition_fn: Callable[[Any, Hashable], Any],
                                   score_fn: Callable[[Any], float], depth: int) -> Hashable:
        best_action: Hashable | None = None
        best_score: float | None = None
        for sequence in itertools.product(self.actions, repeat=depth):
            simulated_state = state
            for action in sequence:
                simulated_state = transition_fn(simulated_state, action)
            score = score_fn(simulated_state)
            if best_score is None or score > best_score:
                best_score = score
                best_action = sequence[0]
        return best_action

    def _choose_action_beam_search(self, state: Any, transition_fn: Callable[[Any, Hashable], Any],
                                    score_fn: Callable[[Any], float], depth: int, beam_width: int) -> Hashable:
        """Keeps only the `beam_width` best-scoring candidates after each
        depth step, expanding every survivor by every action rather than
        every possible sequence -- `O(depth * beam_width * len(actions))`
        `transition_fn` calls instead of `O(len(actions)**depth)`. Each
        beam entry tracks (current simulated state, first action taken
        to reach it) so the eventual winner's *first* action -- the only
        one actually executed under the receding-horizon protocol -- is
        always available, however deep the beam has gone."""
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        beam: list[tuple[Any, Hashable]] = [(state, None)]  # (simulated_state, first_action_or_None)
        for _ in range(depth):
            candidates: list[tuple[Any, Hashable]] = []
            for simulated_state, first_action in beam:
                for action in self.actions:
                    next_state = transition_fn(simulated_state, action)
                    candidates.append((next_state, action if first_action is None else first_action))
            candidates.sort(key=lambda candidate: score_fn(candidate[0]), reverse=True)
            beam = candidates[:beam_width]
        _, best_action = max(beam, key=lambda candidate: score_fn(candidate[0]))
        return best_action
