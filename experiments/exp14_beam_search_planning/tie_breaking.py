"""Follow-up to experiment 14 -- does a smarter exhaustive-search
tie-breaking rule (prefer minimal-effort actions among score-tied
sequences) close the gap to beam search, or is beam search's robustness
advantage a separate mechanism entirely? Experiment 14's own module
docstring flagged this explicitly as "a real, separate question this
experiment does not attempt to answer" -- it fixed an EARLIER, different
tie-breaking pathology (a 21-action vocabulary producing 19+ exact ties)
by reverting to a coarser 6-action vocabulary with few ties (1-2 per
state), then found the REAL, reported finding (the oscillation
pathology) traces to exhaustive search scoring only the FINAL state,
blind to path -- a mechanism that doesn't obviously need any ties to
operate at all.

This follow-up tests the tie-breaking hypothesis directly rather than
just noting it as untested: `choose_action_min_effort_tiebreak` -- a
standalone alternate exhaustive search (not merged into
`RecedingHorizonPlanner`; this is exploratory, not yet a validated
kernel change) -- finds the best final score exactly as the original
does, then among all sequences within floating-point tolerance of that
best score, picks the one with the SMALLEST cumulative |action| (the
most conservative path among equally "optimal" ones) rather than
whichever sequence enumeration order happens to find first.

Run: PYTHONPATH=. python experiments/exp14_beam_search_planning/tie_breaking.py
"""
from __future__ import annotations

import itertools
import statistics
import time
from typing import Callable

from experiments.exp14_beam_search_planning.run import (
    ACTIONS, CONVERGED_THRESHOLD, DEPTHS_WITH_GROUND_TRUTH, N_STATES, ROLLOUT_LENGTH, SEED,
    CountingTransition, evaluate, sample_states, score, true_transition,
)

TOLERANCE = 1e-9


def choose_action_min_effort_tiebreak(state: tuple[float, float], transition_fn: Callable, score_fn: Callable,
                                       depth: int, actions: tuple[float, ...]) -> float:
    """Same exhaustive search as `RecedingHorizonPlanner._choose_action_exhaustive`
    -- enumerate every action sequence of length `depth`, score by the
    final simulated state -- but among all sequences within `TOLERANCE`
    of the single best score, returns the first action of whichever one
    has the SMALLEST cumulative |action|, instead of whichever sequence
    `itertools.product` happens to enumerate first."""
    scored = []
    for sequence in itertools.product(actions, repeat=depth):
        simulated_state = state
        for action in sequence:
            simulated_state = transition_fn(simulated_state, action)
        scored.append((score_fn(simulated_state), sequence))
    best_score = max(s for s, _ in scored)
    tied = [sequence for s, sequence in scored if abs(s - best_score) <= TOLERANCE]
    best_sequence = min(tied, key=lambda sequence: sum(abs(a) for a in sequence))
    return best_sequence[0]


def evaluate_min_effort(depth: int, states: list[tuple[float, float]]) -> tuple[float, float, float, int]:
    """Mirrors experiment 14's own `evaluate()` exactly (same rollout
    protocol, same metrics), substituting `choose_action_min_effort_tiebreak`
    for `PLANNER.choose_action`."""
    scores, calls, times, converged = [], [], [], 0
    for s in states:
        state = s
        total_calls = 0
        total_time = 0.0
        for _ in range(ROLLOUT_LENGTH):
            counter = CountingTransition()
            t0 = time.perf_counter()
            action = choose_action_min_effort_tiebreak(state, counter, score, depth, ACTIONS)
            total_time += time.perf_counter() - t0
            total_calls += counter.calls
            state = true_transition(state, action)
        final_score = score(state)
        scores.append(final_score)
        calls.append(total_calls / ROLLOUT_LENGTH)
        times.append(total_time / ROLLOUT_LENGTH)
        if final_score > CONVERGED_THRESHOLD:
            converged += 1
    return statistics.mean(scores), statistics.mean(calls), statistics.mean(times), converged


def main() -> None:
    states = sample_states(N_STATES, SEED)
    for depth in DEPTHS_WITH_GROUND_TRUTH:
        print(f"=== depth={depth} ===")
        exhaustive_score, _, _, exhaustive_converged = evaluate(depth, None, states)
        min_effort_score, _, _, min_effort_converged = evaluate_min_effort(depth, states)
        beam_score, _, _, beam_converged = evaluate(depth, 2, states)
        print(f"{'method':<24} {'mean_score':>12} {'converged':>11}")
        print(f"{'exhaustive (original)':<24} {exhaustive_score:>12.4f} {f'{exhaustive_converged}/{len(states)}':>11}")
        print(f"{'exhaustive (min-effort)':<24} {min_effort_score:>12.4f} {f'{min_effort_converged}/{len(states)}':>11}")
        print(f"{'beam_width=2':<24} {beam_score:>12.4f} {f'{beam_converged}/{len(states)}':>11}")
        print()


if __name__ == "__main__":
    main()
