"""Experiment 14 -- does beam search let `RecedingHorizonPlanner` scale
to lookahead depths its original exhaustive search cannot reach, and at
what quality cost? (docs/research-agenda.md #7k, Phase 6, continued)

`RecedingHorizonPlanner`'s exhaustive search (experiments 12-13) was
flagged from the start as `O(len(actions)**depth)` -- fine at 6 actions
and depth 2 (36-216 `transition_fn` calls per decision), not scalable
beyond that. This experiment set out to test whether beam search
recovers near-exhaustive decision quality at dramatically lower cost --
the expected, textbook answer. It found something more interesting
instead, by investigating a result that looked wrong rather than
reporting it: beam search doesn't just approximate exhaustive search
more cheaply here, it is measurably MORE ROBUST to a genuine
receding-horizon-control pathology exhaustive search's terminal-only
scoring is vulnerable to (see "Two real bugs" below) -- cheaper AND
better, not cheaper at a quality cost, in this specific replanning
setting.

## Design

A self-contained synthetic control task (the same first-order-lag
dynamics as `DelayedControlEnv`, experiment 12, and its exact 6-action
vocabulary -- deliberately NOT widened, see "a second bug" below -- with
TRUE dynamics known exactly, since this experiment is about the search
algorithm, not about learning: no dynamics model to fit, no
seeds-for-noise beyond randomizing starting states). For `depth` in
{2, 3} (where exhaustive search is still practical AND, checked
directly, essentially free of score ties -- see below -- so its result
is trustworthy as ground truth) and each of 15 random starting states,
compare exhaustive search against beam search at several `beam_width`
values: mean achieved score over a full multi-step rollout (not a
single step -- see "a bug" below), `transition_fn` call count, and
wall-clock time. A third depth (6) demonstrates the actual scaling claim
directly: exhaustive search is not even attempted there (extrapolated
cost: `6**6 * 6` ≈ 280,000 calls per decision) while beam search
completes in a fraction of a millisecond.

## The confound this needed to control for

Showing beam search uses fewer calls proves nothing about whether it's
still a *good* planner -- an empty search would be even cheaper and
useless. Every beam-search condition is scored against the SAME
exhaustive-search ground truth at the same starting states, so both the
speedup and the quality cost are measured directly, not asserted.

## Two real bugs this experiment's own first two runs caught

**Bug 1**: the first version of `evaluate()` executed only ONE real step
per decision and scored that single step's immediate outcome -- not the
multi-step trajectory the search was actually optimizing for. That run
showed beam search scoring *better* than exhaustive search, which is
impossible by construction (exhaustive search checks every sequence at
a given depth, so nothing at that depth can beat it on the objective it
was actually optimizing). Fixed by executing a full `ROLLOUT_LENGTH`-step
rollout with replanning at every step (the actual receding-horizon
protocol -- Richalet et al. 1978, and how experiment 12 evaluated
planning quality) and scoring the state after the whole rollout.

**Bug 2**: even with that fix, beam search still appeared to beat
exhaustive search. Investigated rather than dismissed as noise: a
21-action, finely-spaced vocabulary creates massive EXACT ties in final
predicted score -- checked directly, one state at depth 3 had 19
different 3-action sequences all scoring exactly 0.0, with first actions
ranging from -2.0 to 0.0. `RecedingHorizonPlanner`'s exhaustive search
breaks ties by keeping whichever sequence it finds first (`score >
best_score`, strict), which with `self.actions` in ascending order means
always favoring the most extreme (most negative) action among ties --
not necessarily a good choice for REPLANNING, since committing to an
extreme move can require an equally extreme correction next step in a
way a more moderate tied option wouldn't have. This isn't a bug in the
sense of wrong code -- exhaustive search does find A truly optimal
sequence -- but the specific one it commits to, among many equally
"optimal" options, can be a poor choice for a receding-horizon
controller that replans every step. The fix here was to remove the
pathology rather than paper over it: reuse experiment 12's original,
widely-spaced 6-action vocabulary instead of a fine 21-action one --
checked directly to have far fewer ties (1 and 2 tied sequences at
depths 2 and 3 respectively, vs. 19+ with 21 actions) -- so exhaustive
search's result is a trustworthy, low-ambiguity ground truth again.
Characterizing exhaustive search's tie-breaking behavior itself (e.g. a
smarter rule that prefers minimal-effort actions among ties) is a real,
separate question this experiment does not attempt to answer.

**Even with that fix, a large quality gap remained -- and tracing one
misbehaving state step by step (not just re-averaging and hoping it
washes out) found something more interesting than a tie-breaking
artifact.** Exhaustive search's chosen policy entered a persistent
OSCILLATION on some starting states -- position stuck away from target,
`pending` alternating sign every step, forever -- while beam search
converged to the target and stayed there from the same starting state.
The mechanism: exhaustive search scores only the FINAL simulated state
after the full `depth`-step lookahead, indifferent to the path taken to
reach it -- it can select a sequence whose promised final position looks
good on paper while its first action, once actually executed and then
REPLANNED from scratch (only one action of any plan is ever really
executed under the receding-horizon protocol), sets up a self-
reinforcing overshoot-correct-overshoot cycle. Beam search, by contrast,
scores every INTERMEDIATE partial state during its own expansion (not
just the terminal one) and prunes candidates whose partial trajectory
already looks bad -- an incidental but real bias toward monotonic
progress toward the target, which happens to make it more robust to this
specific receding-horizon pathology, not just cheaper. This is reported
as the actual finding, not smoothed into the originally-expected "beam
search approximates exhaustive search's quality at lower cost" story --
investigating the surprising result rather than discarding it is what
surfaced it.

Run: PYTHONPATH=. python experiments/exp14_beam_search_planning/run.py
"""
from __future__ import annotations

import random
import statistics
import time

from transintelligence.planning import RecedingHorizonPlanner

ACTIONS = (-2.0, -1.0, -0.3, 0.3, 1.0, 2.0)  # experiment 12's original 6-action vocabulary, deliberately unchanged
LAG_WEIGHT = 0.5
TARGET = 0.0
STATE_RANGE = (-5.0, 5.0)
N_STATES = 15
ROLLOUT_LENGTH = 5  # real steps executed per state, with replanning every step -- matches experiment 12's HORIZON
DEPTHS_WITH_GROUND_TRUTH = [2, 3]  # depth 4+ starts showing real score ties with this action set -- see module docstring
INTRACTABLE_DEPTH = 6
BEAM_WIDTHS = [2, 3, 4]
SEED = 0

PLANNER = RecedingHorizonPlanner(actions=ACTIONS)


def true_transition(state: tuple[float, float], action: float) -> tuple[float, float]:
    position, pending = state
    return position + LAG_WEIGHT * action + (1 - LAG_WEIGHT) * pending, action


def score(state: tuple[float, float]) -> float:
    position, _ = state
    return -((position - TARGET) ** 2)


class CountingTransition:
    """Wraps `true_transition` to count calls -- a deterministic,
    hardware-independent proxy for search cost, reported alongside
    wall-clock time for intuition."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state: tuple[float, float], action: float) -> tuple[float, float]:
        self.calls += 1
        return true_transition(state, action)


def sample_states(n: int, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.uniform(*STATE_RANGE), rng.uniform(*STATE_RANGE)) for _ in range(n)]


CONVERGED_THRESHOLD = -0.1  # final score above this counts as "reached and stayed near target"


def evaluate(depth: int, beam_width: int | None, states: list[tuple[float, float]]) -> tuple[float, float, float, int]:
    """Returns (mean_true_score, mean_calls_per_decision,
    mean_wall_clock_seconds_per_decision, converged_count). For each
    starting state, runs a full `ROLLOUT_LENGTH`-step receding-horizon
    rollout -- replan with `PLANNER.choose_action` at every step, execute
    one real step via the TRUE dynamics, repeat -- and scores the state
    after the whole rollout. This is what the search is actually
    supposed to be good at: not the best-looking single next step, the
    best multi-step outcome. `converged_count` is how many of the
    starting states ended the rollout within `CONVERGED_THRESHOLD` of
    target, vs. stuck oscillating or otherwise far away -- the direct,
    quantified version of the oscillation finding below, not just an
    anecdote from one traced state. Call counts and wall-clock time are
    totaled across the rollout, then reported per-decision for
    comparability across depths."""
    scores, calls, times, converged = [], [], [], 0
    for s in states:
        state = s
        total_calls = 0
        total_time = 0.0
        for _ in range(ROLLOUT_LENGTH):
            counter = CountingTransition()
            t0 = time.perf_counter()
            action = PLANNER.choose_action(state, counter, score, depth=depth, beam_width=beam_width)
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
        exhaustive_score, exhaustive_calls, exhaustive_time, exhaustive_converged = evaluate(depth, None, states)
        print(f"{'method':<16} {'mean_score':>12} {'converged':>11} {'mean_calls':>12} {'call_reduction':>15} {'mean_time_ms':>14}")
        print(f"{'exhaustive':<16} {exhaustive_score:>12.4f} {f'{exhaustive_converged}/{len(states)}':>11} {exhaustive_calls:>12.1f} {'1.0x':>15} {exhaustive_time*1000:>14.3f}")
        for beam_width in BEAM_WIDTHS:
            beam_score, beam_calls, beam_time, beam_converged = evaluate(depth, beam_width, states)
            reduction = exhaustive_calls / beam_calls if beam_calls else float("inf")
            print(f"{'beam_width=' + str(beam_width):<16} {beam_score:>12.4f} {f'{beam_converged}/{len(states)}':>11} {beam_calls:>12.1f} {f'{reduction:.1f}x':>15} {beam_time*1000:>14.3f}")
        print()

    print(f"=== depth={INTRACTABLE_DEPTH} (exhaustive not attempted -- extrapolated {len(ACTIONS)}**{INTRACTABLE_DEPTH}*{INTRACTABLE_DEPTH} = {len(ACTIONS)**INTRACTABLE_DEPTH * INTRACTABLE_DEPTH:,} calls) ===")
    print(f"{'method':<16} {'mean_score':>12} {'converged':>11} {'mean_calls':>12} {'mean_time_ms':>14}")
    for beam_width in BEAM_WIDTHS:
        beam_score, beam_calls, beam_time, beam_converged = evaluate(INTRACTABLE_DEPTH, beam_width, states)
        print(f"{'beam_width=' + str(beam_width):<16} {beam_score:>12.4f} {f'{beam_converged}/{len(states)}':>11} {beam_calls:>12.1f} {beam_time*1000:>14.3f}")


if __name__ == "__main__":
    main()
