"""Follow-up to experiment 19 -- can the cold-start cost (Finding 1:
`oracle_adapts` performs WORSE than `never_adapts` under a mild shift)
be engineered around rather than just reported? Not a new numbered
experiment -- mitigates a cost experiment 19 diagnosed but didn't try to
fix, the same "diagnose in the main experiment, engineer a fix in a
follow-up" pattern experiment 13's `refit_interval_calibration.py`
already used for a different problem (false-positive rate).

## Attempt 1 (FAILED): shrink `refit_interval`

The first, seemingly well-motivated fix: `_training_transitions()`'s
post-reset pool grows continuously regardless of `refit_interval` --
`refit_interval` only controls how often `_refit()` re-checks it, so
shrinking it should only help or be neutral, never hurt, by catching new
sufficient data sooner. **Directly tested and REFUTED**: shrinking
`refit_interval` from 20 down to 2 made the mild-shift reversal
substantially WORSE (oracle - never went from -0.14 to -0.71), not
better, and even flipped the SEVERE-shift case from a large adaptation
benefit (+1.25) to a net cost (-0.43) at `refit_interval=2`.

**Investigated rather than left as a surprising negative result.**
Direct instrumentation shows why: with a short `refit_interval`, the
FIRST action to reach 3 observations (from the initial random-fallback
phase) triggers immediate, PERMANENT greedy exploitation of that one
action's noisy, barely-identified coefficient estimate --
`choose_action` has no exploration bonus once *any* action is known.
Measured directly: at `refit_interval=20`, the agent's post-reset action
selection stays diverse (a single action never exceeds ~38% of choices
in the first 60 post-reset trials); at `refit_interval=2`, one action
dominates 88-97% of choices almost immediately -- a premature lock-in
onto a single, unreliable estimate, actively worse than continuing
uniform random exploration would have been. This is a genuine, deeper
finding experiment 19 itself didn't surface: the cold-start cost isn't
only about insufficient data to fit anything -- it's compounded by a
purely-greedy policy's total lack of exploration once *any* action
becomes technically "known."

## Attempt 2: require full action coverage before going greedy

The diagnosis above suggests a different, better-targeted fix:
`choose_action` should keep exploring randomly until EVERY action has a
fitted coefficient, not just one -- avoiding lock-in onto a single early,
unreliable estimate. `FullCoverageOracleAgent`/`FullCoverageCUSUMAgent`
implement exactly that one-line change.

Run: PYTHONPATH=. python experiments/exp19_nonlinear_regime_shift/cold_start_mitigation.py
"""
from __future__ import annotations

import random
import statistics

from experiments.exp19_nonlinear_regime_shift.run import (
    ACTIONS, CONDITIONS, REGIME_SHIFT_TRIAL, TARGET, CUSUMAdaptiveAgent, OracleAdaptsAgent, run_condition,
)

SEEDS = list(range(10))
REFIT_INTERVALS = [20, 10, 5, 2]  # 20 is experiment 19's original value


class FullCoverageOracleAgent(OracleAdaptsAgent):
    """Only acts greedily once EVERY action has a fitted coefficient --
    otherwise keeps exploring uniformly at random, avoiding lock-in onto
    a single early, unreliable estimate."""

    def choose_action(self, state: float, rng: random.Random) -> str:
        known = self.model.known_actions()
        if len(known) < len(ACTIONS):
            return rng.choice(ACTIONS)
        return min(known, key=lambda a: (self.model.predict(state, a) - TARGET) ** 2)


class FullCoverageCUSUMAgent(CUSUMAdaptiveAgent):
    def choose_action(self, state: float, rng: random.Random) -> str:
        known = self.model.known_actions()
        if len(known) < len(ACTIONS):
            return rng.choice(ACTIONS)
        return min(known, key=lambda a: (self.model.predict(state, a) - TARGET) ** 2)


def run_agent(agent_factory, seed: int, post_shift_scale: float) -> float:
    return run_condition(agent_factory, seed, post_shift_scale)[1]


def main() -> None:
    for severity_label, post_shift_scale in (("mild_attenuation", 0.4), ("sign_flip", -1.0)):
        print(f"=== {severity_label} (post_shift_scale={post_shift_scale}) ===")
        never_post = statistics.mean(run_agent(CONDITIONS["never_adapts"], s, post_shift_scale) for s in SEEDS)
        print(f"never_adapts post-shift reward: {never_post:.4f}")
        print()
        print("--- Attempt 1: refit_interval sweep (greedy-on-first-known-action) ---")
        print(f"{'refit_interval':>14} {'oracle_adapts_post':>19} {'oracle - never':>15} {'cusum_adapts_post':>18} {'cusum - never':>14}")
        for refit_interval in REFIT_INTERVALS:
            oracle_post = statistics.mean(
                run_agent(lambda ri=refit_interval: OracleAdaptsAgent(REGIME_SHIFT_TRIAL, refit_interval=ri), s, post_shift_scale)
                for s in SEEDS
            )
            cusum_post = statistics.mean(
                run_agent(lambda ri=refit_interval: CUSUMAdaptiveAgent(refit_interval=ri), s, post_shift_scale)
                for s in SEEDS
            )
            print(f"{refit_interval:>14} {oracle_post:>19.4f} {oracle_post - never_post:>15.4f} "
                  f"{cusum_post:>18.4f} {cusum_post - never_post:>14.4f}")

        print()
        print("--- Attempt 2: full-coverage-before-greedy (refit_interval=20, the original) ---")
        oracle_fc = statistics.mean(
            run_agent(lambda: FullCoverageOracleAgent(REGIME_SHIFT_TRIAL), s, post_shift_scale) for s in SEEDS
        )
        cusum_fc = statistics.mean(
            run_agent(lambda: FullCoverageCUSUMAgent(), s, post_shift_scale) for s in SEEDS
        )
        sliding_window_post = statistics.mean(
            run_agent(CONDITIONS["sliding_window_baseline"], s, post_shift_scale) for s in SEEDS
        )
        print(f"full_coverage_oracle_post={oracle_fc:.4f}  (oracle - never = {oracle_fc - never_post:.4f})")
        print(f"full_coverage_cusum_post={cusum_fc:.4f}  (cusum - never = {cusum_fc - never_post:.4f})")
        print(f"sliding_window_baseline_post={sliding_window_post:.4f}  "
              f"(full_coverage_cusum - sliding_window = {cusum_fc - sliding_window_post:.4f})")
        print()


if __name__ == "__main__":
    main()
