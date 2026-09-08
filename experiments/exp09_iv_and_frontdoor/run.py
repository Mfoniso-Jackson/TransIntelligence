"""Experiment 9 -- do the two classic identification strategies for an
*unobserved* confounder (two-stage least squares via an instrument, and
front-door adjustment via a mediator) actually recover the true effect
where the backdoor criterion cannot apply at all -- and do they fail in
the specific, theory-predicted ways when their own assumptions are
violated? (docs/research-agenda.md #7f)

Two parts, two different confound controls:

## Part 1 -- 2SLS and the weak-instrument problem

`U` (never passed to `two_stage_least_squares` -- simulating "unobserved")
confounds `X` and `Y`. `Z` is a valid instrument: it affects `X` with
strength `INSTRUMENT_STRENGTH`, has no direct effect on `Y`, and is
independent of `U`. Naive OLS of `Y` on `X` is biased by `U` regardless of
`INSTRUMENT_STRENGTH` (it never uses `Z` at all). The confound this part
needs to control for: showing only a strong-instrument case would be
unconvincing on its own -- Bound, Jaeger, Baker (*Problems with
Instrumental Variables Estimation When the Correlation between the
Instruments and the Endogenous Explanatory Variable is Weak*, JASA
90(430), 1995) make a specific, falsifiable prediction: 2SLS bias
approaches *naive OLS's* bias as the instrument's explanatory power over
the treatment approaches zero. This part sweeps instrument strength down
to (and including) zero and checks that prediction directly, rather than
just picking one comfortable instrument strength and stopping.

## Part 2 -- front-door adjustment and a violated assumption

`U` confounds `X` and `Y` directly but must NOT affect the mediator `M`
for front-door identification to apply -- `X -> M -> Y` must fully
mediate `X`'s effect on `Y`, with no unblocked backdoor path from `M` to
`Y` other than through `X`. The confound control: a second condition
where `U` is ALSO given a direct effect on `M` (violating exactly that
assumption) uses the identical `front_door_adjustment` call on data that
now breaks its precondition -- the same "don't just show the positive
case, show what happens when the method's own assumption is violated"
discipline experiment 6's collider-adjustment condition used.

Run: PYTHONPATH=. python experiments/exp09_iv_and_frontdoor/run.py
"""
from __future__ import annotations

import random
import statistics

from transintelligence.reasoning.causal import (
    front_door_adjustment,
    ordinary_least_squares,
    two_stage_least_squares,
)

# --- Part 1: 2SLS ---
TRUE_EFFECT_XY = 0.6
CONFOUND_STRENGTH = 0.8  # U -> X and U -> Y
NOISE_SIGMA = 0.3
INSTRUMENT_STRENGTHS = [0.9, 0.5, 0.2, 0.05, 0.0]
N_PER_TRIAL = 500
SEEDS = list(range(30))


def simulate_iv(seed: int, instrument_strength: float, n: int = N_PER_TRIAL) -> tuple[list[float], list[float], list[float]]:
    rng = random.Random(seed)
    z_list, x_list, y_list = [], [], []
    for _ in range(n):
        u = rng.gauss(0, 1)
        z = rng.gauss(0, 1)
        x = instrument_strength * z + CONFOUND_STRENGTH * u + rng.gauss(0, NOISE_SIGMA)
        y = TRUE_EFFECT_XY * x + CONFOUND_STRENGTH * u + rng.gauss(0, NOISE_SIGMA)
        z_list.append(z); x_list.append(x); y_list.append(y)
    return z_list, x_list, y_list


# --- Part 2: front-door ---
TRUE_XM = 0.7
TRUE_MY = 0.5
CONFOUND_STRENGTH_FD = 0.8  # U -> X and U -> Y
VIOLATION_STRENGTH = 0.6    # U -> M, in the "violated assumption" condition only


def simulate_frontdoor(seed: int, u_to_m_strength: float, n: int = N_PER_TRIAL) -> tuple[list[float], list[float], list[float]]:
    rng = random.Random(seed)
    x_list, m_list, y_list = [], [], []
    for _ in range(n):
        u = rng.gauss(0, 1)
        x = CONFOUND_STRENGTH_FD * u + rng.gauss(0, NOISE_SIGMA)
        m = TRUE_XM * x + u_to_m_strength * u + rng.gauss(0, NOISE_SIGMA)
        y = TRUE_MY * m + CONFOUND_STRENGTH_FD * u + rng.gauss(0, NOISE_SIGMA)
        x_list.append(x); m_list.append(m); y_list.append(y)
    return x_list, m_list, y_list


def main() -> None:
    print("=== Part 1: 2SLS vs. instrument strength (true effect = 0.6) ===")
    print(f"{'strength':>9} {'mean_naive':>12} {'mean_2sls':>12} {'2sls_stdev':>12} {'naive_abs_bias':>15} {'2sls_abs_bias':>14}")
    for strength in INSTRUMENT_STRENGTHS:
        naive_estimates, iv_estimates = [], []
        for seed in SEEDS:
            z, x, y = simulate_iv(seed, strength)
            naive_estimates.append(ordinary_least_squares([[1.0, xv] for xv in x], y)[1])
            iv_estimates.append(two_stage_least_squares(z, x, y))
        mean_naive = statistics.mean(naive_estimates)
        mean_iv = statistics.mean(iv_estimates)
        iv_stdev = statistics.pstdev(iv_estimates)
        naive_bias = abs(mean_naive - TRUE_EFFECT_XY)
        iv_bias = abs(mean_iv - TRUE_EFFECT_XY)
        print(f"{strength:>9} {mean_naive:>12.4f} {mean_iv:>12.4f} {iv_stdev:>12.4f} {naive_bias:>15.4f} {iv_bias:>14.4f}")

    print("\n=== Part 2: front-door adjustment, valid vs. violated assumption (true effect = 0.35) ===")
    true_total_effect = TRUE_XM * TRUE_MY
    print(f"{'condition':<20} {'mean_naive':>12} {'mean_frontdoor':>16} {'naive_abs_bias':>15} {'fd_abs_bias':>13}")
    for label, u_to_m in [("valid (U ->/-> M)", 0.0), ("violated (U -> M)", VIOLATION_STRENGTH)]:
        naive_estimates, fd_estimates = [], []
        for seed in SEEDS:
            x, m, y = simulate_frontdoor(seed, u_to_m)
            naive_estimates.append(ordinary_least_squares([[1.0, xv] for xv in x], y)[1])
            fd_estimates.append(front_door_adjustment(x, m, y))
        mean_naive = statistics.mean(naive_estimates)
        mean_fd = statistics.mean(fd_estimates)
        naive_bias = abs(mean_naive - true_total_effect)
        fd_bias = abs(mean_fd - true_total_effect)
        print(f"{label:<20} {mean_naive:>12.4f} {mean_fd:>16.4f} {naive_bias:>15.4f} {fd_bias:>13.4f}")


if __name__ == "__main__":
    main()
