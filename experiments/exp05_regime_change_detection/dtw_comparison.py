"""Experiment 5, temporal comparison via DTW (docs/research-agenda.md #7b,
docs/related-work.md §9a) -- does dynamic_time_warp() correctly rank
trajectory similarity when naive same-index (pointwise) comparison can't,
because of time misalignment? This is the exact motivating case for DTW
(Sakoe & Chiba 1978), constructed directly rather than asserted from
authority.

Three trajectories, same length:
  - `template`: flat, bump to a high value, flat again.
  - `shifted`: the IDENTICAL bump shape, delayed onset by `shift` steps.
  - `different`: the bump at the SAME position as `template`, but a
    smaller/different height -- genuinely a different shape, not a
    warped version of the same one.

The correct ranking (by shape) is template ~ shifted << different.
Naive pointwise comparison, which only ever compares same-index values,
can rank this backwards once the shift is large enough that the bumps no
longer overlap in index -- swept below to find exactly where.

Run: PYTHONPATH=. python experiments/exp05_regime_change_detection/dtw_comparison.py
"""
from __future__ import annotations

from transintelligence.reasoning.temporal import dynamic_time_warp

LENGTH = 30
BUMP_START = 10
BUMP_LEN = 10
LOW, HIGH, DIFFERENT_HIGH = 0.2, 0.8, 0.6


def make_template() -> list[float]:
    return [LOW] * BUMP_START + [HIGH] * BUMP_LEN + [LOW] * (LENGTH - BUMP_START - BUMP_LEN)


def make_shifted(shift: int) -> list[float]:
    start = BUMP_START + shift
    return [LOW] * start + [HIGH] * BUMP_LEN + [LOW] * (LENGTH - start - BUMP_LEN)


def make_different() -> list[float]:
    return [LOW] * BUMP_START + [DIFFERENT_HIGH] * BUMP_LEN + [LOW] * (LENGTH - BUMP_START - BUMP_LEN)


def naive_distance(x: list[float], y: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(x, y))


def main() -> None:
    template = make_template()
    different = make_different()

    print(f"{'shift':<8} {'dtw(shifted)':>13} {'dtw(different)':>15} {'dtw_correct':>12} "
          f"{'naive(shifted)':>15} {'naive(different)':>17} {'naive_correct':>14}")
    for shift in range(0, BUMP_LEN + 6):
        shifted = make_shifted(shift)
        dtw_shifted = dynamic_time_warp(template, shifted)
        dtw_different = dynamic_time_warp(template, different)
        naive_shifted = naive_distance(template, shifted)
        naive_different = naive_distance(template, different)
        dtw_correct = dtw_shifted <= dtw_different
        naive_correct = naive_shifted <= naive_different
        print(f"{shift:<8} {dtw_shifted:>13.3f} {dtw_different:>15.3f} {str(dtw_correct):>12} "
              f"{naive_shifted:>15.3f} {naive_different:>17.3f} {str(naive_correct):>14}")


if __name__ == "__main__":
    main()
