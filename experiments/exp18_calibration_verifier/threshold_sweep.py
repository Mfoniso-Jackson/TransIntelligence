"""Follow-up to experiment 18 -- does `CalibrationVerifier`'s finding
(no measurable calibration difference between `greedy_cusum_adapts` and
`mpc_beam_cusum_adapts` post-shift) hold at coverage thresholds other
than the single one tested (`threshold_sigmas=1.0`)? Not a new numbered
experiment -- fills a gap experiment 18's own RESULTS.md flagged
explicitly under "What this does not establish": "whether a different
threshold changes the picture (e.g. tail-focused calibration at 2-3
sigma) is untested."

Central (near-1-sigma) coverage and TAIL coverage (2-3 sigma) are
genuinely different questions: a model's predictions could be
well-calibrated near the center (most errors close to what's claimed)
while still having "fatter tails" (occasional much-larger errors than
claimed) that a 1-sigma check alone wouldn't surface. Since
`mpc_beam_cusum_adapts` chains two predictions from the same learned
model, an occasional large single-step error compounding into a much
larger two-step one is exactly the kind of tail-specific effect a
central-coverage check could miss.

Agents are trained ONCE (the expensive part, reusing experiment 18's own
`train_instrumented_agent`/`pooled_residuals` unchanged) and then
checked at every threshold in `THRESHOLD_SIGMAS` against the SAME
residual data -- threshold only changes how `CalibrationVerifier`
interprets already-collected residuals, not what gets collected.

Run: PYTHONPATH=. python experiments/exp18_calibration_verifier/threshold_sweep.py
"""
from __future__ import annotations

from experiments.exp16_regime_shift_multistep_planning.run import BEAM_WIDTH, LOOKAHEAD, NOISE_SIGMA
from experiments.exp18_calibration_verifier.run import SEEDS, pooled_residuals, train_instrumented_agent
from transintelligence.verification import CalibrationVerifier

THRESHOLD_SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def main() -> None:
    greedy_agents = [train_instrumented_agent(lookahead=1, beam_width=None, seed=s) for s in SEEDS]
    mpc_agents = [train_instrumented_agent(lookahead=LOOKAHEAD, beam_width=BEAM_WIDTH, seed=s) for s in SEEDS]
    greedy_post = pooled_residuals(greedy_agents, post_shift=True)
    mpc_post = pooled_residuals(mpc_agents, post_shift=True)

    print(f"{'threshold_sigmas':>17} {'greedy_obs_cov':>15} {'greedy_status':>16} "
          f"{'mpc_obs_cov':>12} {'mpc_status':>16} {'cov_gap':>9}")
    for threshold in THRESHOLD_SIGMAS:
        verifier = CalibrationVerifier(threshold_sigmas=threshold)
        greedy_result = verifier.verify(greedy_post, claimed_sigma=NOISE_SIGMA)
        mpc_result = verifier.verify(mpc_post, claimed_sigma=NOISE_SIGMA)
        gap = greedy_result.observed_coverage - mpc_result.observed_coverage
        print(f"{threshold:>17.1f} {greedy_result.observed_coverage:>15.4f} {greedy_result.status.value:>16} "
              f"{mpc_result.observed_coverage:>12.4f} {mpc_result.status.value:>16} {gap:>9.4f}")


if __name__ == "__main__":
    main()
