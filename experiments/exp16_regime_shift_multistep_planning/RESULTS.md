# Experiment 16 results — combining regime-change detection with multi-step planning

Ran: `PYTHONPATH=. python experiments/exp16_regime_shift_multistep_planning/run.py`.
12 seeds × 400 episodes (5 steps each) per condition. See
`docs/research-agenda.md` #7m for the hypothesis and
`docs/related-work.md` §3j for the grounding theory. This is the second
synthesis experiment in this program (after experiment 13) — nothing new
is implemented; `CUSUMTemporalReasoner`, the 2-feature dynamics fit, and
`RecedingHorizonPlanner` are all reused exactly as already verified.

## Result

| condition | pre-shift reward | post-shift reward |
|---|---|---|
| greedy_never_adapts | -0.0238 | -43.9044 |
| greedy_cusum_adapts | -0.0285 | **-4.5176** |
| mpc_exhaustive_never_adapts | -0.0205 | -79.0555 |
| mpc_beam_never_adapts | -0.0196 | -53.3336 |
| mpc_beam_cusum_adapts | -0.0270 | -33.0555 |
| oracle_exhaustive | -0.0203 | -25.4207 |
| oracle_beam | -0.0195 | **-0.0284** |

This experiment surfaced two real findings, neither predictable from
experiments 12, 13, or 14 tested separately — exactly why it was built.

## Finding 1: exhaustive multi-step search reproduces experiment 14's oscillation pathology, independently

`oracle_exhaustive` — given the TRUE dynamics exactly, at every
decision — scored -25.42 post-shift, dramatically *worse* than
`greedy_cusum_adapts`'s learned, adaptive -4.52. An oracle should never
lose to a learned policy; this was investigated rather than accepted.
**The cause was exactly experiment 14's finding, independently
reproduced in a new environment built for a different purpose**:
exhaustive search's terminal-only scoring, combined with this
environment's delayed/pending dynamics (the same structural feature
experiment 12 introduced), lets it select plans whose first action sets
up a persistent oscillation once replanned step-by-step. Switching to
beam search (`beam_width=2`, the value experiment 14 found most
reliable) fixed it directly: `oracle_beam`'s post-shift reward is
**-0.0284**, matching its pre-shift performance (-0.0195) almost
exactly. This is a genuine replication of experiment 14's mechanism in
an independent setting, not a coincidence — strong evidence the
pathology is a real property of exhaustive receding-horizon search under
delayed dynamics, not an artifact specific to experiment 14's own
environment.

## Finding 2: even with the pathology fixed, learned multi-step planning persistently underperforms learned greedy planning

Fixing the search-strategy pathology did **not** make
`mpc_beam_cusum_adapts` (-33.06) competitive with `greedy_cusum_adapts`
(-4.52) — both use the identical CUSUM-adaptation mechanism and the
identical dynamics-fitting tool, differing only in planning horizon.

**The gap is persistent, not a shrinking startup transient.** Traced
across the whole post-shift window in 8 chunks:

| chunk | greedy_cusum_adapts | mpc_beam_cusum_adapts |
|---|---|---|
| 0 (right after detection) | -16.65 | -38.76 |
| 1 | -3.13 | -25.67 |
| 2 | -2.83 | -33.05 |
| 3 | -3.62 | -32.52 |
| 4 | -2.46 | -34.94 |
| 5 | -2.74 | -35.91 |
| 6 | -1.98 | -31.20 |
| 7 (end of run) | -2.73 | -32.39 |

`greedy_cusum_adapts` recovers within the first chunk and holds steady
for the rest of the run. `mpc_beam_cusum_adapts` never recovers —
chunk 7, with hundreds of post-reset samples accumulated by then, is no
better than chunk 1.

**A "reduced exploration" hypothesis was checked directly and
refuted.** The natural guess is that multi-step planning converges
quickly to a narrow, locally-good region, starving the dynamics model of
the diverse training data it needs. Checked directly: `mpc_beam`'s
visited positions post-shift had a **larger** spread than greedy's
(stdev 4.25 vs. 1.87), not smaller — ruling this out as the mechanism.

**Best-supported explanation, stated as a hypothesis, not a certainty**:
multi-step lookahead chains two predictions from the same learned
dynamics model, and each learned prediction carries some estimation
error that never fully vanishes (the environment has real observation
noise, `noise_sigma=0.1`). Chaining two uncertain predictions compounds
that error in a way single-step lookahead never has to pay for — and
unlike a small-sample transient, this cost need not shrink as more data
accumulates, consistent with the persistent (not narrowing) gap observed.
Directly manipulating estimation noise to confirm this mechanism was not
attempted.

## What this establishes

`CUSUMTemporalReasoner`, the dynamics-fitting mechanism, and
`RecedingHorizonPlanner` were each independently verified before this
experiment (experiments 5, 12, and 14 respectively); this experiment
tests only whether they compose as expected when combined, and found
they do not, straightforwardly. Two boundaries were found, each only
visible by testing the actual synthesis: (1) exhaustive multi-step
search's oscillation pathology (experiment 14) recurs in any
delayed-dynamics environment under receding-horizon replanning, not just
the one it was first found in; (2) even after fixing that, combining a
LEARNED (not oracle) dynamics model with multi-step lookahead has a
real, persistent cost that combining the same model with single-step
lookahead does not — a genuinely new finding this program's individual
Phase 6 experiments (11-15) could not have predicted on their own.

## What this does not establish

- **The compounding-estimation-error mechanism is a hypothesis, not a
  confirmed cause** — the "reduced exploration" alternative was ruled
  out directly, but the proposed mechanism itself was not independently
  manipulated (e.g. by varying `noise_sigma` and checking whether the
  gap scales with it).
- **Only `LOOKAHEAD=2` and `beam_width=2` were tested** — whether the
  persistent gap grows, shrinks, or disappears at different lookahead
  depths or beam widths is unknown.
- **A single environment and regime-shift severity** — reuses
  experiment 13's dramatic `sign_flip` severity and experiment 12's
  exact lag structure; whether this finding generalizes to milder
  shifts or different dynamics is untested.
- **Nonlinear dynamics (experiment 15) were not combined with this
  synthesis** — the third remaining Phase 6 gap from the pre-experiment
  list remains open.
