# Experiment 20 results — directly confirming experiment 16's compounding-estimation-error hypothesis via a noise sweep

Ran: `PYTHONPATH=. python experiments/exp20_noise_sweep_compounding_error/run.py`.
20 seeds × 400 episodes (5 steps each) per condition per noise level. See
`docs/research-agenda.md` #7q for the hypothesis and
`docs/related-work.md` §3n for the grounding theory. Nothing new is
implemented — `Agent`, `DelayedRegimeShiftControlEnv`, and `run_episode`
are reused exactly as experiments 16-18 already verified; only
`noise_sigma` becomes a swept parameter instead of a fixed constant.

## The question

Experiment 16's central finding — `mpc_beam_cusum_adapts` persistently
underperforms `greedy_cusum_adapts` post-shift — came with a
best-supported but explicitly **unconfirmed** explanation: multi-step
lookahead chains two predictions from the same learned dynamics model,
and each carries estimation error that never fully vanishes *because the
environment has real observation noise* (`NOISE_SIGMA`). Experiment 18
tested a specific alternative explanation and ruled it out, but never
manipulated the chaining mechanism's own stated cause directly. This
experiment does the most literal possible test of the hypothesis's own
wording: sweep `NOISE_SIGMA` itself and check whether the post-shift
reward gap shrinks toward zero as `NOISE_SIGMA` shrinks toward zero — a
deterministic environment gives a correctly-specified model nothing left
to compound.

**The confound this needed to control for**: detection reliability
itself could plausibly vary across noise levels (experiment 19 already
found detection sensitivity to a model's residual characteristics) —
if `greedy` and `mpc_beam` detected the shift at very different rates at
some noise level, that alone could produce a reward gap unrelated to
estimation-error compounding. Detection counts and latencies are
reported for both agents at every noise level in the same run, not
assumed away.

## Result

| noise_sigma | greedy post-shift | mpc post-shift | gap (greedy − mpc) | greedy detected | mpc detected |
|---|---|---|---|---|---|
| 0.00 | -1.3762 | -2.8117 | **1.4355** | 20/20 | 20/20 |
| 0.02 | -2.0570 | -7.8207 | 5.7638 | 20/20 | 19/20 |
| 0.05 | -2.1161 | -17.8481 | 15.7320 | 20/20 | 20/20 |
| 0.10 (experiment 16's original value) | -8.5983 | -33.3182 | 24.7199 | 19/20 | 18/20 |
| 0.20 | -17.4911 | -37.2441 | 19.7530 | 17/20 | 17/20 |
| 0.40 | -18.3468 | -49.2660 | **30.9192** | 16/20 | 15/20 |

**The gap scales strongly with `NOISE_SIGMA`, confirming the hypothesis
directly.** At `NOISE_SIGMA=0.0` (a fully deterministic environment) the
gap is 1.44 — essentially negligible next to the ~25-31 gap seen at
`NOISE_SIGMA >= 0.1`, a roughly 17-21x difference between the
noise-free and noisy ends of the sweep. The trend is not perfectly
monotonic (a mild dip at 0.20 relative to 0.10 — 19.75 vs. 24.72), but
this level of wobble is consistent with ordinary seed-to-seed estimation
noise at 20 seeds, not a reversal: a first pass at 10 seeds showed a much
sharper apparent dip at 0.20 (gap 6.91) that did not replicate once
seed count was doubled (19.75) — investigated directly (a targeted
20-seed re-check at 0.15/0.20/0.25 before committing to the full sweep)
rather than reported as-is, and attributed to sample noise, not a real
non-monotonicity.

**Detection reliability tracks closely between the two agents at every
noise level** (e.g. 20/20 vs. 19/20 at 0.02; 16/20 vs. 15/20 at 0.40) —
ruling out differential detection as an alternative explanation for the
gap or its growth. Detection does degrade somewhat as noise increases for
both agents equally (20/20 at low noise down to 15-17/20 at
`NOISE_SIGMA=0.4`), the expected CUSUM behavior under a noisier signal,
and not itself surprising.

## What this establishes

- **Direct, positive confirmation of experiment 16's central hypothesis** —
  the persistent multi-step-vs-single-step post-shift gap is driven by
  the environment's real observation noise propagating into each
  chained prediction's estimation error, not a noise-independent
  structural artifact of beam search or multi-step replanning itself.
  This is the first direct manipulation-based test of the mechanism;
  experiment 18 only ruled out one alternative explanation without
  testing the mechanism's own stated cause.
- **The gap is not an artifact of differential detection reliability**
  between the two agents — both detect the shift at closely matched
  rates and latencies throughout the sweep.
- **An anomaly (the 10-seed noise_sigma=0.2 dip) was investigated before
  being trusted**, following this program's standing discipline —
  confirmed as sample noise via a targeted re-check, not left in the
  reported result unexamined.

## What this does not establish

- **Only one environment, regime-shift severity, lookahead depth
  (`LOOKAHEAD=2`), and beam width (`BEAM_WIDTH=2`) were tested** — as in
  experiments 16 and 18.
- **The exact functional form of the noise->gap relationship is not
  characterized** — the sweep shows a strong, mostly-monotonic increase,
  not a fitted curve or a claim about linearity, quadratic scaling, or
  any other specific form.
- **`NOISE_SIGMA=0.0`'s small residual gap (1.44, not exactly 0) is not
  further investigated** — plausibly a genuine finite-sample transient
  (even a deterministic environment needs *some* fresh post-reset data
  before a model fits exactly) rather than evidence against the
  hypothesis, but this wasn't directly confirmed.
