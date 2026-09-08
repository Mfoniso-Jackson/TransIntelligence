# Experiment 18 results — testing experiment 16's compounding-estimation-error hypothesis directly with `CalibrationVerifier`

Ran: `PYTHONPATH=. python experiments/exp18_calibration_verifier/run.py`.
12 seeds (the same seeds experiments 16 and 17 used) × 400 episodes
(5 steps each), residuals pooled across seeds within each window. See
`docs/research-agenda.md` #7o for the hypothesis and
`docs/related-work.md` §3l for the grounding theory. This fills
`Verifier` — the last reasoning-protocol stub from before Phase 4 left
unbuilt — and, unlike experiment 17, is a genuine new hypothesis test,
not a validation against an already-known answer.

## The question

Experiment 16's central finding — `mpc_beam_cusum_adapts` persistently
underperforms `greedy_cusum_adapts` post-shift — came with a
best-supported but explicitly **unconfirmed** explanation: multi-step
lookahead chains two predictions from the same learned dynamics model,
and each learned prediction's estimation error compounds in a way
single-step lookahead never pays for. That explanation was inferred from
the *pattern of final rewards*, never checked against the actual
estimation error of either agent's dynamics model directly. Both agents
fit the identical kind of one-step model via the identical OLS procedure
— the only difference is how many steps ahead each one *searches* when
choosing an action — but each agent's planning strategy steers it into
different regions of state space (experiment 16 already measured
`mpc_beam`'s post-shift visited positions have a larger spread than
`greedy`'s), which feeds back into what data each agent's own model
trains on. `CalibrationVerifier`, applied to each agent's one-step
prediction residuals against the environment's true noise floor
(`NOISE_SIGMA=0.1` — what a correctly-specified, well-fit model's
residuals should look like), asks directly: does `mpc_beam`'s dynamics
model carry more excess estimation error than `greedy`'s does,
post-shift?

**The confound this needed to control for:** pre-shift residuals (stable
data, no regime confusion) were checked as a baseline in the same run —
if pre-shift were also miscalibrated for both agents, that would point
to a cause unrelated to the regime shift.

## Result

| agent | window | n | observed coverage | claimed coverage | p-value | status |
|---|---|---|---|---|---|---|
| greedy_cusum_adapts | pre-shift | 11666 | 0.6582 | 0.6827 | <0.000001 | miscalibrated |
| greedy_cusum_adapts | post-shift | 11924 | 0.6656 | 0.6827 | 0.000068 | miscalibrated |
| mpc_beam_cusum_adapts | pre-shift | 11647 | 0.6584 | 0.6827 | <0.000001 | miscalibrated |
| mpc_beam_cusum_adapts | post-shift | 11943 | 0.6637 | 0.6827 | 0.000010 | miscalibrated |

**Both agents' one-step dynamics models are flagged `miscalibrated` in
every window** — but the observed coverage is nearly identical across
all four cells (0.658–0.666), a spread of under 0.008 versus a raw
shortfall from nominal of about 0.02–0.024. With this much pooled data
(n≈12,000 per cell), Kupiec's test has enough statistical power to flag
even a small, uninteresting effect as significant — the practically
relevant question is whether `mpc_beam` is *more* miscalibrated than
`greedy`, and it is not: post-shift, `greedy`'s observed coverage
(0.6656) is actually very slightly *higher* (closer to nominal) than
`mpc_beam`'s (0.6637), a 0.0019 gap — smaller than the 0.0074 pre-to-post
shift greedy itself shows on its own, i.e. within the noise floor of
this comparison, not a real, attributable difference.

**Robustness check on the pooling window**: restricting to only the
first 500 post-shift steps per seed (closer to experiment 16's earliest,
largest-gap chunk) gives the same pattern — greedy 0.6442, mpc 0.6399, a
0.0043 gap, still smaller than either agent's own pre-to-post shift —
so the null result isn't an artifact of averaging over the whole
post-shift window.

## What this establishes

- **The "self-steered training distribution differentially degrades the
  model" alternative explanation is not supported.** `mpc_beam`'s
  self-steering into a wider spread of positions (established in
  experiment 16) does not measurably degrade its own one-step dynamics
  model's calibration relative to `greedy`'s — ruling out one plausible
  mechanistic explanation experiment 16 didn't consider.
- **The small, universal miscalibration (~66% vs. 68.27% claimed
  coverage) is present pre-shift too, identically for both agents** —
  a baseline property of the fitting procedure (plausibly the familiar
  fact that in-sample-adjacent OLS residuals tend to slightly
  understate true out-of-sample prediction variance), not something the
  regime shift or either planning strategy causes.
- **By elimination, this leaves experiment 16's original
  compounding-across-chained-predictions hypothesis as the more
  plausible remaining explanation** for the reward gap — since the
  single-step model quality itself is not the differentiator, whatever
  the multi-step planner does *with* that model (chaining two uncertain
  predictions together before choosing an action) becomes the more
  likely remaining source of the gap. This experiment does not directly
  test the chaining mechanism itself — only rules out one alternative to
  it — so the hypothesis remains unconfirmed, now with one fewer
  competing explanation.

## Follow-up: does the finding hold at other coverage thresholds?

Ran: `PYTHONPATH=. python experiments/exp18_calibration_verifier/threshold_sweep.py`.
Same 12 seeds and post-shift residuals as above, `CalibrationVerifier`
re-applied at six thresholds (`threshold_sigmas` 0.5 through 3.0) instead
of only 1.0 — central and tail coverage are genuinely different
questions, and chained multi-step predictions could plausibly have
"fatter tails" (occasional much-larger errors) even while looking
well-calibrated near the center.

| threshold_sigmas | greedy observed coverage | mpc observed coverage | coverage gap |
|---|---|---|---|
| 0.5 | 0.3642 | 0.3630 | 0.0012 |
| 1.0 | 0.6656 | 0.6637 | 0.0019 |
| 1.5 | 0.8447 | 0.8399 | 0.0048 |
| 2.0 | 0.9326 | 0.9293 | 0.0032 |
| 2.5 | 0.9670 | 0.9637 | 0.0033 |
| 3.0 | 0.9792 | 0.9764 | 0.0028 |

**The gap stays tiny (0.001–0.005) at every threshold tested, from
central to tail coverage** — no evidence of a tail-specific calibration
difference the original 1-sigma check could have missed. Both agents
remain flagged `miscalibrated` throughout (the same small, universal
overconfidence found before, present at every threshold), but the
*difference between the two agents* specifically stays negligible across
the whole range. This robustifies the original finding rather than
revealing anything new: the "no measurable calibration difference"
conclusion is not an artifact of the one threshold originally tested.

## What this does not establish

- **The compounding-across-chained-predictions mechanism itself was
  still not directly manipulated** — this experiment tests single-step
  residual quality, which is necessarily identical in kind regardless of
  how many steps a planner chains at decision time; a direct test would
  need to compare the *planner's own* multi-step-ahead prediction error
  (not just the one-step model's residual) against reality. (Experiment
  20 later did test this, directly, by manipulating the environment's
  own noise level.)
- **Only one environment, one regime-shift severity, one lookahead depth
  (LOOKAHEAD=2) were tested** — as in experiment 16.
