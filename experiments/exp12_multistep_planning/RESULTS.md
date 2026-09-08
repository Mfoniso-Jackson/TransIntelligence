# Experiment 12 results — multi-step planning (receding-horizon control)

Ran: `PYTHONPATH=. python experiments/exp12_multistep_planning/run.py`.
15 seeds × 300 episodes per condition (5 steps per episode, first 20
episodes per seed are a shared random-action warmup, excluded from the
reported metric). See `docs/research-agenda.md` #7i for the hypothesis
and `docs/related-work.md` §3f for the grounding theory
(Richalet et al. 1978, MPC/receding-horizon control).

## The environment and the confound this needed to control for

`environments/transworld/delayed_control_env.py`'s `DelayedControlEnv`:
each nudge splits between an immediate effect (`lag_weight=0.5`) and a
delayed one landing on the position *after* next. Episodes persist for
`HORIZON=5` steps with reward only at the final step — a genuine
multi-step credit-assignment problem, unlike experiment 11's
single-step, fresh-state-per-trial design. A 2×2 factorial design
isolates "does planning horizon matter" from "does model quality
matter," rather than conflating them into one number: `greedy` (1-step
lookahead) vs. `mpc` (`LOOKAHEAD=2`-step receding-horizon lookahead,
replanned every step) crossed with `learned` (fits dynamics via
`ordinary_least_squares`) vs. `oracle` (given the true dynamics
exactly).

## Result

| condition | mean final-step reward |
|---|---|
| greedy_learned | -0.0246 |
| greedy_oracle | -0.0241 |
| mpc_learned | -0.0204 |
| mpc_oracle | -0.0200 |

**`mpc` beats `greedy` in both the learned and oracle pairs, by almost
exactly the same margin** (learned: 0.0246 → 0.0204, a gap of 0.0042;
oracle: 0.0241 → 0.0200, a gap of 0.0041) — this near-identical gap size
is exactly what the 2×2 design was built to check: the advantage is
about **planning horizon**, not about the learned model happening to be
better calibrated in one condition than the other. Checked directly
across seeds (not just on the aggregate mean): `mpc_learned` beats
`greedy_learned` in **14 of 15 seeds**, a consistent effect, not one
driven by a couple of outliers. Learned and oracle dynamics also perform
almost identically within each planning-horizon pair (0.0246 vs. 0.0241;
0.0204 vs. 0.0200) — consistent with `LinearDynamicsModel`-style linear
dynamics being recovered almost exactly given enough data, the same
finding experiment 11 made.

**The effect size is real but modest, and that's reported honestly, not
inflated.** A relative reduction in mean squared final-distance of
roughly 17% is a genuine, consistently-replicated planning advantage —
but nowhere near as dramatic as experiment 11's ~29x regret reduction
(world model vs. the linear model-free baseline). The reason is
structural, not a weakness of the mechanism: because every step gives
the agent full state feedback and lets it replan, a myopic greedy policy
already self-corrects reasonably well over several free steps even
without explicit lookahead — multi-step planning's advantage here is in
*efficiency* (reaching the target with less residual error, having
"anticipated" the delayed component of its own actions), not in
avoiding catastrophic, uncorrectable mistakes the way a longer, truly
irreversible commitment might expose.

## What this establishes

`choose_action`'s receding-horizon logic was verified by hand before
being trusted for anything: given known true dynamics and a specific
`(position, pending)` starting point, both the 1-step and 2-step
versions were checked to select the correct, hand-computed optimal
first action, and capping lookahead depth at the episode's remaining
steps was confirmed to make `mpc` reduce to exactly `greedy`'s choice at
the final step (as it must — with one step left, "look further ahead"
has nothing left to look at). All of this in
`tests/test_exp12_multistep_planning.py`, before the full noisy,
learned-dynamics experiment was trusted.

## What this does not establish

- **Only a 2-step lookahead was tested** — sufficient to see a nudge's
  entire effect under this specific 1-step lag structure, but not a
  general claim about how performance scales with longer horizons or
  more complex delay structures.
- **A single, fixed lag structure (`lag_weight=0.5`, 1-step delay)** —
  different lag strengths or multi-step delays weren't swept; the
  planning advantage's size likely depends on how much of an action's
  effect is genuinely invisible to a 1-step lookahead.
- **Exhaustive search over action sequences** — with 6 actions and
  lookahead 2, only 36 sequences per decision; this wouldn't scale to
  larger action sets or longer horizons without a smarter search. (This
  search logic has since been generalized into
  `transintelligence/planning/`'s `RecedingHorizonPlanner`, filling the
  `Planner` protocol stub — a domain-agnostic kernel primitive, but the
  same exhaustive-enumeration limitation carries over unchanged.)
- **No comparison against a nonlinear or momentum-based delay
  structure** — the delay here is a simple linear first-order lag; a
  genuinely nonlinear multi-step-critical environment (e.g. a
  double-integrator/"velocity" system) wasn't attempted, and might show
  a larger, more dramatic planning advantage than this modest,
  linear-lag result.
- **Stationary, single environment configuration** — no regime changes,
  no sweep across noise levels, unlike the noise/regime-length sweeps
  experiments 4-5 ran for the temporal-reasoning mechanism.
