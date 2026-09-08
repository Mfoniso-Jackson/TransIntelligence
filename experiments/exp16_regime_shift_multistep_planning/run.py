"""Experiment 16 -- does regime-change detection compose with multi-step
planning, or does testing them separately (experiments 12/14 for
planning, 13 for detection) miss something about how they interact?
(docs/research-agenda.md #7m, Phase 6, continued)

This is the second synthesis experiment in this program (after
experiment 13, which combined CUSUM detection with single-step world-
model planning): does the same detection-triggered-adaptation mechanism
still work when the planner is a multi-step `RecedingHorizonPlanner`
(experiments 12/14) instead of a single-step greedy lookahead? Nothing
new is implemented here -- `CUSUMTemporalReasoner`, the 2-feature
(position, pending) dynamics fit from experiment 12, and
`RecedingHorizonPlanner` are all reused exactly as already verified.

`DelayedRegimeShiftControlEnv`
(`environments/transworld/delayed_regime_shift_env.py`) combines
experiment 12's persistent multi-step episodes (state=(position,
pending), `HORIZON` steps per episode, terminal-only reward) with
experiment 13's silent mid-run dynamics shift (the actuator's effect
reverses direction at an unknown cumulative step -- experiment 13's
dramatic `sign_flip` severity, reused rather than re-derived).

## The confound this needed to control for

A 2x2 factorial design (planning horizon x adaptation) isolates whether
multi-step planning and regime detection interfere with, are
independent of, or amplify each other -- rather than testing only the
"everything on" condition and being unable to attribute the result to
either ingredient specifically.

## A real finding this experiment's first run reproduced, unprompted

The first version used exhaustive search (`beam_width=None`) for every
multi-step condition, matching experiments 12/13's own defaults. The
`oracle` condition -- given the TRUE dynamics exactly, at every decision
-- scored a post-shift reward of -25.4, dramatically WORSE than
`greedy_cusum_adapts`'s learned, adaptive -4.5. An oracle should never
lose to a learned policy; this was investigated rather than accepted.
The cause was exactly experiment 14's finding, independently reproduced
in a new environment built for a different purpose: exhaustive search's
terminal-only scoring, combined with this environment's delayed/pending
dynamics (the same structural feature experiment 12 introduced), lets it
select plans whose first action sets up a persistent oscillation once
replanned step-by-step. Switching the multi-step conditions to beam
search (`beam_width=2`, the same value experiment 14 found most reliable)
fixed it directly: the oracle's post-shift reward went from -25.4 to
~-0.02-0.03, matching its pre-shift performance almost exactly. Both
exhaustive and beam-search multi-step conditions are kept below, so the
pathology and its fix are both visible, not just the fixed version.

- `greedy_never_adapts` / `greedy_cusum_adapts`: 1-step lookahead
  (depth=1 has no path to be blind to -- exhaustive and beam search are
  identical there), with and without CUSUM-triggered adaptation
  (experiment 13's mechanism).
- `mpc_exhaustive_never_adapts`: multi-step lookahead via exhaustive
  search, no adaptation -- reproduces the oscillation pathology alone.
- `mpc_beam_never_adapts`: multi-step lookahead via beam search, no
  adaptation -- does planning ahead alone help cope with a stale,
  post-shift-wrong model, once the pathology is controlled for?
- `mpc_beam_cusum_adapts`: multi-step lookahead via beam search +
  CUSUM-triggered adaptation -- the full, correctly-composed combination.
- `oracle_exhaustive` / `oracle_beam`: the true dynamics at every
  decision, with each search strategy -- isolates the cost of learning
  from the cost of the search-strategy pathology.

## A second, deeper finding -- the actual answer to the original question

Fixing the search-strategy pathology (beam search) did NOT make
`mpc_beam_cusum_adapts` competitive with `greedy_cusum_adapts`.
Investigated rather than reported as a residual quirk: the gap is
PERSISTENT, not a shrinking startup transient -- traced across the whole
post-shift window in 8 chunks, `mpc_beam_cusum_adapts` stayed around
-25 to -39 throughout, while `greedy_cusum_adapts` recovered within the
first chunk and then held steady near -2 to -3, for the rest of the run.
A "reduced exploration" hypothesis (multi-step planning converges
quickly to a narrow region, starving the model of diverse training data)
was checked directly and REFUTED: `mpc_beam`'s visited positions
post-shift had a LARGER spread than greedy's (stdev 4.25 vs. 1.87), not
smaller. The best-supported remaining explanation: multi-step lookahead
chains two predictions from the same learned model, and each learned
prediction carries some estimation error that never fully vanishes (the
environment has real observation noise, `NOISE_SIGMA`) -- chaining two
uncertain predictions compounds that error in a way a single-step
lookahead never has to pay for, and unlike a small-sample transient, this
cost doesn't shrink as more data accumulates. This is stated as the
best-supported hypothesis given what was checked, not a certainty --
directly manipulating estimation noise to confirm the mechanism wasn't
attempted.

Run: PYTHONPATH=. python experiments/exp16_regime_shift_multistep_planning/run.py
"""
from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta, timezone

from environments.transworld import DelayedRegimeShiftControlEnv, NUDGES
from transintelligence.core.states import State, StateHistory
from transintelligence.planning import RecedingHorizonPlanner
from transintelligence.reasoning.causal import ordinary_least_squares
from transintelligence.reasoning.temporal import CUSUMTemporalReasoner

ACTIONS = list(NUDGES.keys())
TARGET = 0.0
LAG_WEIGHT = 0.5
HORIZON = 5
INIT_RANGE = (-5.0, 5.0)
NOISE_SIGMA = 0.1
LOOKAHEAD = 2
BEAM_WIDTH = 2  # the value experiment 14 found most reliable
REGIME_SHIFT_STEP = 1000
POST_SHIFT_SCALE = -1.0
WARMUP_EPISODES = 15
REFIT_INTERVAL = 20
CUSUM_CHECK_INTERVAL = 20
N_EPISODES = 400
SEEDS = list(range(12))
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class Agent:
    """Shared bookkeeping for all four learned conditions: log every
    transition with its global step, periodically refit a 2-feature
    (position, pending) linear dynamics model per action, and choose
    actions via `RecedingHorizonPlanner` at `lookahead` depth (1 =
    greedy, matching experiments 11/13; >1 = genuine multi-step
    planning, matching experiment 12). `adapts=True` additionally
    tracks residuals and applies experiment 13's CUSUM-triggered
    data-discarding when a shift is detected; `adapts=False` reproduces
    experiment 12's `never_adapts` baseline, now facing a regime shift
    it cannot see."""

    def __init__(self, lookahead: int, adapts: bool, beam_width: int | None = None) -> None:
        self.lookahead = lookahead
        self.adapts = adapts
        self.beam_width = beam_width
        self.all_transitions: list[tuple[int, float, float, str, float]] = []  # (global_step, position, pending, action, next_position)
        self.coefficients: dict[str, tuple[float, float, float]] = {}
        self.reset_from_step = 0
        self.residual_states: list[State] = []
        self.reasoner = CUSUMTemporalReasoner()  # default calibration, unchanged from experiments 5/13
        self.reset_steps: list[int] = []
        self.planner = RecedingHorizonPlanner(actions=tuple(ACTIONS))

    def _training_transitions(self) -> list[tuple[int, float, float, str, float]]:
        if not self.adapts:
            return self.all_transitions
        return [t for t in self.all_transitions if t[0] >= self.reset_from_step]

    def _refit(self) -> None:
        by_action: dict[str, list[tuple[float, float, float]]] = {}
        for _, p, pend, a, np_ in self._training_transitions():
            by_action.setdefault(a, []).append((p, pend, np_))
        coefficients: dict[str, tuple[float, float, float]] = {}
        for a, rows in by_action.items():
            if len(rows) < 3:
                continue
            features = [[1.0, p, pend] for p, pend, _ in rows]
            targets = [np_ for _, _, np_ in rows]
            try:
                intercept, b_pos, b_pend = ordinary_least_squares(features, targets)
            except ValueError:
                continue
            coefficients[a] = (intercept, b_pos, b_pend)
        self.coefficients = coefficients

    def _predict(self, position: float, pending: float, action: str) -> float:
        intercept, b_pos, b_pend = self.coefficients[action]
        return intercept + b_pos * position + b_pend * pending

    def choose_action(self, position: float, pending: float, remaining_steps: int, rng: random.Random) -> str:
        known = list(self.coefficients.keys())
        if not known:
            return rng.choice(ACTIONS)
        depth = max(1, min(self.lookahead, remaining_steps))
        planner = self.planner if len(known) == len(ACTIONS) else RecedingHorizonPlanner(actions=tuple(known))

        def transition_fn(state: tuple[float, float], action: str) -> tuple[float, float]:
            p, pend = state
            return self._predict(p, pend, action), NUDGES[action]

        def score_fn(state: tuple[float, float]) -> float:
            p, _ = state
            return -((p - TARGET) ** 2)

        return planner.choose_action((position, pending), transition_fn, score_fn, depth, beam_width=self.beam_width)

    def observe(self, global_step: int, position: float, pending: float, action: str, next_position: float) -> None:
        if self.adapts and action in self.coefficients:
            predicted = self._predict(position, pending, action)
            residual = next_position - predicted
            self.residual_states.append(State("world_model_residual", {"residual": residual}, T0 + timedelta(seconds=global_step)))
        self.all_transitions.append((global_step, position, pending, action, next_position))
        if len(self.all_transitions) % REFIT_INTERVAL == 0:
            self._refit()
        if self.adapts and global_step > 0 and global_step % CUSUM_CHECK_INTERVAL == 0 and len(self.residual_states) >= self.reasoner.burn_in + 5:
            history = StateHistory(self.residual_states)
            change_points = self.reasoner.change_points(history, "residual")
            if change_points:
                latest_cp = int(max((cp - T0).total_seconds() for cp in change_points))
                if latest_cp > self.reset_from_step:
                    self.reset_from_step = latest_cp
                    self.reset_steps.append(latest_cp)
                    self._refit()
                    self.residual_states = [s for s in self.residual_states if s.timestamp >= T0 + timedelta(seconds=latest_cp)]


class OracleAgent:
    """Given the true dynamics exactly at every decision (the current
    regime known exactly, though not future shifts within its own
    lookahead window) -- the ceiling, isolating the cost of learning
    from the cost of detection delay."""

    def __init__(self, env: DelayedRegimeShiftControlEnv, lookahead: int, beam_width: int | None = None) -> None:
        self.env = env
        self.lookahead = lookahead
        self.beam_width = beam_width
        self.planner = RecedingHorizonPlanner(actions=tuple(ACTIONS))

    def choose_action(self, position: float, pending: float, remaining_steps: int, rng: random.Random) -> str:
        depth = max(1, min(self.lookahead, remaining_steps))
        step_at_decision = self.env.global_step

        def transition_fn(state: tuple[float, float], action: str) -> tuple[float, float]:
            p, pend = state
            return self.env.true_next_position(p, pend, action, step_at_decision), NUDGES[action]

        def score_fn(state: tuple[float, float]) -> float:
            p, _ = state
            return -((p - TARGET) ** 2)

        return self.planner.choose_action((position, pending), transition_fn, score_fn, depth, beam_width=self.beam_width)

    def observe(self, global_step: int, position: float, pending: float, action: str, next_position: float) -> None:
        pass


CONDITIONS = {
    "greedy_never_adapts": lambda env: Agent(lookahead=1, adapts=False),
    "greedy_cusum_adapts": lambda env: Agent(lookahead=1, adapts=True),
    "mpc_exhaustive_never_adapts": lambda env: Agent(lookahead=LOOKAHEAD, adapts=False, beam_width=None),
    "mpc_beam_never_adapts": lambda env: Agent(lookahead=LOOKAHEAD, adapts=False, beam_width=BEAM_WIDTH),
    "mpc_beam_cusum_adapts": lambda env: Agent(lookahead=LOOKAHEAD, adapts=True, beam_width=BEAM_WIDTH),
    "oracle_exhaustive": lambda env: OracleAgent(env, lookahead=LOOKAHEAD, beam_width=None),
    "oracle_beam": lambda env: OracleAgent(env, lookahead=LOOKAHEAD, beam_width=BEAM_WIDTH),
}


def run_episode(env: DelayedRegimeShiftControlEnv, agent, episode_idx: int, rng: random.Random) -> float:
    position, pending = env.reset()
    remaining = env.horizon
    for _ in range(env.horizon):
        global_step = env.global_step
        if episode_idx < WARMUP_EPISODES:
            action = rng.choice(ACTIONS)
        else:
            action = agent.choose_action(position, pending, remaining, rng)
        result = env.step(action)
        agent.observe(global_step, result.position, result.pending, action, result.next_position)
        position, pending = result.next_position, result.next_pending
        remaining -= 1
        if result.done:
            return result.reward
    raise AssertionError("episode did not terminate")  # pragma: no cover


def run_condition(agent_factory, seed: int) -> tuple[float, float]:
    env = DelayedRegimeShiftControlEnv(target=TARGET, lag_weight=LAG_WEIGHT, horizon=HORIZON,
                                        init_range=INIT_RANGE, noise_sigma=NOISE_SIGMA,
                                        regime_shift_step=REGIME_SHIFT_STEP, post_shift_scale=POST_SHIFT_SCALE, seed=seed)
    rng = random.Random(seed + 50_000)
    agent = agent_factory(env)
    pre_rewards, post_rewards = [], []
    for episode_idx in range(N_EPISODES):
        episode_start_step = env.global_step
        reward = run_episode(env, agent, episode_idx, rng)
        if episode_idx >= WARMUP_EPISODES:
            (pre_rewards if episode_start_step < REGIME_SHIFT_STEP else post_rewards).append(reward)
    return statistics.mean(pre_rewards), statistics.mean(post_rewards)


def main() -> None:
    print(f"{'condition':<24} {'pre_shift_reward':>17} {'post_shift_reward':>18}")
    for label, factory in CONDITIONS.items():
        pre_rewards, post_rewards = [], []
        for seed in SEEDS:
            pre_r, post_r = run_condition(factory, seed)
            pre_rewards.append(pre_r)
            post_rewards.append(post_r)
        print(f"{label:<24} {statistics.mean(pre_rewards):>17.4f} {statistics.mean(post_rewards):>18.4f}")


if __name__ == "__main__":
    main()
