"""Experiment 13 -- does an agent that detects a mid-experiment shift in
the environment's true dynamics (via `CUSUMTemporalReasoner`, Phase 4)
and discards its stale pre-shift data recover performance an agent that
never adapts loses -- and does it do so because it detected something
real, not just because favoring recent data helps regardless?
(docs/research-agenda.md #7j, Phase 6, continued)

This is a synthesis experiment: it combines two already-built,
already-verified kernel primitives (`CUSUMTemporalReasoner` from Phase 4,
`LinearDynamicsModel` from Phase 6) rather than introducing a new
mechanism. The empirical question is whether they compose as expected,
not whether either one individually works -- that was already
established in experiments 5 and 11.

`RegimeShiftControlEnv` (`environments/transworld/regime_shift_control_env.py`)
is `ResourceControlEnv` (experiment 11) with one difference: at an
unknown trial (`REGIME_SHIFT_TRIAL`), the nudge magnitudes silently
rescale by `post_shift_scale` -- a dynamics model fit on pre-shift data
becomes systematically wrong afterward, without the agent being told.

## A severity sweep, not one fixed shift -- an honest finding this
## experiment's own first run forced, not the original plan

The first version of this experiment used only a single, mild shift
(`post_shift_scale=0.4`, nudges attenuated but not reversed) and found
that adaptation barely mattered at all: `oracle_adapts` (given the true
shift trial exactly, zero detection delay) performed statistically
indistinguishably from `never_adapts`. Investigated rather than
reported as-is: with this environment's wide state range relative to its
nudge magnitudes, most trials start far enough from target that both a
stale and a correctly-calibrated model pick the *same* largest-available
nudge regardless of the exact scale factor -- mild miscalibration rarely
changes which action looks best. `SEVERITIES` below keeps that finding
(`mild_attenuation`) as a real, honestly-reported boundary case, and
adds `sign_flip` (`post_shift_scale=-1.0`, the actuator's effect
reverses direction entirely) -- a shift severe enough that the stale
model's choices become actively counterproductive, not merely
suboptimal, which is where adaptation's value should show up clearly if
the mechanism works at all.

## The confound this needed to control for

Showing CUSUM-triggered adaptation beat an agent that never adapts at
all would only prove that *some* adaptation helps -- the same trivial
"any extra information helps a little" trap experiments 3, 8, and 11 each
had to control for. `sliding_window_baseline` is the real test: it always
trains on only the most recent `SLIDING_WINDOW_SIZE` transitions,
regardless of whether anything was actually detected. If
`cusum_detects_and_adapts` doesn't beat it, explicit change detection
isn't earning its complexity over a much simpler recency heuristic.

Four conditions, all sharing the identical greedy 1-step-lookahead
policy and the identical `LinearDynamicsModel` fitting tool -- they
differ ONLY in which transitions each one trains on:

- `never_adapts`: trains on every transition ever observed.
- `oracle_adapts`: given `REGIME_SHIFT_TRIAL` exactly, discards
  everything before it the moment that trial is reached -- the ceiling,
  isolating the cost of detection *delay* specifically.
- `sliding_window_baseline`: always trains on only the most recent
  transitions -- the confound control described above.
- `cusum_detects_and_adapts`: tracks its own model's prediction
  residuals as a `StateHistory`, periodically calls
  `CUSUMTemporalReasoner.change_points()` on that residual series
  (default calibration, unchanged from experiment 5), and discards
  pre-detection data once a shift is found.

Run: PYTHONPATH=. python experiments/exp13_regime_shift_world_model/run.py
"""
from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta, timezone

from environments.transworld import NUDGES, RegimeShiftControlEnv
from transintelligence.core.states import State, StateHistory
from transintelligence.reasoning.temporal import CUSUMTemporalReasoner
from transintelligence.world_models import LinearDynamicsModel

ACTIONS = list(NUDGES.keys())
TARGET = 0.0
STATE_RANGE = (-5.0, 5.0)
NOISE_SIGMA = 0.3
REGIME_SHIFT_TRIAL = 1500
WARMUP_TRIALS = 60
REFIT_INTERVAL = 20
CUSUM_CHECK_INTERVAL = 20
SLIDING_WINDOW_SIZE = 200
N_TRIALS = 3000
SEEDS = list(range(15))
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

SEVERITIES = {
    "mild_attenuation": 0.4,   # nudges weaker but same direction -- the honest null/boundary case
    "sign_flip": -1.0,         # nudges reverse direction entirely -- where adaptation should matter
}


class BaseAgent:
    """Shared bookkeeping: log every transition with its trial index,
    periodically refit a `LinearDynamicsModel` on whatever subset
    `_training_transitions()` selects. Subclasses differ only in that
    selection, never in how they act or fit."""

    def __init__(self) -> None:
        self.all_transitions: list[tuple[int, float, str, float]] = []  # (trial, state, action, next_state)
        self.model = LinearDynamicsModel()

    def _training_transitions(self) -> list[tuple[int, float, str, float]]:
        raise NotImplementedError

    def choose_action(self, state: float, rng: random.Random) -> str:
        known = self.model.known_actions()
        if not known:
            return rng.choice(ACTIONS)
        return min(known, key=lambda a: (self.model.predict(state, a) - TARGET) ** 2)

    def observe(self, trial: int, state: float, action: str, next_state: float) -> None:
        self.all_transitions.append((trial, state, action, next_state))
        if len(self.all_transitions) % REFIT_INTERVAL == 0:
            self._refit()

    def _refit(self) -> None:
        transitions = [(s, a, ns) for (_, s, a, ns) in self._training_transitions()]
        self.model = LinearDynamicsModel.fit(transitions)


class NeverAdaptsAgent(BaseAgent):
    def _training_transitions(self):
        return self.all_transitions


class OracleAdaptsAgent(BaseAgent):
    """Given the TRUE regime-shift trial exactly -- the ceiling,
    isolating the cost of detection delay from the cost of adapting at
    all."""

    def __init__(self, regime_shift_trial: int) -> None:
        super().__init__()
        self.regime_shift_trial = regime_shift_trial

    def _training_transitions(self):
        latest_trial = self.all_transitions[-1][0] if self.all_transitions else 0
        cutoff = self.regime_shift_trial if latest_trial >= self.regime_shift_trial else 0
        return [t for t in self.all_transitions if t[0] >= cutoff]


class SlidingWindowAgent(BaseAgent):
    """Always trains on only the most recent SLIDING_WINDOW_SIZE
    transitions -- the confound control: does explicit detection beat
    just always favoring recent data?"""

    def _training_transitions(self):
        return self.all_transitions[-SLIDING_WINDOW_SIZE:]


class CUSUMAdaptiveAgent(BaseAgent):
    """Tracks its own model's prediction residuals as a StateHistory,
    periodically checks CUSUMTemporalReasoner.change_points() for a
    shift in that series, and discards pre-detection data once found --
    the treatment condition."""

    def __init__(self) -> None:
        super().__init__()
        self.residual_states: list[State] = []
        self.reset_from_trial = 0
        self.reasoner = CUSUMTemporalReasoner()  # default calibration, unchanged from experiment 5
        self.reset_trials: list[int] = []  # every detected reset point, chronological

    def _training_transitions(self):
        return [t for t in self.all_transitions if t[0] >= self.reset_from_trial]

    def observe(self, trial: int, state: float, action: str, next_state: float) -> None:
        if action in self.model.known_actions():
            predicted = self.model.predict(state, action)
            residual = next_state - predicted
            self.residual_states.append(State("world_model_residual", {"residual": residual}, T0 + timedelta(seconds=trial)))
        super().observe(trial, state, action, next_state)
        if trial > 0 and trial % CUSUM_CHECK_INTERVAL == 0 and len(self.residual_states) >= self.reasoner.burn_in + 5:
            history = StateHistory(self.residual_states)
            change_points = self.reasoner.change_points(history, "residual")
            if change_points:
                latest_cp_trial = int(max((cp - T0).total_seconds() for cp in change_points))
                if latest_cp_trial > self.reset_from_trial:
                    self.reset_from_trial = latest_cp_trial
                    self.reset_trials.append(latest_cp_trial)
                    self._refit()
                    self.residual_states = [s for s in self.residual_states if s.timestamp >= T0 + timedelta(seconds=latest_cp_trial)]


def run_condition(agent_factory, seed: int, post_shift_scale: float) -> tuple[float, float, list[int]]:
    env = RegimeShiftControlEnv(target=TARGET, state_range=STATE_RANGE, noise_sigma=NOISE_SIGMA,
                                 regime_shift_trial=REGIME_SHIFT_TRIAL, post_shift_scale=post_shift_scale, seed=seed)
    rng = random.Random(seed + 30_000)
    agent = agent_factory()
    pre_shift_rewards, post_shift_rewards = [], []
    for trial in range(N_TRIALS):
        info = env.observe()
        if trial < WARMUP_TRIALS:
            action = rng.choice(ACTIONS)
        else:
            action = agent.choose_action(info.state, rng)
        info = env.step(action)
        agent.observe(trial, info.state, action, info.next_state)
        if trial >= WARMUP_TRIALS:
            (pre_shift_rewards if trial < REGIME_SHIFT_TRIAL else post_shift_rewards).append(info.reward)
    reset_trials = getattr(agent, "reset_trials", [])
    return statistics.mean(pre_shift_rewards), statistics.mean(post_shift_rewards), reset_trials


CONDITIONS = {
    "never_adapts": lambda: NeverAdaptsAgent(),
    "oracle_adapts": lambda: OracleAdaptsAgent(REGIME_SHIFT_TRIAL),
    "sliding_window_baseline": lambda: SlidingWindowAgent(),
    "cusum_detects_and_adapts": lambda: CUSUMAdaptiveAgent(),
}


def main() -> None:
    for severity_label, post_shift_scale in SEVERITIES.items():
        print(f"=== {severity_label} (post_shift_scale={post_shift_scale}) ===")
        print(f"{'condition':<25} {'pre_shift_reward':>17} {'post_shift_reward':>18}")
        for label, factory in CONDITIONS.items():
            pre_rewards, post_rewards = [], []
            for seed in SEEDS:
                pre_r, post_r, _ = run_condition(factory, seed, post_shift_scale)
                pre_rewards.append(pre_r)
                post_rewards.append(post_r)
            print(f"{label:<25} {statistics.mean(pre_rewards):>17.4f} {statistics.mean(post_rewards):>18.4f}")

        print("  --- cusum_detects_and_adapts: detection behavior ---")
        seeds_with_false_positive = 0
        latencies = []
        for seed in SEEDS:
            _, _, reset_trials = run_condition(CONDITIONS["cusum_detects_and_adapts"], seed, post_shift_scale)
            if any(t < REGIME_SHIFT_TRIAL for t in reset_trials):
                seeds_with_false_positive += 1
            post_shift_resets = [t for t in reset_trials if t >= REGIME_SHIFT_TRIAL]
            if post_shift_resets:
                latencies.append(post_shift_resets[0] - REGIME_SHIFT_TRIAL)
        print(f"  seeds with a detection before the true shift (false positive): {seeds_with_false_positive}/{len(SEEDS)}")
        print(f"  seeds that detected the true shift at all: {len(latencies)}/{len(SEEDS)}")
        if latencies:
            print(f"  mean detection latency (trials after the true shift): {statistics.mean(latencies):.1f}")
        print()


if __name__ == "__main__":
    main()
