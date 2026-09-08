"""Experiment 19 -- does CUSUM-triggered regime-change detection still
compose correctly with a learned dynamics model when that model is
NONLINEAR, or was experiment 13's success specific to a linear one?
(docs/research-agenda.md #7p, Phase 6, continued)

The last of Phase 6's remaining-gaps list from after experiment 14
("extending regime-adaptation to the nonlinear dynamics model
[experiment 15]") that hadn't been tested -- experiment 16 tested
"regime detection + multi-step planning"; this is the other named
combination. Nothing new is implemented -- `CUSUMTemporalReasoner`
(Phase 4) and `NonlinearDynamicsModel` (generalized from experiment 15
into `transintelligence/world_models/`) are both reused exactly as
already verified. This experiment is otherwise a direct, minimal
substitution into experiment 13's own design: identical structure, same
severity sweep, same four conditions, same CUSUM detection logic --
`LinearDynamicsModel` swapped for `NonlinearDynamicsModel` throughout,
and `RegimeShiftControlEnv` swapped for the new
`NonlinearRegimeShiftControlEnv` (`environments/transworld/`), which adds
experiment 15's quadratic restoring force on top of experiment 13's
silent actuator rescaling.

## Why this isn't a foregone conclusion

`CUSUMTemporalReasoner` monitors a model's own PREDICTION RESIDUALS, not
the raw state -- it doesn't know or care whether the model producing
those residuals is linear or nonlinear. But a nonlinear model has one
more coefficient to estimate from the same refit window, and its
residual behavior near the restoring force's curvature could plausibly
be noisier or more state-dependent than a linear model's -- there's no
a priori guarantee detection calibrated correctly for one generalizes
to the other. GAMMA=0.3 (experiment 15's dramatic value, not the
GAMMA=0.05 value experiment 15 found too weak to matter) keeps the
nonlinearity a real, consequential feature of the environment, not
throttled down to where the linear-vs-nonlinear question would be moot.

## The confound this needed to control for

Reused directly from experiment 13: `sliding_window_baseline` always
trains on only the most recent transitions, regardless of detection --
if `cusum_detects_and_adapts` doesn't beat it, explicit change detection
isn't earning its complexity over a much simpler recency heuristic, even
with a nonlinear model.

Four conditions, all sharing the identical greedy 1-step-lookahead
policy and the identical `NonlinearDynamicsModel` fitting tool:

- `never_adapts`: trains on every transition ever observed.
- `oracle_adapts`: given the true regime-shift trial exactly -- the
  ceiling, isolating the cost of detection delay.
- `sliding_window_baseline`: always trains on only the most recent
  transitions -- the confound control.
- `cusum_detects_and_adapts`: the treatment condition.

Run: PYTHONPATH=. python experiments/exp19_nonlinear_regime_shift/run.py
"""
from __future__ import annotations

import random
import statistics
from datetime import datetime, timedelta, timezone

from environments.transworld import NUDGES, NonlinearRegimeShiftControlEnv
from transintelligence.core.states import State, StateHistory
from transintelligence.reasoning.temporal import CUSUMTemporalReasoner
from transintelligence.world_models import NonlinearDynamicsModel

ACTIONS = list(NUDGES.keys())
TARGET = 0.0
STATE_RANGE = (-5.0, 5.0)
NOISE_SIGMA = 0.3
GAMMA = 0.3  # experiment 15's dramatic value -- 0.05 was too weak to change the optimal action
REGIME_SHIFT_TRIAL = 1500
WARMUP_TRIALS = 60
REFIT_INTERVAL = 20
CUSUM_CHECK_INTERVAL = 20
SLIDING_WINDOW_SIZE = 200
N_TRIALS = 3000
SEEDS = list(range(15))
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

SEVERITIES = {
    "mild_attenuation": 0.4,   # nudges weaker but same direction -- experiment 13's honest null/boundary case
    "sign_flip": -1.0,         # nudges reverse direction entirely -- where adaptation should matter
}


class BaseAgent:
    """Identical bookkeeping to experiment 13's `BaseAgent`, with
    `NonlinearDynamicsModel` in place of `LinearDynamicsModel` -- the
    only change; subclasses differ only in which transitions
    `_training_transitions()` selects, never in how they act or fit."""

    def __init__(self, refit_interval: int = REFIT_INTERVAL) -> None:
        self.all_transitions: list[tuple[int, float, str, float]] = []  # (trial, state, action, next_state)
        self.model = NonlinearDynamicsModel()
        self.refit_interval = refit_interval

    def _training_transitions(self) -> list[tuple[int, float, str, float]]:
        raise NotImplementedError

    def choose_action(self, state: float, rng: random.Random) -> str:
        known = self.model.known_actions()
        if not known:
            return rng.choice(ACTIONS)
        return min(known, key=lambda a: (self.model.predict(state, a) - TARGET) ** 2)

    def observe(self, trial: int, state: float, action: str, next_state: float) -> None:
        self.all_transitions.append((trial, state, action, next_state))
        if len(self.all_transitions) % self.refit_interval == 0:
            self._refit()

    def _refit(self) -> None:
        transitions = [(s, a, ns) for (_, s, a, ns) in self._training_transitions()]
        self.model = NonlinearDynamicsModel.fit(transitions)


class NeverAdaptsAgent(BaseAgent):
    def _training_transitions(self):
        return self.all_transitions


class OracleAdaptsAgent(BaseAgent):
    def __init__(self, regime_shift_trial: int, refit_interval: int = REFIT_INTERVAL) -> None:
        super().__init__(refit_interval=refit_interval)
        self.regime_shift_trial = regime_shift_trial

    def _training_transitions(self):
        latest_trial = self.all_transitions[-1][0] if self.all_transitions else 0
        cutoff = self.regime_shift_trial if latest_trial >= self.regime_shift_trial else 0
        return [t for t in self.all_transitions if t[0] >= cutoff]


class SlidingWindowAgent(BaseAgent):
    def _training_transitions(self):
        return self.all_transitions[-SLIDING_WINDOW_SIZE:]


class CUSUMAdaptiveAgent(BaseAgent):
    def __init__(self, refit_interval: int = REFIT_INTERVAL) -> None:
        super().__init__(refit_interval=refit_interval)
        self.residual_states: list[State] = []
        self.reset_from_trial = 0
        self.reasoner = CUSUMTemporalReasoner()  # default calibration, unchanged from experiments 5/13
        self.reset_trials: list[int] = []

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
    env = NonlinearRegimeShiftControlEnv(target=TARGET, state_range=STATE_RANGE, noise_sigma=NOISE_SIGMA, gamma=GAMMA,
                                          regime_shift_trial=REGIME_SHIFT_TRIAL, post_shift_scale=post_shift_scale, seed=seed)
    rng = random.Random(seed + 60_000)
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
