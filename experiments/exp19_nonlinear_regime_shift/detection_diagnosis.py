"""Follow-up to experiment 19 -- what actually causes CUSUM detection to
degrade under `NonlinearDynamicsModel` (9/15 seeds detected vs. linear's
15/15, ~5.4x slower mean latency), given RESULTS.md already checked and
REFUTED the most obvious explanation (higher steady-state residual
noise)? Not a new numbered experiment -- this investigates an open
question experiment 19 itself left unresolved, the same pattern
experiment 8's Meek's-rules follow-up and experiment 13's
refit_interval_calibration.py already used.

Three hypotheses checked here, all against matched linear (experiment
13's mechanism) vs. nonlinear (experiment 19's) agents on IDENTICAL
seeds and severities:

1. **Steady-state residual noise** (RESULTS.md's original check,
   reproduced here as code instead of an ad-hoc one-off): pre-shift
   residual standard deviation. Already refuted; reproduced for the
   record.
2. **Residual autocorrelation**: CUSUM assumes roughly independent
   residuals around a stable mean -- serially correlated residuals
   (e.g. alternating sign) would make the cumulative-sum statistic
   accumulate more slowly toward threshold even at identical variance,
   which could explain slower detection without needing higher variance
   at all. Checked via lag-1 autocorrelation, pre-shift.
3. **Post-shift (pre-detection) residual variability across seeds**:
   not steady-state noise, but whether the STALE model's own action
   choices, once dynamics have actually shifted, interact with the
   environment more erratically for the nonlinear model than the linear
   one -- checked by comparing the SPREAD (not just the mean) of
   post-shift residual standard deviation across seeds.

Run: PYTHONPATH=. python experiments/exp19_nonlinear_regime_shift/detection_diagnosis.py
"""
from __future__ import annotations

import random
import statistics

from environments.transworld import RegimeShiftControlEnv
from experiments.exp13_regime_shift_world_model.run import CUSUMAdaptiveAgent as LinearCUSUMAgent
from experiments.exp19_nonlinear_regime_shift.run import (
    ACTIONS, GAMMA, NOISE_SIGMA, REGIME_SHIFT_TRIAL, STATE_RANGE, T0, TARGET, WARMUP_TRIALS,
    CUSUMAdaptiveAgent as NonlinearCUSUMAgent, NonlinearRegimeShiftControlEnv,
)

SEEDS = list(range(10))
POST_SHIFT_SCALE = -1.0  # sign_flip: the severity experiment 19 found the detection gap in
N_PRE_SHIFT_RESIDUALS = 1000  # trials 200-1200 of the pre-shift window, well past warmup


def _lag1_autocorr(xs: list[float]) -> float:
    n = len(xs)
    mean = statistics.mean(xs)
    num = sum((xs[i] - mean) * (xs[i + 1] - mean) for i in range(n - 1))
    den = sum((x - mean) ** 2 for x in xs)
    return num / den if den else 0.0


def pre_shift_residuals(seed: int, nonlinear: bool) -> list[float]:
    """Residuals from a NeverAdaptsAgent-like run (the CUSUM agent
    itself, before any detection occurs) over trials 200..1200 -- well
    into steady state, matched between the two model types."""
    agent_cls = NonlinearCUSUMAgent if nonlinear else LinearCUSUMAgent
    if nonlinear:
        env = NonlinearRegimeShiftControlEnv(target=TARGET, state_range=STATE_RANGE, noise_sigma=NOISE_SIGMA, gamma=GAMMA,
                                              regime_shift_trial=REGIME_SHIFT_TRIAL, post_shift_scale=POST_SHIFT_SCALE, seed=seed)
    else:
        env = RegimeShiftControlEnv(target=TARGET, state_range=STATE_RANGE, noise_sigma=NOISE_SIGMA,
                                     regime_shift_trial=REGIME_SHIFT_TRIAL, post_shift_scale=POST_SHIFT_SCALE, seed=seed)
    rng = random.Random(60_000 + seed)
    agent = agent_cls()
    for trial in range(200 + N_PRE_SHIFT_RESIDUALS):
        info = env.observe()
        action = rng.choice(ACTIONS) if trial < WARMUP_TRIALS else agent.choose_action(info.state, rng)
        info = env.step(action)
        agent.observe(trial, info.state, action, info.next_state)
    return [s.values["residual"] for s in agent.residual_states
            if 200 <= (s.timestamp - T0).total_seconds() < 200 + N_PRE_SHIFT_RESIDUALS]


def post_shift_residual_stdev(seed: int, nonlinear: bool, n_post: int = 200) -> float | None:
    """Standard deviation of the (agent's own, unmodified CUSUM
    machinery's) residual stream for the first `n_post` trials after
    the shift -- before any detection has necessarily occurred, using
    the agent's real greedy action selection, not a forced/uniform one."""
    agent_cls = NonlinearCUSUMAgent if nonlinear else LinearCUSUMAgent
    if nonlinear:
        env = NonlinearRegimeShiftControlEnv(target=TARGET, state_range=STATE_RANGE, noise_sigma=NOISE_SIGMA, gamma=GAMMA,
                                              regime_shift_trial=REGIME_SHIFT_TRIAL, post_shift_scale=POST_SHIFT_SCALE, seed=seed)
    else:
        env = RegimeShiftControlEnv(target=TARGET, state_range=STATE_RANGE, noise_sigma=NOISE_SIGMA,
                                     regime_shift_trial=REGIME_SHIFT_TRIAL, post_shift_scale=POST_SHIFT_SCALE, seed=seed)
    rng = random.Random(60_000 + seed)
    agent = agent_cls()
    for trial in range(REGIME_SHIFT_TRIAL + n_post):
        info = env.observe()
        action = rng.choice(ACTIONS) if trial < WARMUP_TRIALS else agent.choose_action(info.state, rng)
        info = env.step(action)
        agent.observe(trial, info.state, action, info.next_state)
    post = [s.values["residual"] for s in agent.residual_states if (s.timestamp - T0).total_seconds() >= REGIME_SHIFT_TRIAL]
    return statistics.pstdev(post) if len(post) > 1 else None


def main() -> None:
    print("=== Hypothesis 1: steady-state (pre-shift) residual noise ===")
    nl_stdevs, li_stdevs = [], []
    nl_autocorrs, li_autocorrs = [], []
    for seed in SEEDS:
        nl_res = pre_shift_residuals(seed, nonlinear=True)
        li_res = pre_shift_residuals(seed, nonlinear=False)
        nl_stdevs.append(statistics.pstdev(nl_res))
        li_stdevs.append(statistics.pstdev(li_res))
        nl_autocorrs.append(_lag1_autocorr(nl_res))
        li_autocorrs.append(_lag1_autocorr(li_res))
    print(f"nonlinear pre-shift residual stdev: mean={statistics.mean(nl_stdevs):.4f} (n={len(SEEDS)} seeds)")
    print(f"linear    pre-shift residual stdev: mean={statistics.mean(li_stdevs):.4f} (n={len(SEEDS)} seeds)")

    print()
    print("=== Hypothesis 2: residual autocorrelation (pre-shift, lag-1) ===")
    print(f"nonlinear lag-1 autocorrelation: mean={statistics.mean(nl_autocorrs):.4f}")
    print(f"linear    lag-1 autocorrelation: mean={statistics.mean(li_autocorrs):.4f}")

    print()
    print("=== Hypothesis 3: post-shift (pre-detection) residual variability ACROSS seeds ===")
    nl_post_stdevs = [s for s in (post_shift_residual_stdev(seed, True) for seed in SEEDS) if s is not None]
    li_post_stdevs = [s for s in (post_shift_residual_stdev(seed, False) for seed in SEEDS) if s is not None]
    print(f"nonlinear post-shift residual stdev across seeds: {[round(s, 3) for s in nl_post_stdevs]}")
    print(f"  mean={statistics.mean(nl_post_stdevs):.4f}  spread (max - min)={max(nl_post_stdevs) - min(nl_post_stdevs):.4f}")
    print(f"linear    post-shift residual stdev across seeds: {[round(s, 3) for s in li_post_stdevs]}")
    print(f"  mean={statistics.mean(li_post_stdevs):.4f}  spread (max - min)={max(li_post_stdevs) - min(li_post_stdevs):.4f}")


if __name__ == "__main__":
    main()
