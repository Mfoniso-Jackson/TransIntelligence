"""Experiment 18 -- does `CalibrationVerifier`, applied to experiment
16's own agents, find direct evidence for its central unconfirmed
hypothesis? (docs/research-agenda.md #7o, closing Phase 6/opening
meta-intelligence)

Experiment 16 found `mpc_beam_cusum_adapts` persistently underperforms
`greedy_cusum_adapts` post-shift, and refuted the "reduced exploration"
explanation directly. The best-supported remaining explanation was
stated as a HYPOTHESIS, not confirmed: multi-step lookahead chains two
predictions from the same learned dynamics model, and each learned
prediction carries some estimation error that never fully vanishes --
chaining compounds it in a way single-step lookahead never pays for.
That hypothesis was never checked against the actual estimation error
of either agent's dynamics model -- only inferred from the pattern of
final rewards.

This experiment checks it directly. Both agents fit the SAME kind of
one-step dynamics model via the SAME OLS procedure (`Agent._predict`,
`Agent._refit`) -- the only difference is how many steps ahead each
SEARCHES when choosing an action, not how the model itself is fit. But
each agent's planning strategy steers it into different regions of state
space (experiment 16 already measured mpc_beam's post-shift visited
positions have a LARGER spread than greedy's), which feeds back into
what data each agent's OWN model gets trained on. `CalibrationVerifier`
(Kupiec 1995), applied to each agent's own one-step prediction residuals
against the environment's TRUE noise floor (`NOISE_SIGMA` -- what a
correctly-specified, well-fit model's residuals should look like), asks:
does `mpc_beam_cusum_adapts`'s dynamics model carry MORE excess
estimation error beyond the irreducible noise floor than
`greedy_cusum_adapts`'s does, post-shift? Either answer is informative:
a real difference would be a mechanistic, MEASURED explanation
(self-steered training distribution degrades the model itself);  no
difference would instead support the compounding-across-steps
explanation experiment 16 already proposed, since the single-step model
quality alone wouldn't explain the persistent multi-step gap.

## The confound this needs to control for

Pre-shift residuals, where both agents have plenty of stable data and no
regime confusion, are checked as a baseline in the SAME run: if pre-shift
residuals were ALSO miscalibrated for both agents, that would point to a
baseline model-noise mismatch unrelated to the regime shift, not
something specific to post-shift adaptation -- and the post-shift result
would need a different interpretation.

Run: PYTHONPATH=. python experiments/exp18_calibration_verifier/run.py
"""
from __future__ import annotations

from experiments.exp16_regime_shift_multistep_planning.run import (
    BEAM_WIDTH, HORIZON, INIT_RANGE, LAG_WEIGHT, LOOKAHEAD, N_EPISODES,
    NOISE_SIGMA, POST_SHIFT_SCALE, REGIME_SHIFT_STEP, TARGET,
    Agent, DelayedRegimeShiftControlEnv, run_episode,
)
import random

from transintelligence.verification import CalibrationVerifier

SEEDS = list(range(12))  # the same seeds experiments 16 and 17 used
THRESHOLD_SIGMAS = 1.0


class InstrumentedAgent(Agent):
    """Adds a persistent, untrimmed log of every one-step prediction
    residual `Agent.observe` already computes internally -- exactly the
    same value (`next_position - self._predict(position, pending,
    action)`, using the coefficients as fit BEFORE this transition is
    incorporated), but experiment 16's own `residual_states` gets
    periodically trimmed by CUSUM-triggered resets (by design, for
    detection), discarding exactly the history this experiment needs to
    see in full. No changes to `Agent` itself -- this only adds an
    external hook, reusing `_predict`/`coefficients` unchanged."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.full_residual_log: list[tuple[int, float]] = []

    def observe(self, global_step: int, position: float, pending: float, action: str, next_position: float) -> None:
        if action in self.coefficients:
            predicted = self._predict(position, pending, action)
            self.full_residual_log.append((global_step, next_position - predicted))
        super().observe(global_step, position, pending, action, next_position)


def train_instrumented_agent(lookahead: int, beam_width: int | None, seed: int) -> InstrumentedAgent:
    env = DelayedRegimeShiftControlEnv(target=TARGET, lag_weight=LAG_WEIGHT, horizon=HORIZON,
                                        init_range=INIT_RANGE, noise_sigma=NOISE_SIGMA,
                                        regime_shift_step=REGIME_SHIFT_STEP, post_shift_scale=POST_SHIFT_SCALE, seed=seed)
    rng = random.Random(seed + 50_000)
    agent = InstrumentedAgent(lookahead=lookahead, adapts=True, beam_width=beam_width)
    for episode_idx in range(N_EPISODES):
        run_episode(env, agent, episode_idx, rng)
    return agent


def pooled_residuals(agents: list[InstrumentedAgent], post_shift: bool) -> list[float]:
    residuals = []
    for agent in agents:
        for global_step, residual in agent.full_residual_log:
            if (global_step >= REGIME_SHIFT_STEP) == post_shift:
                residuals.append(residual)
    return residuals


def main() -> None:
    greedy_agents = [train_instrumented_agent(lookahead=1, beam_width=None, seed=s) for s in SEEDS]
    mpc_agents = [train_instrumented_agent(lookahead=LOOKAHEAD, beam_width=BEAM_WIDTH, seed=s) for s in SEEDS]

    verifier = CalibrationVerifier(threshold_sigmas=THRESHOLD_SIGMAS)
    print(f"{'agent':<24} {'window':<10} {'n':>6} {'observed_cov':>13} {'claimed_cov':>12} {'p_value':>10} {'status':>16}")
    for label, agents in (("greedy_cusum_adapts", greedy_agents), ("mpc_beam_cusum_adapts", mpc_agents)):
        for window_label, post_shift in (("pre_shift", False), ("post_shift", True)):
            residuals = pooled_residuals(agents, post_shift)
            result = verifier.verify(residuals, claimed_sigma=NOISE_SIGMA)
            observed = f"{result.observed_coverage:.4f}" if result.observed_coverage is not None else "n/a"
            pval = f"{result.p_value:.6f}" if result.p_value is not None else "n/a"
            print(f"{label:<24} {window_label:<10} {result.n_observations:>6} {observed:>13} "
                  f"{result.claimed_coverage:>12.4f} {pval:>10} {result.status.value:>16}")


if __name__ == "__main__":
    main()
