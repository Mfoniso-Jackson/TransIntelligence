"""Regression tests for Experiment 4 (docs/research-agenda.md #7a,
experiments/exp04_frame_discovery/RESULTS.md). Small-scale versions of the
real experiment (fewer steps/seeds, for test-suite speed).
"""
import pytest

from transintelligence import ReferenceFrame

from environments.transworld import FrameSwitchEnv
from experiments.exp01_frame_conditioning.held_out_frame import ALL_FRAMES, HELD_OUT_INDEX, TRAIN_FRAMES
from experiments.exp04_frame_discovery.agents import DiscoveringRFAgent, RandomDiscoveryAgent, _binom_cdf, _fit_frame


def test_binom_cdf_matches_hand_computed_cases():
    # A fair coin: P(0 heads in 1 flip) = 0.5
    assert _binom_cdf(0, 1, 0.5) == pytest.approx(0.5)
    # P(at most n-1 successes in n trials) = 1 - P(all succeed) = 1 - p^n
    assert _binom_cdf(2, 3, 0.9) == pytest.approx(1 - 0.9 ** 3)
    # P(at most n successes in n trials) = 1 always (X can't exceed n)
    assert _binom_cdf(3, 3, 0.9) == pytest.approx(1.0)
    # CDF is monotonically non-decreasing in k
    vals = [_binom_cdf(k, 10, 0.85) for k in range(11)]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    assert vals[-1] == pytest.approx(1.0)


def test_fit_frame_recovers_known_frame_from_noiseless_data():
    true_baseline, true_direction = 0.7, "lower_is_better"

    def true_conclusion(raw: float) -> int:
        score = raw - true_baseline
        if true_direction == "lower_is_better":
            score = -score
        return 1 if score > 0 else -1

    raws = [i / 100 for i in range(0, 101, 2)]
    labels = [true_conclusion(r) for r in raws]
    fitted = _fit_frame(raws, labels, name="test")
    assert fitted.baseline == pytest.approx(true_baseline)
    assert fitted.metadata["direction"] == true_direction


def test_discovering_rf_recovers_the_true_held_out_frame_small_scale():
    """Small-scale check that the discovery mechanism, once triggered,
    fits a frame close to the genuine held-out frame -- not just any
    frame. The real run (RESULTS.md) confirms this at full scale across
    10 seeds; this locks in the mechanism doesn't regress."""
    # 3000 steps, not fewer: per-window detection power is only ~21% (see
    # RESULTS.md), so this needs enough occurrences of the held-out regime
    # for at least one to trigger. Still runs in well under a second.
    agent = DiscoveringRFAgent(TRAIN_FRAMES, window=40, min_accuracy=0.85, alpha=0.01, seed=0)
    env = FrameSwitchEnv(ALL_FRAMES, switch_period=40, jitter=10, noise_sigma=0.05, seed=0)
    for _ in range(3000):
        info = env.observe()
        is_held_out = info.active_frame_index == HELD_OUT_INDEX
        reward = env.feedback(agent.act(info.raw))
        agent.update(reward, true_is_held_out=is_held_out)

    true_frame = ALL_FRAMES[HELD_OUT_INDEX]
    discovered = agent.inner.frames[len(TRAIN_FRAMES):]
    assert discovered, "agent never discovered a new frame"
    closest = min(discovered, key=lambda f: abs(f.baseline - true_frame.baseline))
    assert abs(closest.baseline - true_frame.baseline) < 0.15
    assert closest.metadata["direction"] == true_frame.metadata["direction"]


def test_discovering_beats_random_discovery_after_first_discovery():
    """The confound-control comparison from docs/research-agenda.md #7a:
    fitted discovery should recover held-out accuracy by more than random
    discovery does, once both have discovered something."""
    def run(agent_cls, seed: int, n_steps: int = 3000) -> tuple[float, int | None]:
        agent = agent_cls(TRAIN_FRAMES, window=40, min_accuracy=0.85, alpha=0.01, seed=seed)
        env = FrameSwitchEnv(ALL_FRAMES, switch_period=40, jitter=10, noise_sigma=0.05, seed=seed)
        after_rewards = []
        first_step = None
        for step in range(n_steps):
            info = env.observe()
            is_held_out = info.active_frame_index == HELD_OUT_INDEX
            reward = env.feedback(agent.act(info.raw))
            agent.update(reward, true_is_held_out=is_held_out)
            if first_step is None:
                first_step = agent.first_acted_discovery_step
            if is_held_out and first_step is not None and step >= first_step:
                after_rewards.append(reward)
        acc = sum(after_rewards) / len(after_rewards) if after_rewards else float("nan")
        return acc, first_step

    compared = 0
    for seed in (0, 1, 2, 3, 4):
        fitted_acc, fitted_step = run(DiscoveringRFAgent, seed)
        random_acc, random_step = run(RandomDiscoveryAgent, seed)
        if fitted_step is None or random_step is None:
            continue  # no discovery event for one of the agents on this seed -- skip, don't fail
        compared += 1
        assert fitted_acc > random_acc, f"seed={seed}: fitted={fitted_acc} did not beat random={random_acc}"
    assert compared > 0, "no seed produced a discovery event for both agents -- test is vacuous, investigate"


def test_no_false_discoveries_when_no_novel_regime_is_present():
    """Specificity check: if the environment never introduces a regime
    outside the agent's known frames, the trigger should essentially never
    fire (false-positive rate was ~4.6e-5/window by design, see RESULTS.md)."""
    agent = DiscoveringRFAgent(ALL_FRAMES, window=40, min_accuracy=0.85, alpha=0.01, seed=0)
    env = FrameSwitchEnv(ALL_FRAMES, switch_period=40, jitter=10, noise_sigma=0.05, seed=0)
    for _ in range(2000):
        info = env.observe()
        reward = env.feedback(agent.act(info.raw))
        agent.update(reward, true_is_held_out=False)
    assert len(agent.trigger_log) == 0


def test_sweep_noise_run_seed_returns_expected_shape_small_scale():
    """Smoke test for sweep_noise.py's machinery, not the full 10-seed
    sweep across 6 noise levels in RESULTS.md (~40s) -- confirms run_seed
    wires together correctly at reduced scale."""
    import experiments.exp04_frame_discovery.sweep_noise as sweep_noise

    original_n_steps = sweep_noise.N_STEPS
    sweep_noise.N_STEPS = 500
    try:
        result = sweep_noise.run_seed(DiscoveringRFAgent, seed=0, noise_sigma=0.05)
    finally:
        sweep_noise.N_STEPS = original_n_steps

    for key in ("discovered", "acc_before", "acc_after", "n_triggers", "n_false_triggers"):
        assert key in result


def test_two_missing_regimes_matches_helper():
    """_matches() is the post-hoc classifier deciding whether a discovered
    frame corresponds to a specific held-out regime -- lock in its
    tolerance behavior directly rather than only through a full run."""
    from experiments.exp04_frame_discovery.two_missing_regimes import MATCH_TOLERANCE, _matches

    true_frame = ReferenceFrame("f4", baseline=0.7, metadata={"direction": "lower_is_better"})
    close = ReferenceFrame("discovered", baseline=0.7 + MATCH_TOLERANCE / 2, metadata={"direction": "lower_is_better"})
    far = ReferenceFrame("discovered", baseline=0.7 + MATCH_TOLERANCE * 2, metadata={"direction": "lower_is_better"})
    wrong_direction = ReferenceFrame("discovered", baseline=0.7, metadata={"direction": "higher_is_better"})

    assert _matches(close, true_frame)
    assert not _matches(far, true_frame)
    assert not _matches(wrong_direction, true_frame)


def test_two_missing_regimes_run_seed_returns_expected_shape_small_scale():
    """Smoke test for two_missing_regimes.py's machinery, not the full
    10-seed/4500-step run in RESULTS.md -- confirms the per-held-out-frame
    bookkeeping wires together correctly at reduced scale."""
    import experiments.exp04_frame_discovery.two_missing_regimes as two_missing

    original_n_steps = two_missing.N_STEPS
    two_missing.N_STEPS = 500
    try:
        result = two_missing.run_seed(seed=0)
    finally:
        two_missing.N_STEPS = original_n_steps

    for idx in two_missing.HELD_OUT_INDICES:
        assert idx in result
        for key in ("discovered", "acc_before", "acc_after"):
            assert key in result[idx]
    assert "final_frame_count" in result and "n_triggers" in result
