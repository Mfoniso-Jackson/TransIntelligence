"""Smoke/regression test for Experiment 1 (docs/research-agenda.md #5,
experiments/exp01_frame_conditioning/RESULTS.md). Runs a small-scale version
of the real experiment (fewer seeds/steps, for test-suite speed) and checks
the qualitative finding holds: RFAwareAgent beats FlatBaselineAgent when
neither knows which frame is active. This is not a substitute for the full
run in RESULTS.md (10 seeds x 3000 steps) -- it's a fast regression guard
against a future change accidentally breaking the mechanism.
"""
import pytest

from transintelligence import ReferenceFrame

from environments.transworld import FrameSwitchEnv
from experiments.exp01_frame_conditioning.agents import FlatBaselineAgent, LearnedEmbeddingAgent, RFAwareAgent, TrueOracleAgent
from experiments.exp01_frame_conditioning.rnn_agent import RNNEmbeddingAgent
from experiments.exp01_frame_conditioning.run import SeedLog, _windows, brier_score
from experiments.exp01_frame_conditioning.sweep import run_grid_point

FRAMES = [
    ReferenceFrame("f1", baseline=0.5, metadata={"direction": "higher_is_better"}),
    ReferenceFrame("f2", baseline=0.3, metadata={"direction": "higher_is_better"}),
    ReferenceFrame("f3", baseline=0.5, metadata={"direction": "lower_is_better"}),
    ReferenceFrame("f4", baseline=0.7, metadata={"direction": "lower_is_better"}),
]


def _run(agent, seed: int, n_steps: int = 400) -> float:
    env = FrameSwitchEnv(FRAMES, switch_period=40, jitter=10, noise_sigma=0.05, seed=seed)
    correct = 0
    for _ in range(n_steps):
        info = env.observe()
        predict = agent.act(info.raw)
        reward = env.feedback(predict)
        agent.update(reward)
        correct += reward
    return correct / n_steps


def test_rf_aware_beats_flat_baseline_on_matched_information():
    for seed in (0, 1, 2):
        flat_acc = _run(FlatBaselineAgent(), seed)
        rf_acc = _run(RFAwareAgent(FRAMES), seed)
        assert rf_acc > flat_acc, f"seed={seed}: rf_aware={rf_acc} did not beat flat={flat_acc}"


def test_env_switches_and_agents_produce_valid_actions():
    env = FrameSwitchEnv(FRAMES, switch_period=5, jitter=1, noise_sigma=0.0, seed=0)
    agent = RFAwareAgent(FRAMES)
    saw_switch = False
    for _ in range(50):
        info = env.observe()
        saw_switch = saw_switch or info.switched
        predict = agent.act(info.raw)
        assert predict in (1, -1)
        reward = env.feedback(predict)
        assert reward in (0, 1)
        agent.update(reward)
        assert abs(sum(agent.belief) - 1.0) < 1e-9
    assert saw_switch


def test_learned_embedding_slots_break_symmetry():
    """Regression guard: zero-initialized slots that all start identical and
    receive identical proportionally-scaled updates forever (sign(c*x) ==
    sign(x) for c>0) collapse into a slowed-down copy of a single flat rule
    -- confirmed by an earlier run that produced numerically identical
    results to FlatBaselineAgent, see RESULTS.md. Random init is what
    prevents this; check the slots actually end up distinguishable."""
    agent = LearnedEmbeddingAgent(len(FRAMES), seed=0)
    env = FrameSwitchEnv(FRAMES, switch_period=40, jitter=10, noise_sigma=0.05, seed=0)
    for _ in range(400):
        info = env.observe()
        predict = agent.act(info.raw)
        reward = env.feedback(predict)
        agent.update(reward)
    assert len({round(w, 6) for w, b in agent.experts}) > 1, "slots collapsed back to a single identical rule"


def test_true_oracle_matches_or_beats_rf_aware_given_the_true_frame():
    """TrueOracleAgent has the exact rule and is told which frame is active
    -- it should never need to be worse than RFAwareAgent, which has to
    infer the latter. This is the "cost of inference" isolation RESULTS.md
    describes; check the ceiling is actually a ceiling."""
    for seed in (0, 1, 2):
        env_rf = FrameSwitchEnv(FRAMES, switch_period=40, jitter=10, noise_sigma=0.05, seed=seed)
        rf = RFAwareAgent(FRAMES)
        rf_correct = 0
        for _ in range(400):
            info = env_rf.observe()
            reward = env_rf.feedback(rf.act(info.raw))
            rf.update(reward)
            rf_correct += reward

        env_oracle = FrameSwitchEnv(FRAMES, switch_period=40, jitter=10, noise_sigma=0.05, seed=seed)
        oracle = TrueOracleAgent(FRAMES)
        oracle_correct = 0
        for _ in range(400):
            info = env_oracle.observe()
            reward = env_oracle.feedback(oracle.act(info.raw, info.active_frame_index))
            oracle.update(reward)
            oracle_correct += reward

        assert oracle_correct >= rf_correct, f"seed={seed}: oracle={oracle_correct} < rf_aware={rf_correct}"


def test_windows_scale_with_switch_period_and_match_original_fixed_values():
    """At the original switch_period=40, the proportional windows must
    reduce to exactly the fixed (0,10)/(30,40) windows the first version of
    this experiment used, or RESULTS.md's numbers stop being reproducible."""
    recovery, steady = _windows(40)
    assert recovery == (0, 10)
    assert steady == (30, 40)
    # Shorter periods should shrink both windows, not just one.
    short_recovery, short_steady = _windows(20)
    assert short_recovery[1] < recovery[1]
    assert short_steady[1] - short_steady[0] <= steady[1] - steady[0]


def test_sweep_grid_point_runs_and_returns_expected_shape():
    """Smoke test for sweep.py's machinery (not the full 10-seed/3000-step
    sweep in RESULTS.md, which takes ~50s) -- confirms run_grid_point wires
    together run_agent_on_seed/AGENT_KINDS correctly at a tiny scale."""
    import experiments.exp01_frame_conditioning.sweep as sweep_module
    original_seeds, original_steps = sweep_module.SEEDS, sweep_module.N_STEPS
    sweep_module.SEEDS, sweep_module.N_STEPS = [0, 1], 200
    try:
        results = run_grid_point(switch_period=20, jitter=5, noise_sigma=0.1)
    finally:
        sweep_module.SEEDS, sweep_module.N_STEPS = original_seeds, original_steps

    for key in ("flat", "learned_embedding", "rf_aware", "flat_oracle", "true_oracle",
                "rf_aware-flat", "rf_aware-learned_embedding"):
        assert key in results
        assert "overall_acc" in results[key] and "stdev" in results[key]


def test_brier_score_hand_computed_cases():
    """A perfectly calibrated point mass on the truth scores 0; a uniform
    guess over 4 classes scores 3*(0.25)^2 + (0.75)^2 = 0.75; a confident
    point mass on the WRONG class scores 2 (worst case)."""
    log = SeedLog(
        switched=[True] + [False] * 9,
        active_frame_index=[0] * 10,
        belief_vector=[(1.0, 0.0, 0.0, 0.0)] * 10,
    )
    assert brier_score(log, switch_period=10) == pytest.approx(0.0)

    log.belief_vector = [(0.25, 0.25, 0.25, 0.25)] * 10
    assert brier_score(log, switch_period=10) == pytest.approx(0.75)

    log.belief_vector = [(0.0, 1.0, 0.0, 0.0)] * 10
    assert brier_score(log, switch_period=10) == pytest.approx(2.0)


def test_rf_aware_degrades_more_than_flat_on_a_truly_held_out_frame():
    """Regression guard for the held-out-frame finding in RESULTS.md:
    RFAwareAgent's fixed candidate list makes it fall further when the
    active frame isn't in it than FlatBaselineAgent, which never assumed a
    known frame set. Small-scale version of held_out_frame.py's real run."""
    train_frames = FRAMES[:3]
    held_out_frame = FRAMES[3]
    all_frames = FRAMES

    def run(agent, seed: int, n_steps: int = 600) -> tuple[float, float]:
        env = FrameSwitchEnv(all_frames, switch_period=40, jitter=10, noise_sigma=0.05, seed=seed)
        known_correct = known_total = held_out_correct = held_out_total = 0
        for _ in range(n_steps):
            info = env.observe()
            reward = env.feedback(agent.act(info.raw))
            agent.update(reward)
            if all_frames[info.active_frame_index] is held_out_frame:
                held_out_correct += reward
                held_out_total += 1
            else:
                known_correct += reward
                known_total += 1
        return known_correct / max(1, known_total), held_out_correct / max(1, held_out_total)

    for seed in (0, 1, 2):
        flat_known, flat_held_out = run(FlatBaselineAgent(), seed)
        rf_known, rf_held_out = run(RFAwareAgent(train_frames), seed)
        flat_gap = flat_held_out - flat_known
        rf_gap = rf_held_out - rf_known
        assert rf_gap < flat_gap, f"seed={seed}: rf_aware's held-out gap ({rf_gap}) should be more negative than flat's ({flat_gap})"


def test_rnn_embedding_produces_valid_bounded_output():
    """Sanity check for RNNEmbeddingAgent's mechanics: predictions are
    valid, hidden state stays within tanh's range, weights actually move."""
    agent = RNNEmbeddingAgent(hidden_dim=4, seed=0)
    initial_v = list(agent.v)
    env = FrameSwitchEnv(FRAMES, switch_period=40, jitter=10, noise_sigma=0.05, seed=0)
    for _ in range(200):
        info = env.observe()
        predict = agent.act(info.raw)
        assert predict in (1, -1)
        reward = env.feedback(predict)
        agent.update(reward)
        assert all(-1.0 <= x <= 1.0 for x in agent.h)
    assert agent.v != initial_v


def test_rnn_embedding_underperforms_flat_small_scale():
    """Regression guard for the vanishing-gradient finding in RESULTS.md:
    a genuine tanh-RNN encoder-decoder trained via truncated BPTT
    underperforms even the flat baseline (0.630 vs 0.700 at full scale,
    10/10 seeds), not just loses to learned_embedding. This locks in that
    qualitative result at reduced scale so a future change doesn't
    silently "fix" it without the fix being noticed and documented."""
    def run(agent, seed: int, n_steps: int = 600) -> float:
        env = FrameSwitchEnv(FRAMES, switch_period=40, jitter=10, noise_sigma=0.05, seed=seed)
        correct = 0
        for _ in range(n_steps):
            info = env.observe()
            reward = env.feedback(agent.act(info.raw))
            agent.update(reward)
            correct += reward
        return correct / n_steps

    for seed in (0, 1, 2):
        flat_acc = run(FlatBaselineAgent(), seed)
        rnn_acc = run(RNNEmbeddingAgent(seed=seed), seed)
        assert rnn_acc < flat_acc, f"seed={seed}: rnn_embedding={rnn_acc} did not underperform flat={flat_acc}"


def test_rnn_embedding_bptt_length_has_negligible_effect():
    """Regression guard for the vanishing-gradient diagnosis itself:
    increasing bptt_steps beyond a couple of steps should barely change
    accuracy, because gradient contributions from distant steps vanish
    (tanh' <= 1 times small recurrent weights, compounding). If a future
    change to the architecture or init makes bptt_steps matter a lot, that
    would mean the vanishing-gradient explanation in RESULTS.md needs
    revisiting, not that this test's tolerance should just be loosened."""
    def run(bptt_steps: int, seed: int = 0, n_steps: int = 600) -> float:
        env = FrameSwitchEnv(FRAMES, switch_period=40, jitter=10, noise_sigma=0.05, seed=seed)
        agent = RNNEmbeddingAgent(bptt_steps=bptt_steps, seed=seed)
        correct = 0
        for _ in range(n_steps):
            info = env.observe()
            reward = env.feedback(agent.act(info.raw))
            agent.update(reward)
            correct += reward
        return correct / n_steps

    acc_1 = run(bptt_steps=1)
    acc_20 = run(bptt_steps=20)
    assert abs(acc_1 - acc_20) < 0.05, f"bptt_steps=1 ({acc_1}) vs bptt_steps=20 ({acc_20}) differ more than expected"
