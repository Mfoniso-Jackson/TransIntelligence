"""Smoke/regression test for Experiment 1 (docs/research-agenda.md #5,
experiments/exp01_frame_conditioning/RESULTS.md). Runs a small-scale version
of the real experiment (fewer seeds/steps, for test-suite speed) and checks
the qualitative finding holds: RFAwareAgent beats FlatBaselineAgent when
neither knows which frame is active. This is not a substitute for the full
run in RESULTS.md (10 seeds x 3000 steps) -- it's a fast regression guard
against a future change accidentally breaking the mechanism.
"""
from transintelligence import ReferenceFrame

from environments.transworld import FrameSwitchEnv
from experiments.exp01_frame_conditioning.agents import FlatBaselineAgent, LearnedEmbeddingAgent, RFAwareAgent, TrueOracleAgent

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
