"""Regression tests for Experiment 3 (docs/research-agenda.md #7,
experiments/exp03_cross_domain_transfer/RESULTS.md). Small-scale versions
of the real experiment (fewer trials/steps, for test-suite speed).
"""
from experiments.exp01_frame_conditioning.agents import LearnedEmbeddingAgent
from experiments.exp03_cross_domain_transfer.domains import (
    DOMAIN_A_FRAMES,
    DOMAIN_B_ISOMORPHIC_FRAMES,
    DOMAIN_B_NONISOMORPHIC_FRAMES,
)
from experiments.exp03_cross_domain_transfer.run import (
    random_differentiated_experts,
    run_on_domain_b,
    train_on_domain_a,
)


def test_warm_started_agent_uses_the_given_experts_verbatim():
    experts = [(1.0, -2.0), (3.0, 4.0)]
    agent = LearnedEmbeddingAgent(n_slots=2, initial_experts=experts)
    assert agent.experts == experts


def test_train_on_domain_a_produces_differentiated_experts():
    experts = train_on_domain_a(seed=0)
    assert len(experts) == len(DOMAIN_A_FRAMES)
    assert len({round(w, 4) for w, b in experts}) > 1


def test_transfer_beats_scratch_on_isomorphic_domain_b_small_scale():
    """Small-scale directional check: the full run (RESULTS.md) shows
    transfer beating scratch 10/10 trials on the isomorphic target with a
    sizable margin, so this should hold even at reduced scale."""
    import experiments.exp03_cross_domain_transfer.run as run_module
    original_a, original_b = run_module.N_STEPS_A, run_module.N_STEPS_B
    run_module.N_STEPS_A, run_module.N_STEPS_B = 500, 500
    try:
        for seed in (0, 1, 2):
            trained_experts = train_on_domain_a(seed=seed)
            b_seed = seed + 1000
            t_early, _ = run_on_domain_b(DOMAIN_B_ISOMORPHIC_FRAMES, b_seed, initial_experts=trained_experts)
            s_early, _ = run_on_domain_b(DOMAIN_B_ISOMORPHIC_FRAMES, b_seed, initial_experts=None)
            assert t_early > s_early, f"seed={seed}: transfer={t_early} did not beat scratch={s_early}"
    finally:
        run_module.N_STEPS_A, run_module.N_STEPS_B = original_a, original_b


def test_random_differentiated_control_is_distinct_from_transfer_and_scratch():
    """The confound-isolating control must actually differ from both the
    trained-transfer values and near-zero scratch init, or it isn't
    controlling for anything."""
    trained = train_on_domain_a(seed=0)
    random_diff = random_differentiated_experts(len(DOMAIN_A_FRAMES), seed=0)
    assert trained != random_diff
    for (rw, rb) in random_diff:
        assert abs(rw) > 0.1 or abs(rb) > 0.1  # not near-zero like scratch's init_scale=0.05
