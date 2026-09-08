"""Regression tests for Experiment 8's Meek's-rules follow-up
(docs/research-agenda.md #7e,
experiments/exp08_causal_discovery/RESULTS.md "Follow-up" section).
Small-scale versions of the real follow-up (fewer seeds/samples, for
test-suite speed).
"""
from transintelligence.reasoning.causal import apply_meek_rules, discover_skeleton, orient_colliders

from experiments.exp08_causal_discovery.meek_rules import (
    PROPAGATION_TRUE_DIRECTED_EDGES,
    PROPAGATION_TRUE_SKELETON,
    simulate_plain_chain,
    simulate_propagation_graph,
)


def test_meek_rules_recover_the_full_graph_at_moderate_sample_size():
    data = simulate_propagation_graph(seed=0, n=1500)
    nodes = sorted(data.keys())
    skeleton = discover_skeleton(data, alpha=0.01)
    colliders = orient_colliders(skeleton, nodes)
    full = apply_meek_rules(skeleton, colliders, nodes)
    assert colliders == {("A", "C"), ("B", "C")}  # collider orientation alone: only half
    assert full == PROPAGATION_TRUE_DIRECTED_EDGES  # Meek's rules: all four


def test_meek_rules_never_orient_wrongly_given_a_correct_skeleton():
    reversed_true = {(c, p) for p, c in PROPAGATION_TRUE_DIRECTED_EDGES}
    for seed in range(10):
        data = simulate_propagation_graph(seed, n=1500)
        nodes = sorted(data.keys())
        skeleton = discover_skeleton(data, alpha=0.01)
        if skeleton.undirected_edges != PROPAGATION_TRUE_SKELETON:
            continue  # skeleton-recovery error, already characterized in exp08's own RESULTS.md
        colliders = orient_colliders(skeleton, nodes)
        full = apply_meek_rules(skeleton, colliders, nodes)
        assert not (full & reversed_true)


def test_meek_rules_orient_nothing_on_a_genuinely_undetermined_chain():
    """Negative control: A->B->C has no collider and is Markov-equivalent
    to a fork and a reverse chain -- no amount of data can determine
    which one generated it, so apply_meek_rules must never invent an
    orientation here, at any sample size."""
    for seed in range(10):
        data = simulate_plain_chain(seed, n=1500)
        nodes = sorted(data.keys())
        skeleton = discover_skeleton(data, alpha=0.01)
        colliders = orient_colliders(skeleton, nodes)
        full = apply_meek_rules(skeleton, colliders, nodes)
        assert full == frozenset()
