"""Regression tests for Experiment 8 (docs/research-agenda.md #7e,
experiments/exp08_causal_discovery/RESULTS.md). Small-scale versions of
the real experiment (fewer seeds/samples, for test-suite speed).
"""
from transintelligence.reasoning.causal import discover_skeleton, orient_colliders

from experiments.exp08_causal_discovery.run import (
    CONFOUNDING_TRUE_EDGES,
    COLLIDER_TRUE_EDGES,
    simulate_collider,
    simulate_confounding,
    simulate_independent,
)


def test_confounding_graph_skeleton_recovers_all_five_edges_at_moderate_n():
    data = simulate_confounding(seed=0, n=1500)
    skeleton = discover_skeleton(data, alpha=0.01)
    assert skeleton.undirected_edges == CONFOUNDING_TRUE_EDGES


def test_confounding_graphs_shielded_collider_is_never_falsely_oriented():
    for seed in range(10):
        data = simulate_confounding(seed=seed, n=1000)
        skeleton = discover_skeleton(data, alpha=0.01)
        directed = orient_colliders(skeleton, sorted(data.keys()))
        assert ("X", "W") not in directed
        assert ("Y", "W") not in directed


def test_dedicated_collider_graph_is_recovered_and_correctly_oriented():
    data = simulate_collider(seed=0, n=1000)
    skeleton = discover_skeleton(data, alpha=0.01)
    assert skeleton.undirected_edges == COLLIDER_TRUE_EDGES
    directed = orient_colliders(skeleton, sorted(data.keys()))
    assert directed == frozenset({("A", "B"), ("C", "B")})


def test_independent_variables_negative_control_stays_near_the_alpha_false_edge_rate():
    """Regression guard for the confound this experiment's whole result
    depends on: at alpha=0.01, an occasional false edge from ordinary
    Type-I error is expected (RESULTS.md measured 1/20 trials at n=1000),
    but the discovery procedure must not treat noise as structure at a
    rate anywhere near the true-edge recall rates seen elsewhere in this
    experiment -- total false edges across 10 independent-variable trials
    should stay small, not grow trial-over-trial."""
    total_false_edges = sum(
        len(discover_skeleton(simulate_independent(seed=seed, n=1000), alpha=0.01).undirected_edges)
        for seed in range(10)
    )
    assert total_false_edges <= 2
