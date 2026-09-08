"""Tests for CausalGraph and ordinary_least_squares
(transintelligence/reasoning/causal/model.py, docs/research-agenda.md #7c)
-- the first non-trivial content in transintelligence/reasoning/causal/,
previously just a docstring stub.

The three canonical d-separation structures (chain, fork, collider) are
tested explicitly and independently of any specific experiment's graph --
correctness here is foundational to everything built on top of it.
"""
import random

import pytest

from transintelligence.reasoning.causal import (
    CausalGraph,
    discover_skeleton,
    fisher_z_independence_test,
    front_door_adjustment,
    ordinary_least_squares,
    orient_colliders,
    partial_correlation,
    two_stage_least_squares,
)


def test_chain_is_d_separated_only_when_conditioning_on_the_middle_node():
    chain = CausalGraph((("A", "B"), ("B", "C")))
    assert chain.d_separated("A", "C", set()) is False
    assert chain.d_separated("A", "C", {"B"}) is True


def test_fork_is_d_separated_only_when_conditioning_on_the_common_cause():
    fork = CausalGraph((("B", "A"), ("B", "C")))
    assert fork.d_separated("A", "C", set()) is False
    assert fork.d_separated("A", "C", {"B"}) is True


def test_collider_is_d_separated_unless_conditioned_on():
    collider = CausalGraph((("A", "B"), ("C", "B")))
    assert collider.d_separated("A", "C", set()) is True
    assert collider.d_separated("A", "C", {"B"}) is False


def test_conditioning_on_a_colliders_descendant_also_opens_the_path():
    collider_desc = CausalGraph((("A", "B"), ("C", "B"), ("B", "D")))
    assert collider_desc.d_separated("A", "C", set()) is True
    assert collider_desc.d_separated("A", "C", {"D"}) is False


def test_ancestors_and_descendants():
    g = CausalGraph((("Z", "X"), ("Z", "Y"), ("X", "Y"), ("X", "W"), ("Y", "W")))
    assert g.ancestors("Y") == {"Z", "X"}
    assert g.descendants("X") == {"Y", "W"}
    assert g.parents("W") == {"X", "Y"}
    assert g.children("Z") == {"X", "Y"}
    assert g.nodes == {"Z", "X", "Y", "W"}


def test_backdoor_criterion_on_confounded_graph_with_a_collider():
    """Z confounds X and Y; W is a common effect (collider) of X and Y,
    not a valid adjustment set member."""
    g = CausalGraph((("Z", "X"), ("Z", "Y"), ("X", "Y"), ("X", "W"), ("Y", "W")))
    assert g.satisfies_backdoor_criterion("X", "Y", {"Z"}) is True
    assert g.satisfies_backdoor_criterion("X", "Y", set()) is False
    assert g.satisfies_backdoor_criterion("X", "Y", {"W"}) is False  # W is a descendant of X
    assert g.satisfies_backdoor_criterion("X", "Y", {"Z", "W"}) is False  # still contains a descendant


def test_backdoor_criterion_ignores_front_door_paths():
    """A mediator M on the causal path X->M->Y should not need to be
    blocked -- it's not a backdoor path (doesn't start with an edge into
    X), and blocking it would remove the very effect being estimated."""
    g = CausalGraph((("X", "M"), ("M", "Y")))
    assert g.satisfies_backdoor_criterion("X", "Y", set()) is True


def test_ordinary_least_squares_recovers_exact_linear_relationship():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    features = [[1.0, x] for x in xs]
    targets = [2.0 + 3.0 * x for x in xs]
    coefficients = ordinary_least_squares(features, targets)
    assert coefficients == pytest.approx([2.0, 3.0])


def test_ordinary_least_squares_recovers_multi_covariate_relationship():
    import random
    rng = random.Random(0)
    features, targets = [], []
    for _ in range(50):
        x1, x2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        features.append([1.0, x1, x2])
        targets.append(1.0 + 2.0 * x1 - 1.0 * x2)
    coefficients = ordinary_least_squares(features, targets)
    assert coefficients == pytest.approx([1.0, 2.0, -1.0])


def test_ordinary_least_squares_raises_on_mismatched_shapes():
    with pytest.raises(ValueError):
        ordinary_least_squares([[1.0, 2.0], [1.0]], [1.0, 2.0])
    with pytest.raises(ValueError):
        ordinary_least_squares([[1.0, 2.0]], [1.0, 2.0])


def test_partial_correlation_removes_a_confounder_exactly():
    """X and Y are only correlated through Z here (no direct edge) -- the
    raw correlation should be strongly nonzero, but the partial
    correlation given Z should be indistinguishable from zero."""
    rng = random.Random(1)
    n = 2000
    z = [rng.gauss(0, 1) for _ in range(n)]
    x = [0.9 * v + rng.gauss(0, 0.2) for v in z]
    y = [0.9 * v + rng.gauss(0, 0.2) for v in z]
    data = {"X": x, "Y": y, "Z": z}
    raw = partial_correlation(data, "X", "Y", set())
    partial = partial_correlation(data, "X", "Y", {"Z"})
    assert raw > 0.7
    assert abs(partial) < 0.05


def test_fisher_z_independence_test_matches_direct_hand_computation():
    """r=0.5, n=103, conditioning on 0 vars -> dof=100,
    z = atanh(0.5) * 10 = 0.5493 * 10 = 5.493, an extreme z that must be
    rejected as independent at alpha=0.05; r=0.0 with the same n must not
    be rejected."""
    assert fisher_z_independence_test(r=0.5, n=103, conditioning_set_size=0, alpha=0.05) is False
    assert fisher_z_independence_test(r=0.0, n=103, conditioning_set_size=0, alpha=0.05) is True


def test_fisher_z_independence_test_treats_too_few_degrees_of_freedom_as_independent():
    assert fisher_z_independence_test(r=0.9, n=5, conditioning_set_size=3, alpha=0.05) is True


def test_discover_skeleton_recovers_a_chain_without_the_spurious_direct_edge():
    rng = random.Random(2)
    n = 2000
    a = [rng.gauss(0, 1) for _ in range(n)]
    b = [0.8 * v + rng.gauss(0, 0.3) for v in a]
    c = [0.8 * v + rng.gauss(0, 0.3) for v in b]
    skeleton = discover_skeleton({"A": a, "B": b, "C": c}, alpha=0.01)
    assert skeleton.undirected_edges == {frozenset({"A", "B"}), frozenset({"B", "C"})}
    assert skeleton.separating_sets[frozenset({"A", "C"})] == frozenset({"B"})


def test_discover_skeleton_finds_no_edges_among_independent_variables():
    """Negative control: three mutually independent variables should
    yield an empty skeleton, not spurious edges from noise alone."""
    rng = random.Random(3)
    n = 2000
    data = {name: [rng.gauss(0, 1) for _ in range(n)] for name in ("A", "B", "C")}
    skeleton = discover_skeleton(data, alpha=0.01)
    assert skeleton.undirected_edges == frozenset()


def test_orient_colliders_finds_the_unshielded_v_structure():
    rng = random.Random(4)
    n = 2000
    a = [rng.gauss(0, 1) for _ in range(n)]
    c = [rng.gauss(0, 1) for _ in range(n)]
    b = [0.7 * x + 0.7 * y + rng.gauss(0, 0.3) for x, y in zip(a, c)]
    data = {"A": a, "B": b, "C": c}
    skeleton = discover_skeleton(data, alpha=0.01)
    directed = orient_colliders(skeleton, sorted(data.keys()))
    assert directed == frozenset({("A", "B"), ("C", "B")})


def test_two_stage_least_squares_recovers_the_true_effect_despite_an_unobserved_confounder():
    """U confounds X and Y and is never passed to two_stage_least_squares
    at all (simulating "unobserved") -- naive OLS of Y on X must be
    substantially biased by it, while 2SLS using the valid instrument Z
    must land close to the true effect."""
    rng = random.Random(10)
    n = 4000
    true_effect = 0.6
    z_list, x_list, y_list = [], [], []
    for _ in range(n):
        u = rng.gauss(0, 1)
        z = rng.gauss(0, 1)
        x = 0.9 * z + 0.8 * u + rng.gauss(0, 0.3)
        y = true_effect * x + 0.8 * u + rng.gauss(0, 0.3)
        z_list.append(z); x_list.append(x); y_list.append(y)

    naive = ordinary_least_squares([[1.0, x] for x in x_list], y_list)[1]
    iv_estimate = two_stage_least_squares(z_list, x_list, y_list)
    assert abs(naive - true_effect) > 0.3  # naive is substantially biased
    assert abs(iv_estimate - true_effect) < 0.1  # 2SLS is not


def test_front_door_adjustment_recovers_the_true_effect_despite_an_unobserved_confounder():
    """U confounds X and Y directly but not the mediator M -- the classic
    front-door setup where no valid backdoor adjustment set exists (U
    isn't observed) but front-door identification still applies."""
    rng = random.Random(11)
    n = 4000
    true_xm, true_my = 0.7, 0.5
    x_list, m_list, y_list = [], [], []
    for _ in range(n):
        u = rng.gauss(0, 1)
        x = 0.8 * u + rng.gauss(0, 0.3)
        m = true_xm * x + rng.gauss(0, 0.3)
        y = true_my * m + 0.8 * u + rng.gauss(0, 0.3)
        x_list.append(x); m_list.append(m); y_list.append(y)

    true_total_effect = true_xm * true_my
    naive = ordinary_least_squares([[1.0, x] for x in x_list], y_list)[1]
    fd_estimate = front_door_adjustment(x_list, m_list, y_list)
    assert abs(naive - true_total_effect) > 0.3  # naive is substantially biased
    assert abs(fd_estimate - true_total_effect) < 0.05  # front-door is not


def test_orient_colliders_leaves_a_shielded_triple_unoriented():
    """W is a collider of X and Y, but X->Y is also a direct edge, so the
    triple X-W-Y is shielded -- the same structure as experiment 6's
    graph. The rule must not fire here even though W really is a
    collider, because shielded triples are exactly where the rule is
    inapplicable."""
    rng = random.Random(5)
    n = 2000
    z = [rng.gauss(0, 1) for _ in range(n)]
    x = [0.8 * v + rng.gauss(0, 0.2) for v in z]
    y = [0.8 * zv + 0.5 * xv + rng.gauss(0, 0.2) for zv, xv in zip(z, x)]
    w = [0.5 * xv + 0.5 * yv + rng.gauss(0, 0.2) for xv, yv in zip(x, y)]
    data = {"Z": z, "X": x, "Y": y, "W": w}
    skeleton = discover_skeleton(data, alpha=0.01)
    directed = orient_colliders(skeleton, sorted(data.keys()))
    assert ("X", "W") not in directed and ("Y", "W") not in directed
