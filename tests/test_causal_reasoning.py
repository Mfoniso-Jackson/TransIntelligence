"""Tests for CausalGraph and ordinary_least_squares
(transintelligence/reasoning/causal/model.py, docs/research-agenda.md #7c)
-- the first non-trivial content in transintelligence/reasoning/causal/,
previously just a docstring stub.

The three canonical d-separation structures (chain, fork, collider) are
tested explicitly and independently of any specific experiment's graph --
correctness here is foundational to everything built on top of it.
"""
import pytest

from transintelligence.reasoning.causal import CausalGraph, ordinary_least_squares


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
