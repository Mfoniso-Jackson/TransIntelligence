"""Tests for StructuralCausalModel (transintelligence/reasoning/counterfactual/model.py,
docs/research-agenda.md #7d) -- the first non-trivial content in
transintelligence/reasoning/counterfactual/, previously just a docstring
stub. Verified against hand-computed cases before experiments/exp07 trusts
it for anything.
"""
import pytest

from transintelligence.reasoning.causal import CausalGraph
from transintelligence.reasoning.counterfactual import StructuralCausalModel, StructuralEquation


def test_abduction_recovers_exact_residual_on_a_simple_chain():
    # A -> B, B = 2*A + 3 + noise. Observed A=1, B=8 => noise_B = 8 - 5 = 3.
    graph = CausalGraph((("A", "B"),))
    scm = StructuralCausalModel(graph, {"B": StructuralEquation({"A": 2.0}, intercept=3.0)})
    noise = scm.abduct({"A": 1.0, "B": 8.0})
    assert noise == pytest.approx({"A": 1.0, "B": 3.0})


def test_counterfactual_on_a_simple_chain_matches_hand_computation():
    graph = CausalGraph((("A", "B"),))
    scm = StructuralCausalModel(graph, {"B": StructuralEquation({"A": 2.0}, intercept=3.0)})
    result = scm.counterfactual({"A": 1.0, "B": 8.0}, {"A": 5.0})
    assert result == pytest.approx({"A": 5.0, "B": 16.0})  # 2*5 + 3 + noise_B(3) = 16


def test_counterfactual_preserves_unit_specific_noise_not_population_average():
    """Two units with the same observed A but different B (different
    residuals) must get DIFFERENT counterfactual B's under the same
    intervention -- this is the entire point of abduction over a naive
    population-average plug-in, which would give both units the same
    answer."""
    graph = CausalGraph((("A", "B"),))
    scm = StructuralCausalModel(graph, {"B": StructuralEquation({"A": 2.0}, intercept=3.0)})
    unit_1 = scm.counterfactual({"A": 1.0, "B": 8.0}, {"A": 5.0})   # noise_B = 3
    unit_2 = scm.counterfactual({"A": 1.0, "B": 4.0}, {"A": 5.0})   # noise_B = -1
    assert unit_1["B"] != unit_2["B"]
    assert unit_1["B"] == pytest.approx(16.0)
    assert unit_2["B"] == pytest.approx(12.0)


def test_counterfactual_on_confounded_graph_propagates_through_multiple_nodes():
    graph = CausalGraph((("Z", "X"), ("Z", "Y"), ("X", "Y")))
    scm = StructuralCausalModel(graph, {
        "X": StructuralEquation({"Z": 0.8}),
        "Y": StructuralEquation({"Z": 0.8, "X": 0.5}),
    })
    observed = {"Z": 1.0, "X": 1.1, "Y": 1.55}  # noise_X=0.3, noise_Y=0.2
    noise = scm.abduct(observed)
    assert noise == pytest.approx({"Z": 1.0, "X": 0.3, "Y": 0.2})

    result = scm.counterfactual(observed, {"X": 2.0})
    assert result["Z"] == pytest.approx(1.0)   # Z is unaffected by intervening downstream on X
    assert result["X"] == pytest.approx(2.0)   # the intervened value
    assert result["Y"] == pytest.approx(2.0)   # 0.8*1.0 + 0.5*2.0 + 0.2


def test_intervening_on_a_node_intervenes_regardless_of_its_own_observed_noise():
    """The intervened node's own structural equation/noise should be
    irrelevant to the result -- it's set exogenously by the intervention,
    not recomputed."""
    graph = CausalGraph((("Z", "X"), ("Z", "Y"), ("X", "Y")))
    scm = StructuralCausalModel(graph, {
        "X": StructuralEquation({"Z": 0.8}),
        "Y": StructuralEquation({"Z": 0.8, "X": 0.5}),
    })
    result_a = scm.counterfactual({"Z": 1.0, "X": 1.1, "Y": 1.55}, {"X": 2.0})
    result_b = scm.counterfactual({"Z": 1.0, "X": 999.0, "Y": 1.55}, {"X": 2.0})  # X's own observed value shouldn't matter
    assert result_a["X"] == result_b["X"] == pytest.approx(2.0)


def test_raises_on_cyclic_or_disconnected_equation_reference():
    graph = CausalGraph((("A", "B"),))
    # "C" has an equation but no edges in the graph at all -- topological
    # sort can't place it relative to anything, this should surface as an error.
    scm = StructuralCausalModel(graph, {
        "B": StructuralEquation({"A": 1.0}),
        "C": StructuralEquation({"D": 1.0}),  # D isn't even a node
    })
    with pytest.raises((ValueError, KeyError)):
        scm.abduct({"A": 1.0, "B": 1.0, "C": 1.0})
