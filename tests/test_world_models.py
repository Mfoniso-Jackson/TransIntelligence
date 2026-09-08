"""Tests for LinearDynamicsModel (transintelligence/world_models/model.py,
docs/research-agenda.md #7h) -- the first content in
transintelligence/world_models/, previously an empty package.
"""
import pytest

from transintelligence.world_models import LinearDynamicsModel


def test_fit_recovers_exact_linear_dynamics_per_action():
    transitions = [
        (1.0, "A", 4.0), (2.0, "A", 5.0), (5.0, "A", 8.0),
        (1.0, "B", 1.0), (2.0, "B", 3.0), (3.0, "B", 5.0),
    ]
    model = LinearDynamicsModel.fit(transitions)
    assert model.coefficients["A"] == pytest.approx((3.0, 1.0))
    assert model.coefficients["B"] == pytest.approx((-1.0, 2.0))


def test_predict_matches_hand_computation():
    transitions = [(1.0, "A", 4.0), (2.0, "A", 5.0), (5.0, "A", 8.0)]
    model = LinearDynamicsModel.fit(transitions)
    assert model.predict(10.0, "A") == pytest.approx(13.0)


def test_predict_raises_for_an_action_with_no_fitted_dynamics():
    model = LinearDynamicsModel.fit([(1.0, "A", 4.0), (2.0, "A", 5.0)])
    with pytest.raises(ValueError):
        model.predict(10.0, "B")


def test_fit_skips_actions_with_fewer_than_two_observations():
    """A single transition can't determine both an intercept and a
    slope -- such actions should simply not appear in `coefficients`,
    not silently produce a nonsensical fit."""
    transitions = [(1.0, "A", 4.0), (2.0, "A", 5.0), (3.0, "B", 6.0)]
    model = LinearDynamicsModel.fit(transitions)
    assert "A" in model.coefficients
    assert "B" not in model.coefficients
    assert model.known_actions() == {"A"}


def test_fit_with_noisy_data_recovers_dynamics_approximately():
    import random
    rng = random.Random(0)
    transitions = []
    for _ in range(200):
        s = rng.uniform(-5, 5)
        ns = s + 2.0 + rng.gauss(0, 0.1)  # true dynamics: next_state = state + 2.0
        transitions.append((s, "nudge", ns))
    model = LinearDynamicsModel.fit(transitions)
    intercept, slope = model.coefficients["nudge"]
    assert intercept == pytest.approx(2.0, abs=0.05)
    assert slope == pytest.approx(1.0, abs=0.02)
