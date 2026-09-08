"""Tests for RecedingHorizonPlanner (transintelligence/planning/model.py,
docs/research-agenda.md #7i) -- the first content in
transintelligence/planning/, previously an empty package. Verified
against the same hand-computed cases that validated experiment 12's
original, environment-specific `choose_action` function before it was
generalized into this kernel primitive.
"""
import pytest

from transintelligence.planning import RecedingHorizonPlanner

NUDGES = {"large_down": -2.0, "small_down": -1.0, "tiny_down": -0.3,
          "tiny_up": 0.3, "small_up": 1.0, "large_up": 2.0}
TARGET = 0.0


def true_transition(state, action):
    position, pending = state
    return (position + 0.5 * NUDGES[action] + 0.5 * pending, NUDGES[action])


def score(state):
    position, _ = state
    return -((position - TARGET) ** 2)


def test_depth_1_matches_hand_computed_greedy_choice():
    planner = RecedingHorizonPlanner(actions=tuple(NUDGES.keys()))
    # position=0, pending=2.0: next_position = 0.5*nudge + 1.0, minimized at nudge=-2.0 (exact zero).
    assert planner.choose_action((0.0, 2.0), true_transition, score, depth=1) == "large_down"


def test_depth_2_lookahead_still_finds_the_correct_first_action():
    planner = RecedingHorizonPlanner(actions=tuple(NUDGES.keys()))
    assert planner.choose_action((0.0, 2.0), true_transition, score, depth=2) == "large_down"


def test_depth_is_clamped_to_at_least_one():
    """depth=0 shouldn't produce a degenerate empty-sequence search --
    it should still evaluate single-action sequences."""
    planner = RecedingHorizonPlanner(actions=tuple(NUDGES.keys()))
    assert planner.choose_action((0.0, 2.0), true_transition, score, depth=0) == "large_down"


def test_raises_with_no_actions():
    planner = RecedingHorizonPlanner(actions=())
    with pytest.raises(ValueError):
        planner.choose_action((0.0, 0.0), true_transition, score, depth=1)


def test_maximizes_score_not_minimizes():
    """Sanity check on the sign convention: a score function that
    REWARDS large positions should pick the action that maximizes
    position, not minimizes distance to some target."""
    planner = RecedingHorizonPlanner(actions=tuple(NUDGES.keys()))
    reward_large_position = lambda state: state[0]  # maximize raw position
    assert planner.choose_action((0.0, 0.0), true_transition, reward_large_position, depth=1) == "large_up"
