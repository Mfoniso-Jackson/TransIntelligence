"""Regression tests for the Experiment 2 finding and fix (see
docs/research-agenda.md #6 and experiments/exp02_frame_dependence/RESULTS.md).

BaselineRelativeReasoner.sensitivity() used to reduce algebraically to
abs(r2.baseline - r1.baseline) whenever r1 and r2 shared a `direction` --
the entity's raw value cancelled out of the point-difference, so it carried
zero per-entity signal in exactly the regime the finance demo uses. It was
fixed to compare the *signs* of the two evaluate() results instead (each of
which does depend on the raw value). These tests lock in the fixed
behavior: sensitivity now varies across entities, and correctly flags
conclusion flips, even when direction matches.
"""
import pytest

from transintelligence import Entity, Observation, ReferenceFrame
from transintelligence.reasoning.relative import BaselineRelativeReasoner


def _frame(name: str, baseline: float, direction: str) -> ReferenceFrame:
    return ReferenceFrame(name, baseline=baseline, metadata={"property": "score", "direction": direction})


def _reasoner(raw: float) -> tuple[BaselineRelativeReasoner, Entity]:
    e = Entity("synthetic", "x")
    return BaselineRelativeReasoner([Observation(e.id, "score", raw, "unit")]), e


def test_sensitivity_now_varies_across_entities_when_direction_matches():
    r1 = _frame("historical", baseline=0.5, direction="higher_is_better")
    r2 = _frame("portfolio-risk", baseline=0.3, direction="higher_is_better")

    reasoner_low, low = _reasoner(0.05)   # below both baselines -> both frames agree: negative
    reasoner_high, high = _reasoner(0.95)  # above both baselines -> both frames agree: positive
    reasoner_mid, mid = _reasoner(0.40)    # between the baselines -> conclusion flips

    assert reasoner_low.sensitivity(low, r1, r2).result != pytest.approx(reasoner_high.sensitivity(high, r1, r2).result)
    assert reasoner_mid.sensitivity(mid, r1, r2).result < 0  # flips -> negative
    assert reasoner_low.sensitivity(low, r1, r2).result > 0  # agrees -> positive
    assert reasoner_high.sensitivity(high, r1, r2).result > 0  # agrees -> positive


def test_sensitivity_sign_matches_conclusion_flip_ground_truth():
    r1 = _frame("historical", baseline=0.5, direction="higher_is_better")
    r2 = _frame("portfolio-risk", baseline=0.3, direction="higher_is_better")

    for raw in (0.05, 0.2, 0.29, 0.31, 0.4, 0.49, 0.51, 0.7, 0.95):
        reasoner, e = _reasoner(raw)
        conclusion1 = (raw - r1.baseline) > 0
        conclusion2 = (raw - r2.baseline) > 0
        expected_flip = conclusion1 != conclusion2
        result = reasoner.sensitivity(e, r1, r2).result
        assert (result < 0) == expected_flip, f"raw={raw}: expected flip={expected_flip}, got sensitivity={result}"


def test_sensitivity_still_varies_when_direction_differs():
    r1 = _frame("historical", baseline=0.5, direction="higher_is_better")
    r2 = _frame("portfolio-risk", baseline=0.3, direction="lower_is_better")

    reasoner_low, low = _reasoner(0.05)
    reasoner_high, high = _reasoner(0.95)
    assert reasoner_low.sensitivity(low, r1, r2).result != pytest.approx(reasoner_high.sensitivity(high, r1, r2).result)
