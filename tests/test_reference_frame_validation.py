"""Tests for ReferenceFrame.validate()
(transintelligence/representation/reference_frames/model.py).
"""
from datetime import datetime, timedelta, timezone

from transintelligence.core.common import TimeWindow
from transintelligence.representation.reference_frames import ReferenceFrame

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_a_well_formed_frame_has_no_issues():
    frame = ReferenceFrame(name="Historical", baseline=0.5, domain="finance",
                            time_window=TimeWindow(start=T0, end=T0 + timedelta(days=1)))
    assert frame.validate() == ()


def test_empty_name_is_flagged():
    frame = ReferenceFrame(name="")
    issues = frame.validate()
    assert len(issues) == 1
    assert "name" in issues[0]


def test_whitespace_only_domain_is_flagged():
    frame = ReferenceFrame(name="Frame", domain="   ")
    issues = frame.validate()
    assert any("domain" in issue for issue in issues)


def test_none_domain_is_not_flagged():
    """domain is optional -- None should be fine, only an explicitly
    set-but-empty/whitespace string is a real construction mistake."""
    frame = ReferenceFrame(name="Frame", domain=None)
    assert frame.validate() == ()


def test_inverted_time_window_is_flagged():
    frame = ReferenceFrame(name="Frame", time_window=TimeWindow(start=T0 + timedelta(days=1), end=T0))
    issues = frame.validate()
    assert any("time_window" in issue for issue in issues)


def test_time_window_with_only_start_or_only_end_is_not_flagged():
    """A half-open window (only a start or only an end bound) has no
    inversion to check -- shouldn't be treated as invalid."""
    assert ReferenceFrame(name="Frame", time_window=TimeWindow(start=T0)).validate() == ()
    assert ReferenceFrame(name="Frame", time_window=TimeWindow(end=T0)).validate() == ()


def test_multiple_issues_are_all_reported_together():
    frame = ReferenceFrame(name="", domain="  ", time_window=TimeWindow(start=T0 + timedelta(days=1), end=T0))
    assert len(frame.validate()) == 3
