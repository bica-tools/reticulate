"""Tests for protocol downgrade prevention (Step 89c)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.downgrade import (
    DowngradeResult,
    DowngradeRisk,
    ForcedDowngradeResult,
    analyze_downgrade,
    check_downgrade,
    detect_forced_downgrade,
    downgrade_risk,
    safe_downgrade_set,
)
from reticulate.parser import Branch, End, Select


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strong_type():
    """Strong protocol: &{a: end, b: end, c: end}."""
    return parse("&{a: end, b: end, c: end}")


@pytest.fixture
def weak_type():
    """Weak protocol: &{a: end}."""
    return parse("&{a: end}")


@pytest.fixture
def equal_type():
    """Same as strong: &{a: end, b: end, c: end}."""
    return parse("&{a: end, b: end, c: end}")


@pytest.fixture
def end_type():
    """Terminal type."""
    return parse("end")


@pytest.fixture
def select_type():
    """Selection: +{ok: end, err: end}."""
    return parse("+{ok: end, err: end}")


@pytest.fixture
def nested_strong():
    """Nested strong: &{open: &{read: end, write: end}, close: end}."""
    return parse("&{open: &{read: end, write: end}, close: end}")


@pytest.fixture
def nested_weak():
    """Nested weak: &{open: &{read: end}, close: end}."""
    return parse("&{open: &{read: end}, close: end}")


# ---------------------------------------------------------------------------
# check_downgrade
# ---------------------------------------------------------------------------


class TestCheckDowngrade:
    """Tests for check_downgrade."""

    def test_same_type_is_safe(self, strong_type, equal_type):
        assert check_downgrade(strong_type, equal_type) is True

    def test_subtype_is_safe(self, weak_type, strong_type):
        """Strong offers more -> subtype of weak -> safe replacement for weak."""
        assert check_downgrade(weak_type, strong_type) is True

    def test_supertype_is_not_safe(self, strong_type, weak_type):
        """Weak offers less -> NOT a safe replacement for strong."""
        assert check_downgrade(strong_type, weak_type) is False

    def test_end_replaces_end(self, end_type):
        assert check_downgrade(end_type, end_type) is True

    def test_branch_replaces_end(self, end_type, weak_type):
        """&{a: end} is a subtype of end (end = &{}, more methods = subtype)."""
        assert check_downgrade(end_type, weak_type) is True

    def test_end_does_not_replace_branch(self, weak_type, end_type):
        """end cannot replace &{a: end} -- fewer methods."""
        assert check_downgrade(weak_type, end_type) is False

    def test_nested_safe(self, nested_weak, nested_strong):
        """Strong nested is safe replacement for weak nested."""
        assert check_downgrade(nested_weak, nested_strong) is True

    def test_nested_not_safe(self, nested_strong, nested_weak):
        """Weak nested is NOT safe replacement for strong nested."""
        assert check_downgrade(nested_strong, nested_weak) is False


# ---------------------------------------------------------------------------
# downgrade_risk
# ---------------------------------------------------------------------------


class TestDowngradeRisk:
    """Tests for downgrade_risk."""

    def test_no_risk_same_type(self, strong_type, equal_type):
        risk = downgrade_risk(strong_type, equal_type)
        assert risk.risk_score == 0.0
        assert len(risk.lost_methods) == 0

    def test_partial_risk(self, strong_type, weak_type):
        risk = downgrade_risk(strong_type, weak_type)
        assert risk.risk_score > 0.0
        assert "b" in risk.lost_methods
        assert "c" in risk.lost_methods
        assert "a" in risk.retained_methods

    def test_full_risk(self, strong_type, end_type):
        risk = downgrade_risk(strong_type, end_type)
        assert risk.risk_score == 1.0
        assert len(risk.retained_methods) == 0

    def test_risk_between_zero_and_one(self, strong_type, weak_type):
        risk = downgrade_risk(strong_type, weak_type)
        assert 0.0 < risk.risk_score < 1.0

    def test_lost_methods_correct(self, strong_type, weak_type):
        risk = downgrade_risk(strong_type, weak_type)
        assert risk.lost_methods == frozenset({"b", "c"})

    def test_depth_reduction(self, nested_strong, nested_weak):
        risk = downgrade_risk(nested_strong, nested_weak)
        # Strong has deeper paths (read + write) vs weak (read only)
        assert risk.depth_reduction >= 0

    def test_selection_risk(self):
        strong = parse("+{ok: end, err: end, retry: end}")
        weak = parse("+{ok: end}")
        risk = downgrade_risk(strong, weak)
        assert "err" in risk.lost_selections
        assert "retry" in risk.lost_selections
        assert "ok" in risk.retained_selections

    def test_end_to_end_no_risk(self, end_type):
        risk = downgrade_risk(end_type, end_type)
        assert risk.risk_score == 0.0


# ---------------------------------------------------------------------------
# safe_downgrade_set
# ---------------------------------------------------------------------------


class TestSafeDowngradeSet:
    """Tests for safe_downgrade_set."""

    def test_self_is_always_safe(self, strong_type):
        safe = safe_downgrade_set(strong_type, candidates=[strong_type])
        assert len(safe) == 1

    def test_end_is_safe_for_end(self, end_type):
        safe = safe_downgrade_set(end_type, candidates=[end_type])
        assert len(safe) == 1

    def test_supertype_not_in_safe_set(self, weak_type, strong_type):
        """Weak type is not a safe replacement for strong."""
        safe = safe_downgrade_set(strong_type, candidates=[weak_type])
        assert len(safe) == 0

    def test_subtype_in_safe_set(self, weak_type, strong_type):
        """Strong type IS a safe replacement for weak."""
        safe = safe_downgrade_set(weak_type, candidates=[strong_type])
        assert len(safe) == 1

    def test_multiple_candidates(self, weak_type, strong_type, equal_type):
        safe = safe_downgrade_set(
            weak_type, candidates=[strong_type, equal_type, weak_type]
        )
        assert len(safe) >= 2  # strong and equal are subtypes of weak

    def test_default_candidates(self, end_type):
        safe = safe_downgrade_set(end_type)
        assert len(safe) >= 1  # At least end itself


# ---------------------------------------------------------------------------
# detect_forced_downgrade
# ---------------------------------------------------------------------------


class TestDetectForcedDowngrade:
    """Tests for detect_forced_downgrade."""

    def test_no_attacker_labels(self):
        ss = build_statespace(parse("&{a: end, b: end}"))
        result = detect_forced_downgrade(ss, set())
        assert result.is_vulnerable is False

    def test_attacker_controls_selection(self):
        ss = build_statespace(parse("+{ok: &{a: end, b: end}, err: end}"))
        result = detect_forced_downgrade(ss, {"ok", "err"})
        assert isinstance(result, ForcedDowngradeResult)
        assert result.attacker_labels == frozenset({"ok", "err"})

    def test_end_not_vulnerable(self):
        ss = build_statespace(parse("end"))
        result = detect_forced_downgrade(ss, {"anything"})
        assert result.is_vulnerable is False

    def test_forced_paths_nonempty_when_vulnerable(self):
        ss = build_statespace(parse("+{ok: &{a: end, b: end}, err: end}"))
        result = detect_forced_downgrade(ss, {"ok", "err"})
        if result.is_vulnerable:
            assert len(result.forced_paths) > 0

    def test_weakest_reachable_set(self):
        ss = build_statespace(parse("+{ok: &{a: end, b: end}, err: end}"))
        result = detect_forced_downgrade(ss, {"ok", "err"})
        assert result.weakest_reachable is not None


# ---------------------------------------------------------------------------
# analyze_downgrade
# ---------------------------------------------------------------------------


class TestAnalyzeDowngrade:
    """Tests for analyze_downgrade."""

    def test_result_type(self, strong_type, weak_type):
        result = analyze_downgrade(strong_type, weak_type)
        assert isinstance(result, DowngradeResult)

    def test_safe_analysis(self, weak_type, strong_type):
        result = analyze_downgrade(weak_type, strong_type)
        assert result.is_safe is True
        assert result.is_subtype_relation is True

    def test_unsafe_analysis(self, strong_type, weak_type):
        result = analyze_downgrade(strong_type, weak_type)
        assert result.is_safe is False

    def test_risk_in_result(self, strong_type, weak_type):
        result = analyze_downgrade(strong_type, weak_type)
        assert isinstance(result.risk, DowngradeRisk)
        assert result.risk.risk_score > 0.0

    def test_state_counts(self, strong_type, weak_type):
        result = analyze_downgrade(strong_type, weak_type)
        assert result.strong_state_count > 0
        assert result.weak_state_count > 0

    def test_type_strings(self, strong_type, weak_type):
        result = analyze_downgrade(strong_type, weak_type)
        assert len(result.required_type) > 0
        assert len(result.offered_type) > 0

    def test_with_candidates(self, strong_type, weak_type):
        result = analyze_downgrade(
            weak_type, strong_type, candidates=[strong_type, weak_type]
        )
        assert len(result.safe_replacements) >= 1

    def test_end_analysis(self, end_type):
        result = analyze_downgrade(end_type, end_type)
        assert result.is_safe is True
        assert result.risk.risk_score == 0.0
