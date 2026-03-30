"""Tests for starvation detection via spectral analysis (Step 80d)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.starvation import (
    StarvationResult,
    StarvedTransition,
    RoleSpectralGap,
    detect_starvation,
    per_role_spectral_gap,
    fairness_score,
    pendular_fairness,
    is_starvation_free,
    analyze_starvation,
)


# ---------------------------------------------------------------------------
# Helper: manually-constructed state spaces
# ---------------------------------------------------------------------------

def _make_starved_ss() -> StateSpace:
    """State space with a starved transition.

    0 (top) -> 1 -> 3 (bottom)
    0 -> 2 (dead end, cannot reach bottom)
    2 -> 2 (self-loop, starved)
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "a", 1),
        (0, "b", 2),
        (1, "c", 3),
        (2, "spin", 2),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "mid", 2: "dead", 3: "end"}
    return ss


def _make_all_reachable_ss() -> StateSpace:
    """State space where all transitions are on top-to-bottom paths.

    0 -> 1 -> 3
    0 -> 2 -> 3
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "a", 1),
        (0, "b", 2),
        (1, "c", 3),
        (2, "d", 3),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "left", 2: "right", 3: "end"}
    return ss


def _make_unreachable_from_top() -> StateSpace:
    """State space with a state not reachable from top.

    0 -> 1 -> 2 (bottom)
    3 -> 2 (state 3 unreachable from top)
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "a", 1),
        (1, "b", 2),
        (3, "c", 2),
    ]
    ss.top = 0
    ss.bottom = 2
    ss.labels = {0: "top", 1: "mid", 2: "end", 3: "orphan"}
    return ss


def _make_linear() -> StateSpace:
    """Simple linear: 0 -> 1 -> 2."""
    ss = StateSpace()
    ss.states = {0, 1, 2}
    ss.transitions = [
        (0, "a", 1),
        (1, "b", 2),
    ]
    ss.top = 0
    ss.bottom = 2
    ss.labels = {0: "top", 1: "mid", 2: "end"}
    return ss


def _make_mixed_polarity() -> StateSpace:
    """State space with both branch and select transitions."""
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "a", 1),
        (0, "b", 2),
        (1, "c", 3),
        (2, "d", 3),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.selection_transitions = {(0, "b", 2)}
    ss.labels = {0: "top", 1: "left", 2: "right", 3: "end"}
    return ss


def _make_wide_branch() -> StateSpace:
    """Wide branching: top -> many children -> bottom."""
    ss = StateSpace()
    ss.states = set(range(7))
    ss.transitions = [
        (0, "a", 1),
        (0, "b", 2),
        (0, "c", 3),
        (0, "d", 4),
        (0, "e", 5),
        (1, "f", 6),
        (2, "f", 6),
        (3, "f", 6),
        (4, "f", 6),
        (5, "f", 6),
    ]
    ss.top = 0
    ss.bottom = 6
    ss.labels = {i: f"s{i}" for i in range(7)}
    return ss


# ---------------------------------------------------------------------------
# Tests: detect_starvation
# ---------------------------------------------------------------------------

class TestDetectStarvation:
    """Test the detect_starvation function."""

    def test_starved_transitions(self):
        ss = _make_starved_ss()
        starved = detect_starvation(ss)
        # The transitions involving state 2 are starved
        starved_tuples = {(s.src, s.label, s.tgt) for s in starved}
        assert (0, "b", 2) in starved_tuples
        assert (2, "spin", 2) in starved_tuples

    def test_all_reachable(self):
        ss = _make_all_reachable_ss()
        starved = detect_starvation(ss)
        assert len(starved) == 0

    def test_unreachable_source(self):
        ss = _make_unreachable_from_top()
        starved = detect_starvation(ss)
        assert len(starved) == 1
        assert starved[0].src == 3
        assert starved[0].label == "c"

    def test_linear_no_starvation(self):
        ss = _make_linear()
        starved = detect_starvation(ss)
        assert len(starved) == 0

    def test_returns_starved_transition_type(self):
        ss = _make_starved_ss()
        starved = detect_starvation(ss)
        for s in starved:
            assert isinstance(s, StarvedTransition)
            assert isinstance(s.reason, str)

    def test_empty_ss(self):
        ss = StateSpace()
        ss.states = {0}
        ss.transitions = []
        ss.top = 0
        ss.bottom = 0
        starved = detect_starvation(ss)
        assert len(starved) == 0

    def test_reason_includes_context(self):
        ss = _make_starved_ss()
        starved = detect_starvation(ss)
        reasons = [s.reason for s in starved]
        # At least one should mention "cannot reach bottom"
        assert any("bottom" in r or "top" in r for r in reasons)


# ---------------------------------------------------------------------------
# Tests: per_role_spectral_gap
# ---------------------------------------------------------------------------

class TestPerRoleSpectralGap:
    """Test spectral gap per label."""

    def test_all_reachable_labels(self):
        ss = _make_all_reachable_ss()
        gaps = per_role_spectral_gap(ss)
        assert len(gaps) > 0
        for label, rsg in gaps.items():
            assert isinstance(rsg, RoleSpectralGap)
            assert rsg.label == label
            assert not rsg.is_starved

    def test_starved_label_detected(self):
        ss = _make_starved_ss()
        gaps = per_role_spectral_gap(ss)
        # "spin" label only has starved transitions
        assert "spin" in gaps
        assert gaps["spin"].is_starved

    def test_spectral_gap_range(self):
        ss = _make_all_reachable_ss()
        gaps = per_role_spectral_gap(ss)
        for label, rsg in gaps.items():
            assert 0.0 <= rsg.spectral_gap <= 1.0

    def test_count_matches(self):
        ss = _make_all_reachable_ss()
        gaps = per_role_spectral_gap(ss)
        total = sum(rsg.count for rsg in gaps.values())
        assert total == len(ss.transitions)

    def test_wide_branch_labels(self):
        ss = _make_wide_branch()
        gaps = per_role_spectral_gap(ss)
        # "f" label used 5 times
        assert "f" in gaps
        assert gaps["f"].count == 5


# ---------------------------------------------------------------------------
# Tests: fairness_score
# ---------------------------------------------------------------------------

class TestFairnessScore:
    """Test the fairness_score function."""

    def test_linear_fairness(self):
        ss = _make_linear()
        score = fairness_score(ss)
        assert 0.0 <= score <= 1.0

    def test_all_reachable_fairness(self):
        ss = _make_all_reachable_ss()
        score = fairness_score(ss)
        assert 0.0 <= score <= 1.0

    def test_starved_lower_fairness(self):
        ss = _make_starved_ss()
        score = fairness_score(ss)
        # With starved transitions, fairness should be lower
        assert score < 1.0

    def test_empty_perfect_fairness(self):
        ss = StateSpace()
        ss.states = {0}
        ss.transitions = []
        ss.top = 0
        ss.bottom = 0
        score = fairness_score(ss)
        assert score == 1.0

    def test_wide_branch_fairness(self):
        ss = _make_wide_branch()
        score = fairness_score(ss)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests: pendular_fairness
# ---------------------------------------------------------------------------

class TestPendularFairness:
    """Test the pendular_fairness function."""

    def test_all_branch_transitions(self):
        ss = _make_all_reachable_ss()
        balance = pendular_fairness(ss)
        # All transitions are branch (no selections)
        # With no selections, min(branch, select) = 0, so ratio = 0
        assert 0.0 <= balance <= 1.0

    def test_mixed_polarity(self):
        ss = _make_mixed_polarity()
        balance = pendular_fairness(ss)
        assert 0.0 < balance <= 1.0

    def test_empty_protocol(self):
        ss = StateSpace()
        ss.states = {0}
        ss.transitions = []
        ss.top = 0
        ss.bottom = 0
        balance = pendular_fairness(ss)
        assert balance == 1.0

    def test_range(self):
        ss = _make_wide_branch()
        balance = pendular_fairness(ss)
        assert 0.0 <= balance <= 1.0


# ---------------------------------------------------------------------------
# Tests: is_starvation_free
# ---------------------------------------------------------------------------

class TestIsStarvationFree:
    """Test the is_starvation_free predicate."""

    def test_starved_not_free(self):
        ss = _make_starved_ss()
        assert not is_starvation_free(ss)

    def test_all_reachable_free(self):
        ss = _make_all_reachable_ss()
        assert is_starvation_free(ss)

    def test_linear_free(self):
        ss = _make_linear()
        assert is_starvation_free(ss)

    def test_unreachable_source_not_free(self):
        ss = _make_unreachable_from_top()
        assert not is_starvation_free(ss)


# ---------------------------------------------------------------------------
# Tests: analyze_starvation
# ---------------------------------------------------------------------------

class TestAnalyzeStarvation:
    """Test comprehensive starvation analysis."""

    def test_result_type(self):
        ss = _make_starved_ss()
        result = analyze_starvation(ss)
        assert isinstance(result, StarvationResult)

    def test_starved_analysis(self):
        ss = _make_starved_ss()
        result = analyze_starvation(ss)
        assert not result.is_starvation_free
        assert result.num_starved > 0
        assert result.coverage_ratio < 1.0

    def test_clean_analysis(self):
        ss = _make_all_reachable_ss()
        result = analyze_starvation(ss)
        assert result.is_starvation_free
        assert result.num_starved == 0
        assert result.coverage_ratio == 1.0

    def test_summary_message_starved(self):
        ss = _make_starved_ss()
        result = analyze_starvation(ss)
        assert "STARVATION" in result.summary

    def test_summary_message_free(self):
        ss = _make_all_reachable_ss()
        result = analyze_starvation(ss)
        assert "free" in result.summary.lower()

    def test_reachable_states(self):
        ss = _make_all_reachable_ss()
        result = analyze_starvation(ss)
        assert result.reachable_states == frozenset({0, 1, 2, 3})
        assert len(result.unreachable_states) == 0

    def test_unreachable_states(self):
        ss = _make_unreachable_from_top()
        result = analyze_starvation(ss)
        assert 3 in result.unreachable_states

    def test_total_transitions(self):
        ss = _make_starved_ss()
        result = analyze_starvation(ss)
        assert result.num_total_transitions == 4

    def test_spectral_gaps_present(self):
        ss = _make_all_reachable_ss()
        result = analyze_starvation(ss)
        assert len(result.role_spectral_gaps) > 0

    def test_fairness_in_result(self):
        ss = _make_all_reachable_ss()
        result = analyze_starvation(ss)
        assert 0.0 <= result.fairness <= 1.0

    def test_pendular_in_result(self):
        ss = _make_all_reachable_ss()
        result = analyze_starvation(ss)
        assert 0.0 <= result.pendular_balance <= 1.0


# ---------------------------------------------------------------------------
# Tests with parsed session types
# ---------------------------------------------------------------------------

class TestParsedTypes:
    """Test starvation detection on parsed session types."""

    def test_simple_branch(self):
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        assert is_starvation_free(ss)

    def test_recursive_safe(self):
        ast = parse("rec X . &{a: X, b: end}")
        ss = build_statespace(ast)
        result = analyze_starvation(ss)
        # Recursive types may have starved transitions depending on SCC handling
        assert isinstance(result, StarvationResult)

    def test_selection(self):
        ast = parse("+{ok: end, err: end}")
        ss = build_statespace(ast)
        assert is_starvation_free(ss)

    def test_nested_branch(self):
        ast = parse("&{a: &{b: end, c: end}, d: end}")
        ss = build_statespace(ast)
        assert is_starvation_free(ss)

    def test_parallel(self):
        ast = parse("(&{a: end} || &{b: end})")
        ss = build_statespace(ast)
        result = analyze_starvation(ss)
        assert isinstance(result, StarvationResult)

    def test_end_type(self):
        ast = parse("end")
        ss = build_statespace(ast)
        assert is_starvation_free(ss)

    def test_fairness_on_parsed(self):
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        score = fairness_score(ss)
        assert 0.0 <= score <= 1.0

    def test_spectral_gap_on_parsed(self):
        ast = parse("&{a: end, b: end, c: end}")
        ss = build_statespace(ast)
        gaps = per_role_spectral_gap(ss)
        assert len(gaps) > 0
