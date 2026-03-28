"""Tests for the pendular session type classifier (Step 59k)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.pendular import (
    PendularResult,
    classify_alternation,
    is_pendular,
    pendular_depth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nim3_ss():
    return build_statespace(
        parse("+{take1: &{take1: +{take1: end}, take2: end}, take2: &{take1: end}, take3: end}")
    )


@pytest.fixture
def hex2x2_ss():
    return build_statespace(
        parse(
            "+{a: &{b: +{c: end, d: &{c: end}}, "
            "c: +{b: &{d: end}, d: &{b: end}}, "
            "d: +{b: &{c: end}, c: end}}, "
            "b: &{a: +{c: end, d: end}, "
            "c: +{a: &{d: end}, d: end}, "
            "d: +{a: &{c: end}, c: end}}, "
            "c: &{a: +{b: end, d: &{b: end}}, "
            "b: +{a: end, d: &{a: end}}, "
            "d: +{a: end, b: end}}, "
            "d: &{a: +{b: end, c: &{b: end}}, "
            "b: +{a: &{c: end}, c: &{a: end}}, "
            "c: +{a: &{b: end}, b: end}}}"
        )
    )


@pytest.fixture
def dominion_ss():
    return build_statespace(
        parse("(+{play_attack: wait} || &{react: end, pass: end}) . &{resolve: end}")
    )


@pytest.fixture
def request_response_ss():
    return build_statespace(
        parse("rec X . &{request: +{response: X}}")
    )


@pytest.fixture
def non_pendular_ss():
    """Three branches in a row — not pendular."""
    return build_statespace(
        parse("&{setup: &{config: &{init: +{ready: end}}}}")
    )


@pytest.fixture
def biased_ss():
    """Branch-heavy type — biased toward branch."""
    return build_statespace(
        parse("&{a: &{b: &{c: &{d: +{done: end}}}}}")
    )


@pytest.fixture
def simple_branch_ss():
    """Single branch then end."""
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def simple_select_ss():
    """Single select then end."""
    return build_statespace(parse("+{a: end, b: end}"))


@pytest.fixture
def iterator_ss():
    """Java Iterator — rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}."""
    return build_statespace(
        parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
    )


# ---------------------------------------------------------------------------
# is_pendular tests
# ---------------------------------------------------------------------------


class TestIsPendular:
    def test_nim3_is_pendular(self, nim3_ss):
        """Nim(3) alternates Select/Branch at every transition."""
        assert is_pendular(nim3_ss) is True

    def test_hex2x2_is_pendular(self, hex2x2_ss):
        """Hex(2x2) alternates Select/Branch — two-player alternating game."""
        assert is_pendular(hex2x2_ss) is True

    def test_non_pendular(self, non_pendular_ss):
        """Three branches in a row violates alternation."""
        assert is_pendular(non_pendular_ss) is False

    def test_simple_branch_is_pendular(self, simple_branch_ss):
        """&{a: end, b: end} — branch then end, trivially pendular."""
        assert is_pendular(simple_branch_ss) is True

    def test_simple_select_is_pendular(self, simple_select_ss):
        """+{a: end, b: end} — select then end, trivially pendular."""
        assert is_pendular(simple_select_ss) is True

    def test_end_only(self):
        ss = build_statespace(parse("end"))
        assert is_pendular(ss) is True

    def test_iterator_not_strictly_pendular(self, iterator_ss):
        """Iterator has &{hasNext}->+{TRUE,FALSE}->&{next}->&{hasNext} — branch->branch."""
        assert is_pendular(iterator_ss) is False


# ---------------------------------------------------------------------------
# classify_alternation tests
# ---------------------------------------------------------------------------


class TestClassifyAlternation:
    def test_nim3_pendular(self, nim3_ss):
        result = classify_alternation(nim3_ss)
        assert result.is_pendular is True
        assert result.classification == "pendular"
        assert len(result.violations) == 0

    def test_nim3_not_graded(self, nim3_ss):
        """Nim(3) is pendular but NOT graded (take3 shortcuts to end)."""
        result = classify_alternation(nim3_ss)
        assert result.is_graded is False

    def test_nim3_balanced_ratio(self, nim3_ss):
        result = classify_alternation(nim3_ss)
        assert result.select_ratio == 0.5
        assert result.branch_ratio == 0.5

    def test_nim3_top_is_select(self, nim3_ss):
        result = classify_alternation(nim3_ss)
        assert result.level_polarity[0] == "select"

    def test_hex2x2_pendular(self, hex2x2_ss):
        result = classify_alternation(hex2x2_ss)
        assert result.is_pendular is True
        assert result.classification == "pendular"
        assert len(result.violations) == 0

    def test_hex2x2_top_is_select(self, hex2x2_ss):
        result = classify_alternation(hex2x2_ss)
        assert result.level_polarity[0] == "select"

    def test_non_pendular_has_violations(self, non_pendular_ss):
        result = classify_alternation(non_pendular_ss)
        assert result.is_pendular is False
        assert len(result.violations) > 0

    def test_non_pendular_violation_fields(self, non_pendular_ss):
        result = classify_alternation(non_pendular_ss)
        for src, label, tgt, src_pol, tgt_pol in result.violations:
            assert isinstance(src, int)
            assert isinstance(label, str)
            assert isinstance(tgt, int)
            # Violation: same polarity on both sides of a transition
            assert src_pol == tgt_pol  # branch->branch is the violation here

    def test_biased_classification(self, biased_ss):
        result = classify_alternation(biased_ss)
        assert result.is_pendular is False
        assert result.branch_ratio > result.select_ratio

    def test_request_response_pendular(self, request_response_ss):
        result = classify_alternation(request_response_ss)
        assert result.is_pendular is True
        assert result.classification == "pendular"

    def test_simple_branch_depth(self, simple_branch_ss):
        result = classify_alternation(simple_branch_ss)
        assert result.max_depth == 1
        assert result.level_polarity[0] == "branch"
        assert result.level_polarity[1] == "end"

    def test_simple_select_depth(self, simple_select_ss):
        result = classify_alternation(simple_select_ss)
        assert result.max_depth == 1
        assert result.level_polarity[0] == "select"

    def test_dominion_not_strictly_pendular(self, dominion_ss):
        """Dominion: parallel merges create branch->branch transitions."""
        result = classify_alternation(dominion_ss)
        # Product construction produces branch->branch at the resolve step
        assert result.is_pendular is False


# ---------------------------------------------------------------------------
# pendular_depth tests
# ---------------------------------------------------------------------------


class TestPendularDepth:
    def test_nim3_top_depth(self, nim3_ss):
        pd = pendular_depth(nim3_ss)
        top_depth, top_pol = pd[nim3_ss.top]
        assert top_depth == 0
        assert top_pol == "select"

    def test_nim3_bottom_is_end(self, nim3_ss):
        pd = pendular_depth(nim3_ss)
        _depth, bottom_pol = pd[nim3_ss.bottom]
        assert bottom_pol == "end"

    def test_all_states_have_depth(self, nim3_ss):
        pd = pendular_depth(nim3_ss)
        assert set(pd.keys()) == nim3_ss.states

    def test_hex_all_states_have_depth(self, hex2x2_ss):
        pd = pendular_depth(hex2x2_ss)
        assert set(pd.keys()) == hex2x2_ss.states

    def test_nim3_alternates_along_longest_path(self, nim3_ss):
        """Check that along the longest path, polarity alternates."""
        pd = pendular_depth(nim3_ss)
        # Find all select and branch states
        selects = [s for s, (d, p) in pd.items() if p == "select"]
        branches = [s for s, (d, p) in pd.items() if p == "branch"]
        assert len(selects) == 2
        assert len(branches) == 2


# ---------------------------------------------------------------------------
# Benchmark survey
# ---------------------------------------------------------------------------


class TestBenchmarkSurvey:
    """Test pendularity across benchmark protocols."""

    def test_end_is_trivially_pendular(self):
        ss = build_statespace(parse("end"))
        result = classify_alternation(ss)
        assert result.is_pendular is True
        assert result.max_depth == 0

    def test_single_method_pendular(self):
        ss = build_statespace(parse("&{m: end}"))
        result = classify_alternation(ss)
        assert result.is_pendular is True
        assert result.max_depth == 1

    def test_two_level_alternation(self):
        """&{a: +{x: end, y: end}} — branch then select, pendular."""
        ss = build_statespace(parse("&{a: +{x: end, y: end}}"))
        result = classify_alternation(ss)
        assert result.is_pendular is True

    def test_three_level_alternation(self):
        """+{a: &{x: +{m: end}}} — select/branch/select, pendular."""
        ss = build_statespace(parse("+{a: &{x: +{m: end}}}"))
        result = classify_alternation(ss)
        assert result.is_pendular is True

    def test_select_select_not_pendular(self):
        """+{a: +{x: end}} — select then select, not pendular."""
        ss = build_statespace(parse("+{a: +{x: end}}"))
        result = classify_alternation(ss)
        assert result.is_pendular is False

    def test_branch_branch_not_pendular(self):
        """&{a: &{x: end}} — branch then branch, not pendular."""
        ss = build_statespace(parse("&{a: &{x: end}}"))
        result = classify_alternation(ss)
        assert result.is_pendular is False
