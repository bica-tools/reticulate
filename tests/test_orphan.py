"""Tests for orphaned session detection (Step 80e)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.orphan import (
    OrphanResult,
    OrphanState,
    RepairSuggestion,
    detect_orphans,
    orphan_depth,
    is_orphan_free,
    repair_orphans,
    analyze_orphans,
)


# ---------------------------------------------------------------------------
# Helper: manually-constructed state spaces
# ---------------------------------------------------------------------------

def _make_orphan_ss() -> StateSpace:
    """State space with orphaned states.

    0 (top) -> 1 -> 3 (bottom)
    0 -> 2 (dead end: no path to bottom)
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "a", 1),
        (0, "b", 2),
        (1, "c", 3),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "mid", 2: "dead_end", 3: "end"}
    return ss


def _make_cyclic_orphan() -> StateSpace:
    """State space with a cyclic orphan (trapped cycle).

    0 -> 1 -> 2 -> 1 (cycle, no exit to bottom)
    0 -> 3 (bottom)
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "a", 1),
        (0, "b", 3),
        (1, "x", 2),
        (2, "y", 1),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "cyc_a", 2: "cyc_b", 3: "end"}
    return ss


def _make_clean_ss() -> StateSpace:
    """State space with no orphans.

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


def _make_disconnected_orphan() -> StateSpace:
    """State space with state unreachable from top AND unreachable from bottom.

    0 -> 1 -> 2 (bottom)
    4 (completely disconnected)
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 4}
    ss.transitions = [
        (0, "a", 1),
        (1, "b", 2),
    ]
    ss.top = 0
    ss.bottom = 2
    ss.labels = {0: "top", 1: "mid", 2: "end", 4: "lost"}
    return ss


def _make_self_loop_orphan() -> StateSpace:
    """State with only a self-loop, no path to bottom.

    0 -> 1 (self-loop only), 0 -> 2 (bottom)
    """
    ss = StateSpace()
    ss.states = {0, 1, 2}
    ss.transitions = [
        (0, "a", 1),
        (0, "b", 2),
        (1, "spin", 1),
    ]
    ss.top = 0
    ss.bottom = 2
    ss.labels = {0: "top", 1: "spinner", 2: "end"}
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


def _make_multiple_orphans() -> StateSpace:
    """Multiple disconnected orphan groups.

    0 -> 1 -> 5 (bottom)
    0 -> 2 (dead end)
    0 -> 3 -> 4 -> 3 (trapped cycle)
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3, 4, 5}
    ss.transitions = [
        (0, "a", 1),
        (0, "b", 2),
        (0, "c", 3),
        (1, "d", 5),
        (3, "e", 4),
        (4, "f", 3),
    ]
    ss.top = 0
    ss.bottom = 5
    ss.labels = {0: "top", 1: "safe", 2: "dead", 3: "cyc_a", 4: "cyc_b", 5: "end"}
    return ss


# ---------------------------------------------------------------------------
# Tests: detect_orphans
# ---------------------------------------------------------------------------

class TestDetectOrphans:
    """Test the detect_orphans function."""

    def test_dead_end_orphan(self):
        ss = _make_orphan_ss()
        orphans = detect_orphans(ss)
        assert 2 in orphans
        assert 0 not in orphans
        assert 1 not in orphans
        assert 3 not in orphans

    def test_cyclic_orphan(self):
        ss = _make_cyclic_orphan()
        orphans = detect_orphans(ss)
        assert 1 in orphans
        assert 2 in orphans

    def test_no_orphans(self):
        ss = _make_clean_ss()
        orphans = detect_orphans(ss)
        assert len(orphans) == 0

    def test_disconnected_orphan(self):
        ss = _make_disconnected_orphan()
        orphans = detect_orphans(ss)
        assert 4 in orphans

    def test_self_loop_orphan(self):
        ss = _make_self_loop_orphan()
        orphans = detect_orphans(ss)
        assert 1 in orphans

    def test_returns_frozenset(self):
        ss = _make_orphan_ss()
        orphans = detect_orphans(ss)
        assert isinstance(orphans, frozenset)

    def test_single_state(self):
        ss = StateSpace()
        ss.states = {0}
        ss.transitions = []
        ss.top = 0
        ss.bottom = 0
        orphans = detect_orphans(ss)
        assert len(orphans) == 0

    def test_multiple_orphan_groups(self):
        ss = _make_multiple_orphans()
        orphans = detect_orphans(ss)
        assert 2 in orphans
        assert 3 in orphans
        assert 4 in orphans
        assert len(orphans) == 3


# ---------------------------------------------------------------------------
# Tests: orphan_depth
# ---------------------------------------------------------------------------

class TestOrphanDepth:
    """Test orphan depth computation."""

    def test_direct_orphan_depth(self):
        ss = _make_orphan_ss()
        depth = orphan_depth(ss, 2)
        assert depth == 1  # 0 -> 2

    def test_cyclic_orphan_depth(self):
        ss = _make_cyclic_orphan()
        depth = orphan_depth(ss, 1)
        assert depth == 1  # 0 -> 1

    def test_unreachable_orphan_depth(self):
        ss = _make_disconnected_orphan()
        depth = orphan_depth(ss, 4)
        assert depth == -1

    def test_safe_state_depth(self):
        ss = _make_clean_ss()
        depth = orphan_depth(ss, 1)
        assert depth == 1

    def test_top_depth_zero(self):
        ss = _make_clean_ss()
        depth = orphan_depth(ss, 0)
        assert depth == 0


# ---------------------------------------------------------------------------
# Tests: is_orphan_free
# ---------------------------------------------------------------------------

class TestIsOrphanFree:
    """Test the is_orphan_free predicate."""

    def test_has_orphans(self):
        ss = _make_orphan_ss()
        assert not is_orphan_free(ss)

    def test_clean(self):
        ss = _make_clean_ss()
        assert is_orphan_free(ss)

    def test_linear(self):
        ss = _make_linear()
        assert is_orphan_free(ss)

    def test_cyclic_orphan(self):
        ss = _make_cyclic_orphan()
        assert not is_orphan_free(ss)

    def test_self_loop_orphan(self):
        ss = _make_self_loop_orphan()
        assert not is_orphan_free(ss)


# ---------------------------------------------------------------------------
# Tests: repair_orphans
# ---------------------------------------------------------------------------

class TestRepairOrphans:
    """Test orphan repair suggestions."""

    def test_no_repairs_needed(self):
        ss = _make_clean_ss()
        repairs = repair_orphans(ss)
        assert len(repairs) == 0

    def test_dead_end_repair(self):
        ss = _make_orphan_ss()
        repairs = repair_orphans(ss)
        assert len(repairs) >= 1
        # Should suggest connecting to bottom
        assert any(r.tgt == ss.bottom for r in repairs)

    def test_repair_type(self):
        ss = _make_orphan_ss()
        repairs = repair_orphans(ss)
        for r in repairs:
            assert isinstance(r, RepairSuggestion)
            assert isinstance(r.rationale, str)

    def test_cyclic_repair(self):
        ss = _make_cyclic_orphan()
        repairs = repair_orphans(ss)
        assert len(repairs) >= 1

    def test_multiple_groups_repairs(self):
        ss = _make_multiple_orphans()
        repairs = repair_orphans(ss)
        # Dead end (2) and cycle (3,4) are separate components
        assert len(repairs) >= 2

    def test_repair_target_is_bottom(self):
        ss = _make_orphan_ss()
        repairs = repair_orphans(ss)
        for r in repairs:
            assert r.tgt == ss.bottom

    def test_repair_rationale_nonempty(self):
        ss = _make_orphan_ss()
        repairs = repair_orphans(ss)
        for r in repairs:
            assert len(r.rationale) > 0


# ---------------------------------------------------------------------------
# Tests: analyze_orphans
# ---------------------------------------------------------------------------

class TestAnalyzeOrphans:
    """Test comprehensive orphan analysis."""

    def test_result_type(self):
        ss = _make_orphan_ss()
        result = analyze_orphans(ss)
        assert isinstance(result, OrphanResult)

    def test_orphan_analysis(self):
        ss = _make_orphan_ss()
        result = analyze_orphans(ss)
        assert not result.is_orphan_free
        assert result.num_orphans == 1
        assert result.num_total_states == 4
        assert result.orphan_ratio == 0.25

    def test_clean_analysis(self):
        ss = _make_clean_ss()
        result = analyze_orphans(ss)
        assert result.is_orphan_free
        assert result.num_orphans == 0
        assert result.orphan_ratio == 0.0

    def test_summary_orphans(self):
        ss = _make_orphan_ss()
        result = analyze_orphans(ss)
        assert "ORPHAN" in result.summary

    def test_summary_clean(self):
        ss = _make_clean_ss()
        result = analyze_orphans(ss)
        assert "free" in result.summary.lower()

    def test_orphan_state_details(self):
        ss = _make_orphan_ss()
        result = analyze_orphans(ss)
        assert len(result.orphaned_states) == 1
        orphan = result.orphaned_states[0]
        assert isinstance(orphan, OrphanState)
        assert orphan.state == 2
        assert orphan.depth_from_top == 1
        assert not orphan.has_outgoing  # dead end
        assert not orphan.is_cyclic

    def test_cyclic_orphan_details(self):
        ss = _make_cyclic_orphan()
        result = analyze_orphans(ss)
        cyclic_orphans = [o for o in result.orphaned_states if o.is_cyclic]
        assert len(cyclic_orphans) >= 1

    def test_max_orphan_depth(self):
        ss = _make_orphan_ss()
        result = analyze_orphans(ss)
        assert result.max_orphan_depth == 1

    def test_repairs_present(self):
        ss = _make_orphan_ss()
        result = analyze_orphans(ss)
        assert result.num_repairs_needed >= 1

    def test_reachable_from_top_orphans(self):
        ss = _make_orphan_ss()
        result = analyze_orphans(ss)
        assert 2 in result.reachable_from_top_orphans

    def test_unreachable_from_top_orphans(self):
        ss = _make_disconnected_orphan()
        result = analyze_orphans(ss)
        assert 4 in result.unreachable_from_top_orphans


# ---------------------------------------------------------------------------
# Tests with parsed session types
# ---------------------------------------------------------------------------

class TestParsedTypes:
    """Test orphan detection on parsed session types."""

    def test_simple_branch_no_orphans(self):
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        assert is_orphan_free(ss)

    def test_recursive_with_exit(self):
        ast = parse("rec X . &{a: X, b: end}")
        ss = build_statespace(ast)
        assert is_orphan_free(ss)

    def test_selection(self):
        ast = parse("+{ok: end, err: end}")
        ss = build_statespace(ast)
        assert is_orphan_free(ss)

    def test_nested_branch(self):
        ast = parse("&{a: &{b: end, c: end}, d: end}")
        ss = build_statespace(ast)
        assert is_orphan_free(ss)

    def test_parallel(self):
        ast = parse("(&{a: end} || &{b: end})")
        ss = build_statespace(ast)
        assert is_orphan_free(ss)

    def test_end_type(self):
        ast = parse("end")
        ss = build_statespace(ast)
        assert is_orphan_free(ss)

    def test_analyze_on_parsed(self):
        ast = parse("&{a: &{b: end}, c: end}")
        ss = build_statespace(ast)
        result = analyze_orphans(ss)
        assert result.is_orphan_free

    def test_depth_on_parsed(self):
        ast = parse("&{a: &{b: end}}")
        ss = build_statespace(ast)
        # All states should have non-negative depth
        for s in ss.states:
            d = orphan_depth(ss, s)
            assert d >= 0
