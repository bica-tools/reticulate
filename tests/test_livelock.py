"""Tests for livelock detection via tropical analysis (Step 80c)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.livelock import (
    LivelockResult,
    LivelockSCC,
    HeatKernelLivelockIndicator,
    detect_livelock,
    livelock_sccs,
    tropical_livelock_score,
    heat_kernel_livelock,
    is_livelock_free,
    analyze_livelock,
)


# ---------------------------------------------------------------------------
# Helper: build a manually-constructed state space with livelock
# ---------------------------------------------------------------------------

def _make_livelocked_ss() -> StateSpace:
    """State space with a trapped cycle: 0 -> 1 -> 2 -> 1, bottom = 3.

    State 0 (top) -> state 1 -> state 2 -> back to state 1.
    States 1 and 2 form a cycle with no exit to bottom (state 3).
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "enter", 1),
        (1, "loop_a", 2),
        (2, "loop_b", 1),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "cycle_a", 2: "cycle_b", 3: "end"}
    return ss


def _make_livelock_free_ss() -> StateSpace:
    """State space where cycle has an exit to bottom.

    0 -> 1 -> 2 -> 1 (cycle), but 2 -> 3 (exit to bottom).
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "enter", 1),
        (1, "loop_a", 2),
        (2, "loop_b", 1),
        (2, "exit", 3),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "cycle_a", 2: "cycle_b", 3: "end"}
    return ss


def _make_self_loop_livelock() -> StateSpace:
    """State space with a self-loop livelock: 0 -> 1 -> 1, bottom = 2."""
    ss = StateSpace()
    ss.states = {0, 1, 2}
    ss.transitions = [
        (0, "enter", 1),
        (1, "spin", 1),
    ]
    ss.top = 0
    ss.bottom = 2
    ss.labels = {0: "top", 1: "spinner", 2: "end"}
    return ss


def _make_multiple_sccs_ss() -> StateSpace:
    """State space with two trapped SCCs and one safe SCC.

    SCC1: {1, 2} trapped cycle
    SCC2: {4, 5} trapped cycle
    SCC3: {6, 7} safe cycle (7 -> 8 = bottom)
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 4, 5, 6, 7, 8}
    ss.transitions = [
        (0, "a", 1),
        (0, "b", 4),
        (0, "c", 6),
        (1, "x", 2),
        (2, "y", 1),
        (4, "p", 5),
        (5, "q", 4),
        (6, "r", 7),
        (7, "s", 6),
        (7, "exit", 8),
    ]
    ss.top = 0
    ss.bottom = 8
    ss.labels = {0: "top", 1: "c1a", 2: "c1b", 4: "c2a", 5: "c2b",
                 6: "c3a", 7: "c3b", 8: "end"}
    return ss


def _make_simple_linear() -> StateSpace:
    """Linear: 0 -> 1 -> 2 (bottom). No cycles."""
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


# ---------------------------------------------------------------------------
# Tests: detect_livelock
# ---------------------------------------------------------------------------

class TestDetectLivelock:
    """Test the detect_livelock function."""

    def test_trapped_cycle(self):
        ss = _make_livelocked_ss()
        result = detect_livelock(ss)
        assert 1 in result
        assert 2 in result
        assert 0 not in result
        assert 3 not in result

    def test_safe_cycle(self):
        ss = _make_livelock_free_ss()
        result = detect_livelock(ss)
        assert len(result) == 0

    def test_linear_no_livelock(self):
        ss = _make_simple_linear()
        result = detect_livelock(ss)
        assert len(result) == 0

    def test_self_loop_livelock(self):
        ss = _make_self_loop_livelock()
        result = detect_livelock(ss)
        assert 1 in result

    def test_multiple_trapped_sccs(self):
        ss = _make_multiple_sccs_ss()
        result = detect_livelock(ss)
        assert 1 in result
        assert 2 in result
        assert 4 in result
        assert 5 in result
        # Safe cycle {6, 7} should not be livelocked
        assert 6 not in result
        assert 7 not in result

    def test_returns_frozenset(self):
        ss = _make_livelocked_ss()
        result = detect_livelock(ss)
        assert isinstance(result, frozenset)

    def test_end_only(self):
        """Single end state: no livelock."""
        ss = StateSpace()
        ss.states = {0}
        ss.transitions = []
        ss.top = 0
        ss.bottom = 0
        result = detect_livelock(ss)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: livelock_sccs
# ---------------------------------------------------------------------------

class TestLivelockSCCs:
    """Test the livelock_sccs function."""

    def test_trapped_cycle_returns_one_scc(self):
        ss = _make_livelocked_ss()
        sccs = livelock_sccs(ss)
        assert len(sccs) == 1
        assert sccs[0] == frozenset({1, 2})

    def test_safe_cycle_returns_empty(self):
        ss = _make_livelock_free_ss()
        sccs = livelock_sccs(ss)
        assert len(sccs) == 0

    def test_multiple_trapped(self):
        ss = _make_multiple_sccs_ss()
        sccs = livelock_sccs(ss)
        assert len(sccs) == 2
        scc_sets = {frozenset(s) for s in sccs}
        assert frozenset({1, 2}) in scc_sets
        assert frozenset({4, 5}) in scc_sets

    def test_self_loop_scc(self):
        ss = _make_self_loop_livelock()
        sccs = livelock_sccs(ss)
        assert len(sccs) == 1
        assert frozenset({1}) in [s for s in sccs]

    def test_linear_no_sccs(self):
        ss = _make_simple_linear()
        sccs = livelock_sccs(ss)
        assert len(sccs) == 0


# ---------------------------------------------------------------------------
# Tests: tropical_livelock_score
# ---------------------------------------------------------------------------

class TestTropicalLivelockScore:
    """Test tropical eigenvalue on trapped SCCs."""

    def test_trapped_positive_score(self):
        ss = _make_livelocked_ss()
        score = tropical_livelock_score(ss)
        assert score > 0.0

    def test_safe_zero_score(self):
        ss = _make_livelock_free_ss()
        score = tropical_livelock_score(ss)
        assert score == 0.0

    def test_linear_zero_score(self):
        ss = _make_simple_linear()
        score = tropical_livelock_score(ss)
        assert score == 0.0

    def test_self_loop_positive(self):
        ss = _make_self_loop_livelock()
        score = tropical_livelock_score(ss)
        assert score > 0.0

    def test_unit_weight_eigenvalue(self):
        """For unit-weight cycles, tropical eigenvalue = 1.0."""
        ss = _make_livelocked_ss()
        score = tropical_livelock_score(ss)
        assert abs(score - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: heat_kernel_livelock
# ---------------------------------------------------------------------------

class TestHeatKernelLivelock:
    """Test heat kernel livelock indicators."""

    def test_trapped_states_high_diagonal(self):
        ss = _make_livelocked_ss()
        indicators = heat_kernel_livelock(ss, t=10.0)
        assert len(indicators) > 0
        for ind in indicators:
            assert ind.state in {1, 2}
            assert ind.diagonal_value >= 0.0

    def test_no_livelock_no_indicators(self):
        ss = _make_livelock_free_ss()
        indicators = heat_kernel_livelock(ss, t=10.0)
        assert len(indicators) == 0

    def test_linear_no_indicators(self):
        ss = _make_simple_linear()
        indicators = heat_kernel_livelock(ss, t=5.0)
        assert len(indicators) == 0

    def test_indicator_type(self):
        ss = _make_livelocked_ss()
        indicators = heat_kernel_livelock(ss, t=5.0)
        for ind in indicators:
            assert isinstance(ind, HeatKernelLivelockIndicator)
            assert isinstance(ind.diagonal_value, float)
            assert isinstance(ind.is_trapped, bool)

    def test_small_t_still_detects(self):
        ss = _make_livelocked_ss()
        indicators = heat_kernel_livelock(ss, t=0.1)
        assert len(indicators) > 0


# ---------------------------------------------------------------------------
# Tests: is_livelock_free
# ---------------------------------------------------------------------------

class TestIsLivelockFree:
    """Test the is_livelock_free predicate."""

    def test_trapped_not_free(self):
        ss = _make_livelocked_ss()
        assert not is_livelock_free(ss)

    def test_safe_cycle_free(self):
        ss = _make_livelock_free_ss()
        assert is_livelock_free(ss)

    def test_linear_free(self):
        ss = _make_simple_linear()
        assert is_livelock_free(ss)

    def test_self_loop_not_free(self):
        ss = _make_self_loop_livelock()
        assert not is_livelock_free(ss)


# ---------------------------------------------------------------------------
# Tests: analyze_livelock
# ---------------------------------------------------------------------------

class TestAnalyzeLivelock:
    """Test comprehensive livelock analysis."""

    def test_result_type(self):
        ss = _make_livelocked_ss()
        result = analyze_livelock(ss)
        assert isinstance(result, LivelockResult)

    def test_trapped_analysis(self):
        ss = _make_livelocked_ss()
        result = analyze_livelock(ss)
        assert not result.is_livelock_free
        assert result.num_trapped_sccs == 1
        assert result.total_trapped_states == 2
        assert result.max_tropical_eigenvalue > 0.0
        assert 1 in result.livelocked_states
        assert 2 in result.livelocked_states

    def test_safe_analysis(self):
        ss = _make_livelock_free_ss()
        result = analyze_livelock(ss)
        assert result.is_livelock_free
        assert result.num_trapped_sccs == 0
        assert result.total_trapped_states == 0
        assert result.max_tropical_eigenvalue == 0.0

    def test_summary_message(self):
        ss = _make_livelocked_ss()
        result = analyze_livelock(ss)
        assert "LIVELOCK" in result.summary

    def test_summary_free(self):
        ss = _make_livelock_free_ss()
        result = analyze_livelock(ss)
        assert "free" in result.summary.lower()

    def test_reachable_from_top(self):
        ss = _make_livelocked_ss()
        result = analyze_livelock(ss)
        # States 1 and 2 are reachable from top (0 -> 1 -> 2)
        assert 1 in result.reachable_from_top
        assert 2 in result.reachable_from_top

    def test_heat_indicators_present(self):
        ss = _make_livelocked_ss()
        result = analyze_livelock(ss)
        assert len(result.heat_indicators) > 0

    def test_num_states_and_transitions(self):
        ss = _make_livelocked_ss()
        result = analyze_livelock(ss)
        assert result.num_states == 4
        assert result.num_transitions == 3

    def test_trapped_scc_details(self):
        ss = _make_livelocked_ss()
        result = analyze_livelock(ss)
        assert len(result.trapped_sccs) == 1
        scc = result.trapped_sccs[0]
        assert isinstance(scc, LivelockSCC)
        assert scc.states == frozenset({1, 2})
        assert scc.tropical_eigenvalue > 0.0
        assert "loop_a" in scc.cycle_labels or "loop_b" in scc.cycle_labels

    def test_entry_transitions(self):
        ss = _make_livelocked_ss()
        result = analyze_livelock(ss)
        scc = result.trapped_sccs[0]
        # Entry: (0, "enter", 1)
        assert len(scc.entry_transitions) == 1
        assert scc.entry_transitions[0] == (0, "enter", 1)

    def test_multiple_sccs_analysis(self):
        ss = _make_multiple_sccs_ss()
        result = analyze_livelock(ss)
        assert not result.is_livelock_free
        assert result.num_trapped_sccs == 2
        assert result.total_trapped_states == 4


# ---------------------------------------------------------------------------
# Tests with parsed session types
# ---------------------------------------------------------------------------

class TestParsedTypes:
    """Test livelock detection on parsed session types."""

    def test_simple_branch_no_livelock(self):
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        assert is_livelock_free(ss)

    def test_recursive_with_exit(self):
        ast = parse("rec X . &{a: X, b: end}")
        ss = build_statespace(ast)
        assert is_livelock_free(ss)

    def test_recursive_no_exit_livelock(self):
        """rec X . &{a: X} — pure loop with no exit."""
        ast = parse("rec X . &{a: X}")
        ss = build_statespace(ast)
        # This recursion has only one branch that loops back
        # It forms a cycle: initial state -> via 'a' -> back to initial
        # The bottom is the "end" state but it's unreachable
        livelocked = detect_livelock(ss)
        # The top state forms a self-loop SCC
        assert len(livelocked) > 0 or ss.top == ss.bottom

    def test_selection_with_exit(self):
        ast = parse("+{ok: end, err: end}")
        ss = build_statespace(ast)
        assert is_livelock_free(ss)

    def test_nested_rec_safe(self):
        ast = parse("rec X . &{a: &{b: X, c: end}}")
        ss = build_statespace(ast)
        assert is_livelock_free(ss)

    def test_parallel_branches(self):
        ast = parse("(&{a: end} || &{b: end})")
        ss = build_statespace(ast)
        assert is_livelock_free(ss)

    def test_end_type(self):
        ast = parse("end")
        ss = build_statespace(ast)
        assert is_livelock_free(ss)

    def test_score_on_parsed_safe(self):
        ast = parse("rec X . &{a: X, b: end}")
        ss = build_statespace(ast)
        assert tropical_livelock_score(ss) == 0.0

    def test_analyze_on_parsed(self):
        ast = parse("&{a: &{b: end}, c: end}")
        ss = build_statespace(ast)
        result = analyze_livelock(ss)
        assert result.is_livelock_free
        assert result.num_trapped_sccs == 0
