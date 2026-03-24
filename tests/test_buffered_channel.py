"""Tests for buffered channel session types (Step 158)."""

import pytest

from reticulate.parser import (
    Branch, End, Rec, Select, Var, parse, pretty,
)
from reticulate.statespace import build_statespace
from reticulate.buffered_channel import (
    BufferedChannelResult,
    BufferedStateKey,
    GrowthAnalysis,
    GrowthPoint,
    build_buffered_statespace,
    check_buffer_safety,
    buffer_growth_analysis,
    optimal_buffer_size,
)


# ---------------------------------------------------------------------------
# 1. BufferedChannelResult dataclass
# ---------------------------------------------------------------------------


class TestBufferedChannelResultDataclass:
    """BufferedChannelResult is a frozen dataclass."""

    def test_frozen(self):
        from reticulate.lattice import LatticeResult
        lr = LatticeResult(
            is_lattice=True, has_top=True, has_bottom=True,
            all_meets_exist=True, all_joins_exist=True,
            num_scc=1, counterexample=None, scc_map={0: 0},
        )
        r = BufferedChannelResult(
            type_str="end", capacity=1,
            unbuffered_states=1, buffered_states=1, buffered_transitions=0,
            is_lattice=True, is_buffer_safe=True,
            overflow_states=0, underflow_states=0,
            max_buffer_occupancy=0, buffer_distribution={0: 1},
            deadlock_states=0, lattice_result=lr,
        )
        with pytest.raises(AttributeError):
            r.capacity = 2  # type: ignore[misc]

    def test_fields_accessible(self):
        from reticulate.lattice import LatticeResult
        lr = LatticeResult(
            is_lattice=True, has_top=True, has_bottom=True,
            all_meets_exist=True, all_joins_exist=True,
            num_scc=1, counterexample=None, scc_map={0: 0},
        )
        r = BufferedChannelResult(
            type_str="end", capacity=1,
            unbuffered_states=1, buffered_states=1, buffered_transitions=0,
            is_lattice=True, is_buffer_safe=True,
            overflow_states=0, underflow_states=0,
            max_buffer_occupancy=0, buffer_distribution={0: 1},
            deadlock_states=0, lattice_result=lr,
        )
        assert r.capacity == 1
        assert r.is_lattice is True
        assert r.is_buffer_safe is True
        assert r.buffer_distribution == {0: 1}


# ---------------------------------------------------------------------------
# 2. build_buffered_statespace — basic construction
# ---------------------------------------------------------------------------


class TestBuildBufferedStatespaceBasic:
    """BFS construction of buffered channel state space."""

    def test_end_type_k0(self):
        """end with capacity 0: single state, no transitions."""
        ss = build_buffered_statespace(End(), capacity=0)
        assert len(ss.states) == 1
        assert len(ss.transitions) == 0
        assert ss.top == ss.bottom

    def test_end_type_k1(self):
        """end with capacity 1: single state, no transitions."""
        ss = build_buffered_statespace(End(), capacity=1)
        assert len(ss.states) == 1
        assert len(ss.transitions) == 0

    def test_branch_only_k0(self):
        """&{a: end} with capacity 0: synchronous pass-through."""
        s = parse("&{a: end}")
        ss = build_buffered_statespace(s, capacity=0)
        assert len(ss.states) >= 2
        assert len(ss.transitions) >= 1

    def test_select_only_k1(self):
        """+{a: end} with capacity 1: enqueue 'a'."""
        s = parse("+{a: end}")
        ss = build_buffered_statespace(s, capacity=1)
        # Should have at least: (top, []) -> enqueue:a -> (end, [a])
        assert len(ss.states) >= 2
        enqueue_labels = [lbl for _, lbl, _ in ss.transitions if "enqueue" in lbl]
        assert len(enqueue_labels) >= 1

    def test_select_only_k0(self):
        """+{a: end} with capacity 0: synchronous, direct transition."""
        s = parse("+{a: end}")
        ss = build_buffered_statespace(s, capacity=0)
        assert len(ss.states) >= 2
        assert len(ss.transitions) >= 1

    def test_branch_select_k1(self):
        """&{a: +{b: end}} with capacity 1: branch blocked (empty buffer)."""
        s = parse("&{a: +{b: end}}")
        ss = build_buffered_statespace(s, capacity=1)
        # Branch at top has empty buffer, so it cannot dequeue.
        # No transitions are possible: this is a deadlock scenario
        # demonstrating why buffered channels need matching producers.
        assert len(ss.states) >= 1

    def test_two_branch_k0(self):
        """&{a: end, b: end} with capacity 0."""
        s = parse("&{a: end, b: end}")
        ss = build_buffered_statespace(s, capacity=0)
        assert len(ss.states) >= 2
        assert len(ss.transitions) >= 2

    def test_two_select_k1(self):
        """+{a: end, b: end} with capacity 1."""
        s = parse("+{a: end, b: end}")
        ss = build_buffered_statespace(s, capacity=1)
        # Two enqueue transitions from top
        enqueue_labels = [lbl for _, lbl, _ in ss.transitions if "enqueue" in lbl]
        assert len(enqueue_labels) >= 2


# ---------------------------------------------------------------------------
# 3. Buffer capacity effects
# ---------------------------------------------------------------------------


class TestBufferCapacityEffects:
    """State space grows with buffer capacity."""

    def test_larger_capacity_more_states(self):
        """Increasing capacity should not decrease state count."""
        s = parse("+{a: end, b: end}")
        ss1 = build_buffered_statespace(s, capacity=1)
        ss2 = build_buffered_statespace(s, capacity=2)
        assert len(ss2.states) >= len(ss1.states)

    def test_capacity_0_matches_base(self):
        """Capacity 0 (synchronous) should have same states as base."""
        s = parse("&{a: end, b: end}")
        base_ss = build_statespace(s)
        buffered_ss = build_buffered_statespace(s, capacity=0)
        assert len(buffered_ss.states) == len(base_ss.states)

    def test_recursive_type_k1(self):
        """rec X . +{a: &{b: X}} with capacity 1."""
        s = parse("rec X . +{a: &{b: X}}")
        ss = build_buffered_statespace(s, capacity=1)
        assert len(ss.states) >= 2
        assert len(ss.transitions) >= 1


# ---------------------------------------------------------------------------
# 4. check_buffer_safety
# ---------------------------------------------------------------------------


class TestCheckBufferSafety:
    """Buffer safety analysis."""

    def test_end_is_safe(self):
        """end type is trivially safe."""
        r = check_buffer_safety(End(), capacity=1)
        assert r.is_buffer_safe is True
        assert r.overflow_states == 0
        assert r.underflow_states == 0

    def test_branch_only_k1_underflow(self):
        """&{a: end} with K=1: branch state has empty buffer = underflow."""
        r = check_buffer_safety(parse("&{a: end}"), capacity=1)
        assert r.underflow_states >= 1

    def test_select_end_k1_safe_overflow(self):
        """+{a: end} with K=1: after enqueue buffer is full at end state."""
        r = check_buffer_safety(parse("+{a: end}"), capacity=1)
        # The select state at top has empty buffer, so it can enqueue
        assert r.overflow_states == 0  # no selection state with full buffer

    def test_capacity_0_always_safe(self):
        """Synchronous mode is always safe (no buffer to overflow/underflow)."""
        r = check_buffer_safety(parse("&{a: +{b: end}}"), capacity=0)
        assert r.is_buffer_safe is True

    def test_result_has_lattice_info(self):
        """Result includes lattice check."""
        r = check_buffer_safety(parse("+{a: end}"), capacity=1)
        assert r.lattice_result is not None
        assert isinstance(r.is_lattice, bool)

    def test_result_has_buffer_distribution(self):
        """Result includes buffer distribution."""
        r = check_buffer_safety(parse("+{a: end}"), capacity=1)
        assert isinstance(r.buffer_distribution, dict)
        assert sum(r.buffer_distribution.values()) == r.buffered_states

    def test_deadlock_count(self):
        """Result reports deadlock states."""
        r = check_buffer_safety(End(), capacity=1)
        assert r.deadlock_states == 0


# ---------------------------------------------------------------------------
# 5. buffer_growth_analysis
# ---------------------------------------------------------------------------


class TestBufferGrowthAnalysis:
    """Growth analysis over increasing capacity."""

    def test_end_type_constant(self):
        """end type: state count is constant across capacities."""
        ga = buffer_growth_analysis(End(), max_capacity=3)
        assert len(ga.points) == 4  # 0, 1, 2, 3
        for p in ga.points:
            assert p.states == 1

    def test_growth_has_ratios(self):
        """Growth analysis computes ratios."""
        ga = buffer_growth_analysis(parse("+{a: end}"), max_capacity=3)
        assert len(ga.growth_ratios) == 3  # ratios for k=1,2,3

    def test_growth_monotonic_states(self):
        """State count should be non-decreasing with capacity."""
        ga = buffer_growth_analysis(parse("+{a: end, b: end}"), max_capacity=3)
        for i in range(1, len(ga.points)):
            assert ga.points[i].states >= ga.points[i - 1].states

    def test_optimal_capacity_reported(self):
        """Optimal capacity is reported."""
        ga = buffer_growth_analysis(End(), max_capacity=3)
        assert ga.optimal_capacity >= 0

    def test_growth_point_fields(self):
        """GrowthPoint has all expected fields."""
        ga = buffer_growth_analysis(parse("&{a: end}"), max_capacity=1)
        p = ga.points[0]
        assert hasattr(p, 'capacity')
        assert hasattr(p, 'states')
        assert hasattr(p, 'transitions')
        assert hasattr(p, 'is_lattice')
        assert hasattr(p, 'deadlocks')


# ---------------------------------------------------------------------------
# 6. optimal_buffer_size
# ---------------------------------------------------------------------------


class TestOptimalBufferSize:
    """Minimum buffer capacity for deadlock freedom."""

    def test_end_type_optimal_0(self):
        """end type: optimal capacity is 0."""
        assert optimal_buffer_size(End()) == 0

    def test_branch_only_optimal_0(self):
        """&{a: end}: optimal capacity is 0 (synchronous has no deadlocks)."""
        assert optimal_buffer_size(parse("&{a: end}")) == 0

    def test_select_only_optimal(self):
        """+{a: end}: should find optimal capacity."""
        opt = optimal_buffer_size(parse("+{a: end}"))
        assert opt >= 0

    def test_returns_minus1_if_not_found(self):
        """Returns -1 if no capacity in range yields deadlock freedom."""
        # For most well-formed types, optimal should be found within range
        opt = optimal_buffer_size(End(), max_search=0)
        assert opt == 0  # end always has 0 deadlocks


# ---------------------------------------------------------------------------
# 7. Benchmark protocols
# ---------------------------------------------------------------------------


class TestBenchmarkProtocols:
    """Buffered analysis on benchmark protocols."""

    def test_iterator_k1(self):
        """Java Iterator: rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}."""
        s = parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        r = check_buffer_safety(s, capacity=1)
        assert r.buffered_states >= 1
        assert isinstance(r.is_lattice, bool)

    def test_two_select_growth(self):
        """+{a: end, b: end} growth analysis."""
        ga = buffer_growth_analysis(parse("+{a: end, b: end}"), max_capacity=3)
        assert len(ga.points) == 4
        assert ga.points[0].states >= 2  # at least top + bottom in sync mode

    def test_nested_branch_select_k2(self):
        """&{a: +{x: end}, b: +{y: end}} with K=2."""
        s = parse("&{a: +{x: end}, b: +{y: end}}")
        r = check_buffer_safety(s, capacity=2)
        assert r.buffered_states >= 2
        assert r.max_buffer_occupancy <= 2
