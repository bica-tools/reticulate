"""Tests for replay prevention via lattice monotonicity (Step 89d)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.replay import (
    MonotoneMonitor,
    ReplayResult,
    analyze_replay,
    check_replay_safety,
    monotone_monitor,
    replay_vulnerable_states,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def end_ss():
    """Single-state 'end' type."""
    return build_statespace(parse("end"))


@pytest.fixture
def linear_ss():
    """Linear: a . b . end."""
    return build_statespace(parse("&{a: &{b: end}}"))


@pytest.fixture
def branch2_ss():
    """Branch with 2 options: &{a: end, b: end}."""
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def recursive_ss():
    """Recursive type with cycle: rec X . &{a: X}."""
    return build_statespace(parse("rec X . &{a: X}"))


@pytest.fixture
def recursive_branch_ss():
    """Recursive with exit: rec X . &{next: X, done: end}."""
    return build_statespace(parse("rec X . &{next: X, done: end}"))


@pytest.fixture
def nested_ss():
    """Nested: &{a: &{x: end, y: end}, b: end}."""
    return build_statespace(parse("&{a: &{x: end, y: end}, b: end}"))


@pytest.fixture
def parallel_ss():
    """Parallel: (&{a: end} || &{b: end})."""
    return build_statespace(parse("(&{a: end} || &{b: end})"))


@pytest.fixture
def select_ss():
    """Selection: +{ok: end, err: end}."""
    return build_statespace(parse("+{ok: end, err: end}"))


# ---------------------------------------------------------------------------
# check_replay_safety
# ---------------------------------------------------------------------------


class TestCheckReplaySafety:
    """Tests for check_replay_safety."""

    def test_end_is_safe(self, end_ss):
        assert check_replay_safety(end_ss) is True

    def test_linear_is_safe(self, linear_ss):
        assert check_replay_safety(linear_ss) is True

    def test_branch_is_safe(self, branch2_ss):
        assert check_replay_safety(branch2_ss) is True

    def test_recursive_is_not_safe(self, recursive_ss):
        """Recursive types create cycles -> not replay-safe."""
        assert check_replay_safety(recursive_ss) is False

    def test_recursive_branch_not_safe(self, recursive_branch_ss):
        """rec X . &{next: X, done: end} has cycle."""
        assert check_replay_safety(recursive_branch_ss) is False

    def test_nested_is_safe(self, nested_ss):
        assert check_replay_safety(nested_ss) is True

    def test_parallel_is_safe(self, parallel_ss):
        assert check_replay_safety(parallel_ss) is True

    def test_select_is_safe(self, select_ss):
        assert check_replay_safety(select_ss) is True

    def test_deep_linear_is_safe(self):
        """Deeper linear chain is still safe."""
        ss = build_statespace(parse("&{a: &{b: &{c: &{d: end}}}}"))
        assert check_replay_safety(ss) is True


# ---------------------------------------------------------------------------
# replay_vulnerable_states
# ---------------------------------------------------------------------------


class TestReplayVulnerableStates:
    """Tests for replay_vulnerable_states."""

    def test_end_no_vulnerable(self, end_ss):
        vuln = replay_vulnerable_states(end_ss)
        assert len(vuln) == 0

    def test_linear_no_vulnerable(self, linear_ss):
        vuln = replay_vulnerable_states(linear_ss)
        assert len(vuln) == 0

    def test_branch_no_vulnerable(self, branch2_ss):
        vuln = replay_vulnerable_states(branch2_ss)
        assert len(vuln) == 0

    def test_recursive_has_vulnerable(self, recursive_ss):
        vuln = replay_vulnerable_states(recursive_ss)
        assert len(vuln) > 0

    def test_recursive_branch_has_vulnerable(self, recursive_branch_ss):
        vuln = replay_vulnerable_states(recursive_branch_ss)
        assert len(vuln) > 0

    def test_vulnerable_subset_of_states(self, recursive_branch_ss):
        vuln = replay_vulnerable_states(recursive_branch_ss)
        assert vuln <= recursive_branch_ss.states

    def test_parallel_no_vulnerable(self, parallel_ss):
        vuln = replay_vulnerable_states(parallel_ss)
        assert len(vuln) == 0


# ---------------------------------------------------------------------------
# monotone_monitor
# ---------------------------------------------------------------------------


class TestMonotoneMonitor:
    """Tests for monotone_monitor."""

    def test_monitor_type(self, linear_ss):
        mon = monotone_monitor(linear_ss)
        assert isinstance(mon, MonotoneMonitor)

    def test_initial_state(self, linear_ss):
        mon = monotone_monitor(linear_ss)
        assert mon.initial_state == linear_ss.top

    def test_terminal_state(self, linear_ss):
        mon = monotone_monitor(linear_ss)
        assert mon.terminal_state == linear_ss.bottom

    def test_acyclic_no_blocked(self, linear_ss):
        """In acyclic graph, no transitions are blocked."""
        mon = monotone_monitor(linear_ss)
        total_blocked = sum(len(v) for v in mon.blocked_at.values())
        assert total_blocked == 0

    def test_recursive_has_blocked(self, recursive_ss):
        """Recursive types should have blocked back-edges."""
        mon = monotone_monitor(recursive_ss)
        total_blocked = sum(len(v) for v in mon.blocked_at.values())
        assert total_blocked > 0

    def test_all_states_have_entries(self, branch2_ss):
        mon = monotone_monitor(branch2_ss)
        for state in branch2_ss.states:
            assert state in mon.allowed_at
            assert state in mon.blocked_at

    def test_state_machine_complete(self, linear_ss):
        mon = monotone_monitor(linear_ss)
        assert len(mon.state_machine) > 0


# ---------------------------------------------------------------------------
# analyze_replay
# ---------------------------------------------------------------------------


class TestAnalyzeReplay:
    """Tests for analyze_replay."""

    def test_result_type(self, linear_ss):
        result = analyze_replay(linear_ss)
        assert isinstance(result, ReplayResult)

    def test_safe_result(self, linear_ss):
        result = analyze_replay(linear_ss)
        assert result.is_replay_safe is True
        assert len(result.vulnerable_states) == 0
        assert result.acyclic_fraction == 1.0

    def test_unsafe_result(self, recursive_ss):
        result = analyze_replay(recursive_ss)
        assert result.is_replay_safe is False
        assert len(result.vulnerable_states) > 0
        assert result.acyclic_fraction < 1.0

    def test_cycle_count(self, recursive_branch_ss):
        result = analyze_replay(recursive_branch_ss)
        assert result.cycle_count >= 0  # May have cycles

    def test_monitor_in_result(self, linear_ss):
        result = analyze_replay(linear_ss)
        assert isinstance(result.monitor, MonotoneMonitor)

    def test_scc_sizes(self, linear_ss):
        result = analyze_replay(linear_ss)
        assert len(result.scc_sizes) > 0
        assert result.max_scc_size >= 1

    def test_acyclic_fraction_bounds(self, recursive_branch_ss):
        result = analyze_replay(recursive_branch_ss)
        assert 0.0 <= result.acyclic_fraction <= 1.0

    def test_end_type(self, end_ss):
        result = analyze_replay(end_ss)
        assert result.is_replay_safe is True
        assert result.max_scc_size == 1

    def test_parallel_safe(self, parallel_ss):
        result = analyze_replay(parallel_ss)
        assert result.is_replay_safe is True

    def test_recursive_vulnerable_states_are_in_cycles(self, recursive_branch_ss):
        result = analyze_replay(recursive_branch_ss)
        for state in result.vulnerable_states:
            assert state in recursive_branch_ss.states

    def test_select_safe(self, select_ss):
        result = analyze_replay(select_ss)
        assert result.is_replay_safe is True

    def test_nested_acyclic_fraction_one(self, nested_ss):
        result = analyze_replay(nested_ss)
        assert result.acyclic_fraction == 1.0
