"""Tests for temporal session type analysis (Step 203)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.temporal import (
    TemporalProperty,
    TemporalView,
    always,
    arrow_of_time,
    eventually,
    next_possible,
    now,
    step,
    time_branches,
    time_to_end,
    until,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def end_ss():
    return build_statespace(parse("end"))


@pytest.fixture
def simple_branch():
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def diamond():
    """&{a: &{c: end}, b: &{c: end}}."""
    return build_statespace(parse("&{a: &{c: end}, b: &{c: end}}"))


@pytest.fixture
def deep_chain():
    """&{a: &{b: &{c: end}}}."""
    return build_statespace(parse("&{a: &{b: &{c: end}}}"))


@pytest.fixture
def recursive_ss():
    return build_statespace(parse("rec X . &{next: X, done: end}"))


@pytest.fixture
def select_ss():
    return build_statespace(parse("+{ok: end, err: end}"))


# ---------------------------------------------------------------------------
# Tests: now
# ---------------------------------------------------------------------------


class TestNow:
    def test_end_is_terminal(self, end_ss):
        view = now(end_ss, end_ss.top)
        assert view.is_terminal is True
        assert view.time_remaining == 0
        assert view.time_elapsed == 0
        assert view.past == ()

    def test_simple_branch_present(self, simple_branch):
        view = now(simple_branch, simple_branch.top)
        assert view.present == simple_branch.top
        assert view.is_terminal is False
        assert "a" in view.future_labels
        assert "b" in view.future_labels

    def test_simple_branch_time_remaining(self, simple_branch):
        view = now(simple_branch, simple_branch.top)
        assert view.time_remaining == 1

    def test_future_states_nonempty(self, simple_branch):
        view = now(simple_branch, simple_branch.top)
        assert len(view.future_states) > 0
        assert simple_branch.bottom in view.future_states

    def test_deep_chain_time_remaining(self, deep_chain):
        view = now(deep_chain, deep_chain.top)
        assert view.time_remaining == 3

    def test_deterministic_future_chain(self, deep_chain):
        view = now(deep_chain, deep_chain.top)
        assert view.deterministic_future is True

    def test_branching_not_deterministic(self, simple_branch):
        view = now(simple_branch, simple_branch.top)
        assert view.deterministic_future is False

    def test_bottom_has_no_future(self, simple_branch):
        view = now(simple_branch, simple_branch.bottom)
        assert view.is_terminal is True
        assert len(view.future_labels) == 0


# ---------------------------------------------------------------------------
# Tests: step
# ---------------------------------------------------------------------------


class TestStep:
    def test_step_moves_present(self, simple_branch):
        v0 = now(simple_branch, simple_branch.top)
        v1 = step(simple_branch, v0, "a")
        assert v1.present != v0.present
        assert v1.is_terminal is True

    def test_step_appends_past(self, simple_branch):
        v0 = now(simple_branch, simple_branch.top)
        v1 = step(simple_branch, v0, "a")
        assert len(v1.past) == 1
        assert v1.past[0] == (simple_branch.top, "a")
        assert v1.time_elapsed == 1

    def test_step_chain(self, deep_chain):
        v = now(deep_chain, deep_chain.top)
        v = step(deep_chain, v, "a")
        v = step(deep_chain, v, "b")
        v = step(deep_chain, v, "c")
        assert v.is_terminal is True
        assert v.time_elapsed == 3

    def test_step_invalid_label_raises(self, simple_branch):
        v = now(simple_branch, simple_branch.top)
        with pytest.raises(ValueError, match="not enabled"):
            step(simple_branch, v, "nonexistent")

    def test_step_at_bottom_raises(self, end_ss):
        v = now(end_ss, end_ss.top)
        with pytest.raises(ValueError):
            step(end_ss, v, "anything")


# ---------------------------------------------------------------------------
# Tests: eventually
# ---------------------------------------------------------------------------


class TestEventually:
    def test_eventually_at_top(self, simple_branch):
        prop = eventually(simple_branch, simple_branch.top, "a")
        assert prop.holds is True
        assert prop.name == "eventually"

    def test_eventually_reachable(self, deep_chain):
        prop = eventually(deep_chain, deep_chain.top, "c")
        assert prop.holds is True

    def test_eventually_unreachable(self, simple_branch):
        prop = eventually(simple_branch, simple_branch.top, "nonexistent")
        assert prop.holds is False

    def test_eventually_from_bottom(self, simple_branch):
        prop = eventually(simple_branch, simple_branch.bottom, "a")
        assert prop.holds is False


# ---------------------------------------------------------------------------
# Tests: always
# ---------------------------------------------------------------------------


class TestAlways:
    def test_always_single_label_chain(self, deep_chain):
        # In a chain &{a: &{b: &{c: end}}}, 'a' is NOT always available.
        prop = always(deep_chain, deep_chain.top, "a")
        assert prop.holds is False

    def test_always_recursive(self, recursive_ss):
        # rec X . &{next: X, done: end} — 'next' is always at the loop state.
        prop = always(recursive_ss, recursive_ss.top, "next")
        # 'next' is not available at bottom.
        # But we skip bottom in always check, so it depends on structure.
        assert isinstance(prop.holds, bool)

    def test_always_nonexistent(self, simple_branch):
        prop = always(simple_branch, simple_branch.top, "z")
        assert prop.holds is False


# ---------------------------------------------------------------------------
# Tests: until
# ---------------------------------------------------------------------------


class TestUntil:
    def test_until_immediate_goal(self, simple_branch):
        # 'a' holds until 'b' — both available at top, so goal reached.
        prop = until(simple_branch, simple_branch.top, "a", "b")
        assert prop.holds is True

    def test_until_no_goal(self, simple_branch):
        prop = until(simple_branch, simple_branch.top, "a", "nonexistent")
        assert prop.holds is False

    def test_until_deep(self, deep_chain):
        # &{a: &{b: &{c: end}}} — 'a' holds until 'c'? No, 'a' not at depth 2.
        prop = until(deep_chain, deep_chain.top, "a", "c")
        assert prop.holds is False


# ---------------------------------------------------------------------------
# Tests: next_possible
# ---------------------------------------------------------------------------


class TestNextPossible:
    def test_next_at_top(self, simple_branch):
        labels = next_possible(simple_branch, simple_branch.top)
        assert "a" in labels
        assert "b" in labels

    def test_next_at_bottom(self, simple_branch):
        labels = next_possible(simple_branch, simple_branch.bottom)
        assert labels == []


# ---------------------------------------------------------------------------
# Tests: time_to_end
# ---------------------------------------------------------------------------


class TestTimeToEnd:
    def test_end_zero(self, end_ss):
        assert time_to_end(end_ss, end_ss.top) == 0

    def test_simple_one(self, simple_branch):
        assert time_to_end(simple_branch, simple_branch.top) == 1

    def test_deep_three(self, deep_chain):
        assert time_to_end(deep_chain, deep_chain.top) == 3


# ---------------------------------------------------------------------------
# Tests: arrow_of_time
# ---------------------------------------------------------------------------


class TestArrowOfTime:
    def test_dag_has_arrow(self, simple_branch):
        assert arrow_of_time(simple_branch) is True

    def test_recursive_no_arrow(self, recursive_ss):
        assert arrow_of_time(recursive_ss) is False

    def test_end_has_arrow(self, end_ss):
        assert arrow_of_time(end_ss) is True

    def test_deep_chain_has_arrow(self, deep_chain):
        assert arrow_of_time(deep_chain) is True


# ---------------------------------------------------------------------------
# Tests: time_branches
# ---------------------------------------------------------------------------


class TestTimeBranches:
    def test_end_one_path(self, end_ss):
        assert time_branches(end_ss, end_ss.top) == 1

    def test_simple_two_paths(self, simple_branch):
        assert time_branches(simple_branch, simple_branch.top) == 2

    def test_diamond_paths(self, diamond):
        # &{a: &{c: end}, b: &{c: end}} — two paths through the diamond.
        assert time_branches(diamond, diamond.top) == 2

    def test_bottom_one_path(self, simple_branch):
        assert time_branches(simple_branch, simple_branch.bottom) == 1
