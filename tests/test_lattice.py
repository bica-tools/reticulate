"""Tests for lattice property checking (lattice.py)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import LatticeResult, check_lattice, compute_meet, compute_join


# ===================================================================
# Helpers
# ===================================================================

def _check(source: str) -> LatticeResult:
    """Parse, build state space, check lattice."""
    return check_lattice(build_statespace(parse(source)))


def _ss(source: str) -> StateSpace:
    """Parse and build state space."""
    return build_statespace(parse(source))


# ===================================================================
# Basic lattice checks
# ===================================================================

class TestSingleState:
    """``end`` — trivial one-element lattice."""

    def test_is_lattice(self) -> None:
        r = _check("end")
        assert r.is_lattice is True

    def test_one_scc(self) -> None:
        r = _check("end")
        assert r.num_scc == 1

    def test_has_top_and_bottom(self) -> None:
        r = _check("end")
        assert r.has_top is True
        assert r.has_bottom is True

    def test_no_counterexample(self) -> None:
        r = _check("end")
        assert r.counterexample is None


class TestChain:
    """``a . b . end`` — 3-element chain (total order = lattice)."""

    def test_is_lattice(self) -> None:
        r = _check("a . b . end")
        assert r.is_lattice is True

    def test_three_sccs(self) -> None:
        r = _check("a . b . end")
        assert r.num_scc == 3

    def test_all_properties(self) -> None:
        r = _check("a . b . end")
        assert r.has_top is True
        assert r.has_bottom is True
        assert r.all_meets_exist is True
        assert r.all_joins_exist is True


class TestBranchToEnd:
    """``&{m: end, n: end}`` — two branches both going to end."""

    def test_is_lattice(self) -> None:
        r = _check("&{m: end, n: end}")
        assert r.is_lattice is True

    def test_two_sccs(self) -> None:
        r = _check("&{m: end, n: end}")
        assert r.num_scc == 2


class TestBranchDiamond:
    """``&{m: a . end, n: b . end}`` — diamond: top -> {after_m, after_n} -> end."""

    def test_is_lattice(self) -> None:
        r = _check("&{m: a . end, n: b . end}")
        assert r.is_lattice is True

    def test_four_sccs(self) -> None:
        r = _check("&{m: a . end, n: b . end}")
        assert r.num_scc == 4

    def test_meet_of_branches_is_bottom(self) -> None:
        """The meet of the two intermediate states should be bottom."""
        ss = _ss("&{m: a . end, n: b . end}")
        # Find the two intermediate states (not top, not bottom)
        intermediates = [s for s in ss.states if s != ss.top and s != ss.bottom]
        assert len(intermediates) == 2
        meet = compute_meet(ss, intermediates[0], intermediates[1])
        assert meet == ss.bottom

    def test_join_of_branches_is_top(self) -> None:
        """The join of the two intermediate states should be top."""
        ss = _ss("&{m: a . end, n: b . end}")
        intermediates = [s for s in ss.states if s != ss.top and s != ss.bottom]
        assert len(intermediates) == 2
        join = compute_join(ss, intermediates[0], intermediates[1])
        assert join == ss.top


# ===================================================================
# Recursion and cycles
# ===================================================================

class TestSelfLoopRecursion:
    """``rec X . &{next: X, done: end}`` — self-loop collapses to 1 SCC."""

    def test_is_lattice(self) -> None:
        r = _check("rec X . &{next: X, done: end}")
        assert r.is_lattice is True

    def test_two_sccs(self) -> None:
        """Top loops to itself → 1 SCC for top, 1 for end = 2 SCCs."""
        r = _check("rec X . &{next: X, done: end}")
        assert r.num_scc == 2


class TestCycleRecursion:
    """``rec X . &{a: &{b: X}, done: end}`` — cycle between top and after_a."""

    def test_is_lattice(self) -> None:
        r = _check("rec X . &{a: &{b: X}, done: end}")
        assert r.is_lattice is True

    def test_two_sccs(self) -> None:
        """top and after_a form a cycle → 1 SCC; plus end = 2 SCCs."""
        r = _check("rec X . &{a: &{b: X}, done: end}")
        assert r.num_scc == 2


class TestNestedRec:
    """``rec X . &{a: rec Y . &{b: Y, done: X}, done: end}``"""

    def test_is_lattice(self) -> None:
        r = _check("rec X . &{a: rec Y . &{b: Y, done: X}, done: end}")
        assert r.is_lattice is True

    def test_scc_count(self) -> None:
        """top and inner form a cycle → 1 SCC; plus end = 2 SCCs."""
        r = _check("rec X . &{a: rec Y . &{b: Y, done: X}, done: end}")
        assert r.num_scc == 2


# ===================================================================
# Product (parallel) lattices
# ===================================================================

class TestProduct2x2:
    """``(a . end || b . end)`` — 2×2 product lattice (diamond)."""

    def setup_method(self) -> None:
        self.ss = _ss("(a . end || b . end)")
        self.result = check_lattice(self.ss)

    def test_is_lattice(self) -> None:
        assert self.result.is_lattice is True

    def test_four_sccs(self) -> None:
        assert self.result.num_scc == 4

    def test_meet_of_intermediates(self) -> None:
        """In the 2×2 product, the two intermediate states have meet = bottom."""
        intermediates = sorted(
            s for s in self.ss.states if s != self.ss.top and s != self.ss.bottom
        )
        assert len(intermediates) == 2
        meet = compute_meet(self.ss, intermediates[0], intermediates[1])
        assert meet == self.ss.bottom

    def test_join_of_intermediates(self) -> None:
        """In the 2×2 product, the two intermediate states have join = top."""
        intermediates = sorted(
            s for s in self.ss.states if s != self.ss.top and s != self.ss.bottom
        )
        assert len(intermediates) == 2
        join = compute_join(self.ss, intermediates[0], intermediates[1])
        assert join == self.ss.top

    def test_meet_with_self(self) -> None:
        """Meet of any state with itself is that state."""
        for s in self.ss.states:
            assert compute_meet(self.ss, s, s) == s

    def test_join_with_self(self) -> None:
        """Join of any state with itself is that state."""
        for s in self.ss.states:
            assert compute_join(self.ss, s, s) == s


class TestProduct3x3:
    """``(a . b . end || c . d . end)`` — 3×3 = 9-state product lattice."""

    def setup_method(self) -> None:
        self.result = _check("(a . b . end || c . d . end)")

    def test_is_lattice(self) -> None:
        assert self.result.is_lattice is True

    def test_nine_sccs(self) -> None:
        assert self.result.num_scc == 9


class TestRecursiveProduct:
    """``(rec X . &{a: X, done: end} || rec Y . &{c: Y, stop: end})`` — 2×2."""

    def setup_method(self) -> None:
        src = "(rec X . &{a: X, done: end} || rec Y . &{c: Y, stop: end})"
        self.result = _check(src)

    def test_is_lattice(self) -> None:
        assert self.result.is_lattice is True

    def test_four_sccs(self) -> None:
        assert self.result.num_scc == 4


# ===================================================================
# Non-lattice (manually constructed state spaces)
# ===================================================================

class TestNonLatticeNoMeet:
    """Manually construct a state space where two states have no meet.

    Shape: top → a, top → b, a → c, a → d, b → c, b → d
    (no bottom connecting c and d — they are incomparable lower bounds)

    Actually, for no meet: we need two lower bounds of (a, b) with
    neither being greater. Let's build:
        top → a → c → end
        top → b → c → end
        top → a → d → end
        top → b → d → end
    Here c and d are both lower bounds of (a, b), but neither c ≥ d
    nor d ≥ c, so no meet for (a, b) in the sub-poset above end.
    But they both reach end, so meet(c, d) = end.
    And meet(a, b) candidates are {c, d, end} — c doesn't reach d
    and d doesn't reach c, so no GLB. No meet!
    """

    def setup_method(self) -> None:
        # States: 0=top, 1=a, 2=b, 3=c, 4=d, 5=end
        self.ss = StateSpace(
            states={0, 1, 2, 3, 4, 5},
            transitions=[
                (0, "go_a", 1),
                (0, "go_b", 2),
                (1, "to_c", 3),
                (1, "to_d", 4),
                (2, "to_c", 3),
                (2, "to_d", 4),
                (3, "finish", 5),
                (4, "finish", 5),
            ],
            top=0,
            bottom=5,
            labels={0: "top", 1: "a", 2: "b", 3: "c", 4: "d", 5: "end"},
        )

    def test_not_lattice(self) -> None:
        r = check_lattice(self.ss)
        assert r.is_lattice is False

    def test_counterexample_reports_failure(self) -> None:
        r = check_lattice(self.ss)
        assert r.counterexample is not None
        a, b, kind = r.counterexample
        # This graph fails both meets and joins; algorithm may report either first.
        assert kind in ("no_meet", "no_join")

    def test_all_meets_false(self) -> None:
        r = check_lattice(self.ss)
        assert r.all_meets_exist is False

    def test_has_top_and_bottom(self) -> None:
        """The poset still has top and bottom, just not all meets."""
        r = check_lattice(self.ss)
        assert r.has_top is True
        assert r.has_bottom is True


class TestNonLatticeNoJoin:
    """Manually construct a state space where two states have no join.

    Shape (dual of no-meet):
        top → c → a → end
        top → c → b → end
        top → d → a → end
        top → d → b → end
    Here c and d are both upper bounds of (a, b), but neither c ≥ d
    nor d ≥ c. So no join for (a, b).
    """

    def setup_method(self) -> None:
        # States: 0=top, 1=c, 2=d, 3=a, 4=b, 5=end
        self.ss = StateSpace(
            states={0, 1, 2, 3, 4, 5},
            transitions=[
                (0, "go_c", 1),
                (0, "go_d", 2),
                (1, "to_a", 3),
                (1, "to_b", 4),
                (2, "to_a", 3),
                (2, "to_b", 4),
                (3, "finish", 5),
                (4, "finish", 5),
            ],
            top=0,
            bottom=5,
            labels={0: "top", 1: "c", 2: "d", 3: "a", 4: "b", 5: "end"},
        )

    def test_not_lattice(self) -> None:
        r = check_lattice(self.ss)
        assert r.is_lattice is False

    def test_counterexample_exists(self) -> None:
        r = check_lattice(self.ss)
        assert r.counterexample is not None


# ===================================================================
# Meet/Join API tests
# ===================================================================

class TestMeetJoinAPI:
    """Direct tests for compute_meet and compute_join."""

    def test_meet_top_with_anything(self) -> None:
        """Meet(top, x) = x for any x (top is the greatest element)."""
        ss = _ss("a . b . end")
        for s in ss.states:
            assert compute_meet(ss, ss.top, s) == s

    def test_join_bottom_with_anything(self) -> None:
        """Join(bottom, x) = x for any x (bottom is the least element)."""
        ss = _ss("a . b . end")
        for s in ss.states:
            assert compute_join(ss, ss.bottom, s) == s

    def test_meet_bottom_with_anything(self) -> None:
        """Meet(bottom, x) = bottom for any x."""
        ss = _ss("a . b . end")
        for s in ss.states:
            assert compute_meet(ss, ss.bottom, s) == ss.bottom

    def test_join_top_with_anything(self) -> None:
        """Join(top, x) = top for any x."""
        ss = _ss("a . b . end")
        for s in ss.states:
            assert compute_join(ss, ss.top, s) == ss.top

    def test_meet_in_chain(self) -> None:
        """In a chain a > b > end, meet(a, b) = b (the lower one)."""
        ss = _ss("a . b . end")
        tr = {l: t for s, l, t in ss.transitions if s == ss.top}
        mid = tr["a"]
        assert compute_meet(ss, ss.top, mid) == mid
        assert compute_meet(ss, mid, ss.bottom) == ss.bottom

    def test_join_in_chain(self) -> None:
        """In a chain a > b > end, join(a, b) = a (the higher one)."""
        ss = _ss("a . b . end")
        tr = {l: t for s, l, t in ss.transitions if s == ss.top}
        mid = tr["a"]
        assert compute_join(ss, ss.top, mid) == ss.top
        assert compute_join(ss, mid, ss.bottom) == mid

    def test_meet_nonexistent_state(self) -> None:
        """Computing meet with a state not in the state space returns None."""
        ss = _ss("end")
        assert compute_meet(ss, ss.top, 999) is None

    def test_join_nonexistent_state(self) -> None:
        """Computing join with a state not in the state space returns None."""
        ss = _ss("end")
        assert compute_join(ss, ss.top, 999) is None


# ===================================================================
# SCC map tests
# ===================================================================

class TestSCCMap:
    """Verify that scc_map correctly groups cyclic states."""

    def test_self_loop_same_scc(self) -> None:
        """In ``rec X . &{next: X, done: end}``, top loops to itself → 1 SCC."""
        ss = _ss("rec X . &{next: X, done: end}")
        r = check_lattice(ss)
        # top maps to itself (representative)
        assert r.scc_map[ss.top] == ss.top

    def test_cycle_same_scc(self) -> None:
        """In ``rec X . &{a: &{b: X}, done: end}``, top and after_a share SCC."""
        ss = _ss("rec X . &{a: &{b: X}, done: end}")
        r = check_lattice(ss)
        # Find after_a
        tr = {l: t for s, l, t in ss.transitions if s == ss.top}
        after_a = tr["a"]
        # top and after_a should be in the same SCC
        assert r.scc_map[ss.top] == r.scc_map[after_a]

    def test_end_separate_scc(self) -> None:
        """End state is always in its own SCC (no outgoing edges)."""
        ss = _ss("a . b . end")
        r = check_lattice(ss)
        assert r.scc_map[ss.bottom] == ss.bottom


# ===================================================================
# Full spec example
# ===================================================================

class TestSpecSharedFile:
    """SharedFile protocol from spec §3.2:

    ``init . &{open: +{OK: (read . end || write . end) . close . end, ERROR: end}}``
    """

    def test_is_lattice(self) -> None:
        src = "init . &{open: +{OK: (read . end || write . end) . close . end, ERROR: end}}"
        r = _check(src)
        assert r.is_lattice is True

    def test_has_all_properties(self) -> None:
        src = "init . &{open: +{OK: (read . end || write . end) . close . end, ERROR: end}}"
        r = _check(src)
        assert r.has_top is True
        assert r.has_bottom is True
        assert r.all_meets_exist is True
        assert r.all_joins_exist is True


class TestSpecConcurrentFileRecursive:
    """Concurrent file with recursive branches:

    ``open . (rec X . &{read: X, doneReading: end} || rec Y . &{write: Y, doneWriting: end}) . close . end``
    """

    def test_is_lattice(self) -> None:
        src = (
            "open . (rec X . &{read: X, doneReading: end} "
            "|| rec Y . &{write: Y, doneWriting: end}) . close . end"
        )
        r = _check(src)
        assert r.is_lattice is True


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_single_method(self) -> None:
        """``a . end`` — two-element chain."""
        r = _check("a . end")
        assert r.is_lattice is True
        assert r.num_scc == 2

    def test_long_chain(self) -> None:
        """``a . b . c . d . e . end`` — 6-element chain."""
        r = _check("a . b . c . d . e . end")
        assert r.is_lattice is True
        assert r.num_scc == 6

    def test_select_diamond(self) -> None:
        """``+{OK: a . end, ERR: b . end}`` — diamond with select."""
        r = _check("+{OK: a . end, ERR: b . end}")
        assert r.is_lattice is True
        assert r.num_scc == 4

    def test_three_branches(self) -> None:
        """``&{a: end, b: end, c: end}`` — fan to single end."""
        r = _check("&{a: end, b: end, c: end}")
        assert r.is_lattice is True
        assert r.num_scc == 2
