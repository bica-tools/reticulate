"""Tests for join/meet irreducibles analysis (Step 30i).

Tests cover:
  - Simple chains (every non-bottom element is join-irreducible)
  - Diamond lattice M3 (3 atoms, no join-irreducibles except atoms)
  - Boolean lattice 2^3
  - Benchmark protocols (SMTP, OAuth, Iterator, Two-Buyer)
  - Birkhoff dual reconstruction for distributive lattices
  - Self-dual cardinality property
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice, check_distributive
from reticulate.irreducibles import (
    BirkhoffDualResult,
    IrreduciblesResult,
    analyze_irreducibles,
    birkhoff_dual,
    is_join_irreducible,
    is_meet_irreducible,
    join_irreducibles,
    meet_irreducibles,
    _hasse_edges,
    _lower_covers,
    _upper_covers,
)


# ---------------------------------------------------------------------------
# Helpers: build small lattices by hand
# ---------------------------------------------------------------------------

def _chain(n: int) -> StateSpace:
    """Build a chain lattice 0 > 1 > ... > n-1 (top=0, bottom=n-1)."""
    ss = StateSpace()
    ss.states = set(range(n))
    ss.top = 0
    ss.bottom = n - 1
    ss.transitions = [(i, f"t{i}", i + 1) for i in range(n - 1)]
    ss.labels = {i: f"s{i}" for i in range(n)}
    return ss


def _diamond() -> StateSpace:
    """Build the diamond lattice M3: top -> {a, b, c} -> bottom.

    States: 0=top, 1=a, 2=b, 3=c, 4=bottom
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3, 4}
    ss.top = 0
    ss.bottom = 4
    ss.transitions = [
        (0, "a", 1), (0, "b", 2), (0, "c", 3),
        (1, "x", 4), (2, "y", 4), (3, "z", 4),
    ]
    ss.labels = {0: "top", 1: "a", 2: "b", 3: "c", 4: "bot"}
    return ss


def _boolean_2_3() -> StateSpace:
    """Build the Boolean lattice 2^3 (power set of {a,b,c}).

    8 states arranged as a cube: top={a,b,c}, bottom=empty.
    """
    ss = StateSpace()
    # States: 0=abc(top), 1=ab, 2=ac, 3=bc, 4=a, 5=b, 6=c, 7=empty(bottom)
    ss.states = set(range(8))
    ss.top = 0
    ss.bottom = 7
    ss.transitions = [
        # Top to 2-element sets
        (0, "drop_c", 1), (0, "drop_b", 2), (0, "drop_a", 3),
        # 2-element sets to 1-element sets
        (1, "drop_b", 4), (1, "drop_a", 5),
        (2, "drop_c", 4), (2, "drop_a", 6),
        (3, "drop_c", 5), (3, "drop_b", 6),
        # 1-element sets to bottom
        (4, "drop_a", 7), (5, "drop_b", 7), (6, "drop_c", 7),
    ]
    ss.labels = {
        0: "{a,b,c}", 1: "{a,b}", 2: "{a,c}", 3: "{b,c}",
        4: "{a}", 5: "{b}", 6: "{c}", 7: "{}",
    }
    return ss


def _n5() -> StateSpace:
    """Build the pentagon lattice N5.

    top -> a -> b -> bottom, top -> c -> bottom, with a || c and b || c.
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3, 4}
    ss.top = 0
    ss.bottom = 4
    ss.transitions = [
        (0, "x", 1), (0, "y", 3),
        (1, "z", 2),
        (2, "w", 4),
        (3, "v", 4),
    ]
    ss.labels = {0: "top", 1: "a", 2: "b", 3: "c", 4: "bot"}
    return ss


def _two_chain() -> StateSpace:
    """Build a 2-element chain (top -> bottom)."""
    return _chain(2)


# ---------------------------------------------------------------------------
# Tests: chain lattices
# ---------------------------------------------------------------------------

class TestChainLattice:
    """In a chain of n elements, every element except bottom is join-irreducible
    (each has exactly one lower cover), and every element except top is
    meet-irreducible (each has exactly one upper cover)."""

    def test_chain_3_join_irreducibles(self) -> None:
        ss = _chain(3)
        ji = join_irreducibles(ss)
        # States: 0(top), 1, 2(bottom). JI = {0, 1} minus bottom = {0, 1}
        # But top has 1 lower cover (1), so top is JI
        # State 1 has 1 lower cover (2), so 1 is JI
        # Bottom (2) is never JI
        assert ji == {0, 1}

    def test_chain_3_meet_irreducibles(self) -> None:
        ss = _chain(3)
        mi = meet_irreducibles(ss)
        # top (0) is never MI
        # State 1 has 1 upper cover (0), so MI
        # State 2 has 1 upper cover (1), so MI
        assert mi == {1, 2}

    def test_chain_4_join_irreducibles(self) -> None:
        ss = _chain(4)
        ji = join_irreducibles(ss)
        assert ji == {0, 1, 2}  # all except bottom (3)

    def test_chain_4_meet_irreducibles(self) -> None:
        ss = _chain(4)
        mi = meet_irreducibles(ss)
        assert mi == {1, 2, 3}  # all except top (0)

    def test_chain_2_join_irreducibles(self) -> None:
        ss = _two_chain()
        ji = join_irreducibles(ss)
        # top (0) has 1 lower cover (1=bottom), but bottom is excluded
        # Actually top has lower cover = bottom. top != bottom, so top is JI.
        assert ji == {0}

    def test_chain_2_meet_irreducibles(self) -> None:
        ss = _two_chain()
        mi = meet_irreducibles(ss)
        assert mi == {1}

    def test_chain_self_dual_cardinality(self) -> None:
        """In a chain, |J| = |M| = n-1."""
        for n in [2, 3, 4, 5]:
            ss = _chain(n)
            ji = join_irreducibles(ss)
            mi = meet_irreducibles(ss)
            assert len(ji) == len(mi) == n - 1


# ---------------------------------------------------------------------------
# Tests: diamond lattice M3
# ---------------------------------------------------------------------------

class TestDiamondLattice:
    def test_diamond_join_irreducibles(self) -> None:
        ss = _diamond()
        ji = join_irreducibles(ss)
        # Atoms {1, 2, 3} each have exactly one lower cover (bottom)
        # Top has 3 lower covers => NOT join-irreducible
        assert ji == {1, 2, 3}

    def test_diamond_meet_irreducibles(self) -> None:
        ss = _diamond()
        mi = meet_irreducibles(ss)
        # Atoms {1, 2, 3} each have exactly one upper cover (top)
        # Bottom has 3 upper covers => NOT meet-irreducible
        assert mi == {1, 2, 3}

    def test_diamond_self_dual_cardinality(self) -> None:
        ss = _diamond()
        ji = join_irreducibles(ss)
        mi = meet_irreducibles(ss)
        assert len(ji) == len(mi)

    def test_diamond_is_not_distributive(self) -> None:
        ss = _diamond()
        dr = check_distributive(ss)
        assert not dr.is_distributive

    def test_diamond_single_element_checks(self) -> None:
        ss = _diamond()
        assert is_join_irreducible(ss, 1)
        assert is_join_irreducible(ss, 2)
        assert is_join_irreducible(ss, 3)
        assert not is_join_irreducible(ss, 0)  # top
        assert not is_join_irreducible(ss, 4)  # bottom


# ---------------------------------------------------------------------------
# Tests: Boolean lattice 2^3
# ---------------------------------------------------------------------------

class TestBooleanLattice:
    def test_boolean_join_irreducibles(self) -> None:
        ss = _boolean_2_3()
        ji = join_irreducibles(ss)
        # In 2^3, atoms are {a}=4, {b}=5, {c}=6
        # Each atom has exactly one lower cover (bottom)
        # 2-element sets have 2 lower covers, top has 3 lower covers
        assert ji == {4, 5, 6}

    def test_boolean_meet_irreducibles(self) -> None:
        ss = _boolean_2_3()
        mi = meet_irreducibles(ss)
        # Coatoms are {a,b}=1, {a,c}=2, {b,c}=3
        # Each coatom has exactly one upper cover (top)
        assert mi == {1, 2, 3}

    def test_boolean_cardinality(self) -> None:
        ss = _boolean_2_3()
        ji = join_irreducibles(ss)
        mi = meet_irreducibles(ss)
        assert len(ji) == 3  # n atoms in 2^n
        assert len(mi) == 3  # n coatoms in 2^n

    def test_boolean_is_lattice(self) -> None:
        ss = _boolean_2_3()
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_boolean_birkhoff_downset_count(self) -> None:
        """For 2^3, J(L) is a 3-element antichain, so downsets = 2^3 = 8."""
        ss = _boolean_2_3()
        bd = birkhoff_dual(ss)
        # J(L) has 3 pairwise incomparable atoms
        assert len(bd.join_irr) == 3
        # Downsets of a 3-element antichain = 2^3 = 8
        assert bd.downset_count == 8
        assert bd.downset_count == bd.lattice_size


# ---------------------------------------------------------------------------
# Tests: pentagon lattice N5
# ---------------------------------------------------------------------------

class TestPentagonLattice:
    def test_n5_join_irreducibles(self) -> None:
        ss = _n5()
        ji = join_irreducibles(ss)
        # a(1) has 1 lower cover (b=2), b(2) has 1 lower cover (bot=4),
        # c(3) has 1 lower cover (bot=4), top(0) has 2 lower covers (a, c)
        assert ji == {1, 2, 3}

    def test_n5_meet_irreducibles(self) -> None:
        ss = _n5()
        mi = meet_irreducibles(ss)
        # b(2) has 1 upper cover (a=1), c(3) has 1 upper cover (top=0),
        # a(1) has 1 upper cover (top=0), bot(4) has 2 upper covers (b, c)
        assert mi == {1, 2, 3}

    def test_n5_not_distributive(self) -> None:
        ss = _n5()
        dr = check_distributive(ss)
        assert not dr.is_distributive


# ---------------------------------------------------------------------------
# Tests: single-element checks
# ---------------------------------------------------------------------------

class TestSingleElementChecks:
    def test_is_join_irreducible_bottom_always_false(self) -> None:
        ss = _chain(3)
        assert not is_join_irreducible(ss, ss.bottom)

    def test_is_meet_irreducible_top_always_false(self) -> None:
        ss = _chain(3)
        assert not is_meet_irreducible(ss, ss.top)

    def test_invalid_state_raises(self) -> None:
        ss = _chain(3)
        with pytest.raises(ValueError):
            is_join_irreducible(ss, 999)
        with pytest.raises(ValueError):
            is_meet_irreducible(ss, 999)

    def test_boolean_atoms_are_ji(self) -> None:
        ss = _boolean_2_3()
        for atom in [4, 5, 6]:
            assert is_join_irreducible(ss, atom)
        for non_atom in [0, 1, 2, 3, 7]:
            assert not is_join_irreducible(ss, non_atom)


# ---------------------------------------------------------------------------
# Tests: Birkhoff dual
# ---------------------------------------------------------------------------

class TestBirkhoffDual:
    def test_chain_3_birkhoff(self) -> None:
        ss = _chain(3)
        bd = birkhoff_dual(ss)
        assert bd.is_distributive
        # Chain of 3: J(L) = {0, 1}, ordered 0 > 1
        assert len(bd.join_irr) == 2
        # Downsets of a 2-chain: {}, {1}, {0, 1} => 3 = |L|
        assert bd.downset_count == 3

    def test_chain_4_birkhoff(self) -> None:
        ss = _chain(4)
        bd = birkhoff_dual(ss)
        assert bd.is_distributive
        assert len(bd.join_irr) == 3
        # Downsets of a 3-chain: {}, {2}, {1,2}, {0,1,2} => 4 = |L|
        assert bd.downset_count == 4

    def test_boolean_birkhoff_dual_is_antichain(self) -> None:
        """In 2^3, J(L) is a 3-element antichain."""
        ss = _boolean_2_3()
        bd = birkhoff_dual(ss)
        # All pairs in the dual order should be reflexive only (antichain)
        non_reflexive = {(a, b) for a, b in bd.dual_order if a != b}
        assert len(non_reflexive) == 0

    def test_compression_ratio(self) -> None:
        ss = _boolean_2_3()
        bd = birkhoff_dual(ss)
        assert bd.compression_ratio == pytest.approx(3 / 8)

    def test_diamond_birkhoff_not_distributive(self) -> None:
        ss = _diamond()
        bd = birkhoff_dual(ss)
        assert not bd.is_distributive
        # Even though not distributive, we still compute J(L)
        assert len(bd.join_irr) == 3


# ---------------------------------------------------------------------------
# Tests: analyze_irreducibles
# ---------------------------------------------------------------------------

class TestAnalyzeIrreducibles:
    def test_chain_analysis(self) -> None:
        ss = _chain(4)
        result = analyze_irreducibles(ss)
        assert result.is_lattice
        assert result.num_join_irr == 3
        assert result.num_meet_irr == 3
        assert result.is_self_dual_cardinality
        assert result.birkhoff is not None
        assert result.birkhoff.is_distributive

    def test_diamond_analysis(self) -> None:
        ss = _diamond()
        result = analyze_irreducibles(ss)
        assert result.is_lattice
        assert result.num_join_irr == 3
        assert result.num_meet_irr == 3
        assert result.is_self_dual_cardinality

    def test_non_lattice_returns_empty(self) -> None:
        """A non-lattice state space should return empty irreducibles."""
        ss = StateSpace()
        ss.states = {0, 1, 2, 3}
        ss.top = 0
        ss.bottom = 3
        # Two incomparable paths with no join
        ss.transitions = [
            (0, "a", 1), (0, "b", 2),
            # 1 and 2 both go to 3, but we add a cross edge to break lattice
        ]
        # Actually let's just make it so meet doesn't exist
        # Simple: fork without a meet
        ss.states = {0, 1, 2}
        ss.top = 0
        ss.bottom = 1  # but 2 doesn't reach 1
        ss.transitions = [(0, "a", 1), (0, "b", 2)]
        result = analyze_irreducibles(ss)
        assert not result.is_lattice
        assert result.num_join_irr == 0
        assert result.birkhoff is None


# ---------------------------------------------------------------------------
# Tests: session type benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarkProtocols:
    """Test irreducibles on real benchmark protocols."""

    def test_iterator(self) -> None:
        t = parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        ss = build_statespace(t)
        lr = check_lattice(ss)
        assert lr.is_lattice
        ji = join_irreducibles(ss)
        mi = meet_irreducibles(ss)
        assert len(ji) > 0
        assert len(mi) > 0

    def test_file_object(self) -> None:
        t = parse("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")
        ss = build_statespace(t)
        lr = check_lattice(ss)
        assert lr.is_lattice
        ji = join_irreducibles(ss)
        assert len(ji) > 0

    def test_smtp(self) -> None:
        t = parse(
            "&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: "
            "+{OK: X, ERR: X}}}, quit: end}}}"
        )
        ss = build_statespace(t)
        lr = check_lattice(ss)
        assert lr.is_lattice
        result = analyze_irreducibles(ss)
        assert result.is_lattice
        assert result.num_join_irr > 0

    def test_oauth(self) -> None:
        t = parse(
            "&{requestAuth: +{GRANTED: &{getToken: +{TOKEN: rec X . "
            "&{useToken: X, refreshToken: +{OK: X, EXPIRED: end}, "
            "revoke: end}, ERROR: end}}, DENIED: end}}"
        )
        ss = build_statespace(t)
        lr = check_lattice(ss)
        assert lr.is_lattice
        result = analyze_irreducibles(ss)
        assert result.is_lattice
        assert result.num_join_irr > 0
        assert result.num_meet_irr > 0

    def test_two_buyer_parallel(self) -> None:
        t = parse(
            "&{lookup: &{getPrice: (&{proposeA: end} || "
            "&{proposeB: +{ACCEPT: &{pay: end}, REJECT: end}})}}"
        )
        ss = build_statespace(t)
        lr = check_lattice(ss)
        assert lr.is_lattice
        ji = join_irreducibles(ss)
        mi = meet_irreducibles(ss)
        assert len(ji) > 0
        assert len(mi) > 0

    def test_simple_branch(self) -> None:
        t = parse("&{a: end, b: end}")
        ss = build_statespace(t)
        lr = check_lattice(ss)
        assert lr.is_lattice
        ji = join_irreducibles(ss)
        # &{a: end, b: end} has only 2 states: top and end (bottom).
        # Both a and b lead to the same end state, so top has 1 lower cover.
        # top is JI (1 lower cover, not bottom), bottom is never JI.
        assert len(ji) == 1
        assert ss.top in ji

    def test_simple_select(self) -> None:
        t = parse("+{a: end, b: end}")
        ss = build_statespace(t)
        lr = check_lattice(ss)
        assert lr.is_lattice
        ji = join_irreducibles(ss)
        mi = meet_irreducibles(ss)
        assert len(ji) > 0

    def test_http_connection(self) -> None:
        t = parse(
            "&{connect: rec X . &{request: +{OK200: &{readBody: X}, "
            "ERR4xx: X, ERR5xx: X}, close: end}}"
        )
        ss = build_statespace(t)
        lr = check_lattice(ss)
        assert lr.is_lattice
        result = analyze_irreducibles(ss)
        assert result.is_lattice


# ---------------------------------------------------------------------------
# Tests: Birkhoff's theorem verification on distributive benchmarks
# ---------------------------------------------------------------------------

class TestBirkhoffTheorem:
    """For distributive lattices, |downsets of J(L)| = |L|."""

    def test_simple_branch_birkhoff(self) -> None:
        t = parse("&{a: end, b: end}")
        ss = build_statespace(t)
        bd = birkhoff_dual(ss)
        if bd.is_distributive:
            assert bd.downset_count == bd.lattice_size

    def test_chain_types_birkhoff(self) -> None:
        """A chain session type should produce a chain lattice."""
        t = parse("&{a: &{b: &{c: end}}}")
        ss = build_statespace(t)
        bd = birkhoff_dual(ss)
        assert bd.is_distributive
        assert bd.downset_count == bd.lattice_size

    def test_iterator_birkhoff(self) -> None:
        """Iterator has SCCs from recursion; Birkhoff applies to quotient."""
        t = parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        ss = build_statespace(t)
        bd = birkhoff_dual(ss)
        # The iterator lattice has 4 raw states but 2 SCCs in the quotient.
        # Birkhoff's theorem applies to the quotient poset, not raw states.
        # For recursive types, the lattice_size counts raw states, so the
        # theorem comparison only holds for acyclic state spaces.
        assert bd.is_distributive
        assert len(bd.join_irr) > 0


# ---------------------------------------------------------------------------
# Tests: internal helpers
# ---------------------------------------------------------------------------

class TestInternalHelpers:
    def test_hasse_edges_chain(self) -> None:
        ss = _chain(3)
        edges = _hasse_edges(ss)
        assert len(edges) == 2  # 0->1, 1->2

    def test_lower_covers_diamond(self) -> None:
        ss = _diamond()
        lc = _lower_covers(ss)
        assert len(lc[0]) == 3  # top covers a, b, c
        assert len(lc[1]) == 1  # a covers bottom
        assert len(lc[4]) == 0  # bottom covers nothing

    def test_upper_covers_diamond(self) -> None:
        ss = _diamond()
        uc = _upper_covers(ss)
        assert len(uc[0]) == 0  # top has no upper cover
        assert len(uc[1]) == 1  # a has upper cover: top
        assert len(uc[4]) == 3  # bottom has 3 upper covers
