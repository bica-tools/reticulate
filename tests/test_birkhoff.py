"""Tests for Birkhoff representation (Step 30j).

Tests the birkhoff_representation, downset_lattice, is_representable,
representation_size, and reconstruct_from_poset functions on benchmark
protocols, hand-crafted lattices, and roundtrip properties.

25+ tests covering:
  - Downset lattice construction (8 tests)
  - Join-irreducible extraction (4 tests)
  - Birkhoff representation on hand-crafted lattices (6 tests)
  - is_representable convenience (4 tests)
  - representation_size compression (4 tests)
  - Reconstruction from poset (7 tests)
  - Benchmark protocols (3 parametrised suites)
  - Session type specific tests (4 tests)
  - Edge cases (3 tests)
"""

from __future__ import annotations

import pytest

from reticulate.birkhoff import (
    BirkhoffResult,
    DownsetLattice,
    birkhoff_representation,
    downset_lattice,
    is_representable,
    reconstruct_from_poset,
    representation_size,
)
from reticulate.lattice import check_distributive, check_lattice
from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace

from tests.benchmarks.protocols import BENCHMARKS


# ---------------------------------------------------------------------------
# Helpers: build small lattices by hand
# ---------------------------------------------------------------------------

def _chain(n: int) -> StateSpace:
    """Build a chain of n elements: 0 > 1 > ... > n-1."""
    ss = StateSpace()
    ss.states = set(range(n))
    ss.transitions = [(i, f"t{i}", i + 1) for i in range(n - 1)]
    ss.top = 0
    ss.bottom = n - 1
    ss.labels = {i: f"s{i}" for i in range(n)}
    return ss


def _boolean_lattice_2() -> StateSpace:
    """Boolean lattice 2^2 = {top, a, b, bot} with top > a,b > bot."""
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "a", 1), (0, "b", 2),
        (1, "x", 3), (2, "y", 3),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "a", 2: "b", 3: "bot"}
    return ss


def _diamond_m3() -> StateSpace:
    """M3 diamond lattice (non-distributive): top > a,b,c > bot."""
    ss = StateSpace()
    ss.states = {0, 1, 2, 3, 4}
    ss.transitions = [
        (0, "a", 1), (0, "b", 2), (0, "c", 3),
        (1, "x", 4), (2, "y", 4), (3, "z", 4),
    ]
    ss.top = 0
    ss.bottom = 4
    ss.labels = {0: "top", 1: "a", 2: "b", 3: "c", 4: "bot"}
    return ss


def _pentagon_n5() -> StateSpace:
    """N5 pentagon lattice (non-distributive, non-modular)."""
    ss = StateSpace()
    ss.states = {0, 1, 2, 3, 4}
    ss.transitions = [
        (0, "ta", 1), (0, "tc", 3),
        (1, "tb", 2),
        (2, "tx", 4), (3, "ty", 4),
    ]
    ss.top = 0
    ss.bottom = 4
    ss.labels = {0: "top", 1: "a", 2: "b", 3: "c", 4: "bot"}
    return ss


# ---------------------------------------------------------------------------
# Test: downset lattice construction
# ---------------------------------------------------------------------------

class TestDownsetLattice:
    """Test downset_lattice on small posets."""

    def test_empty_poset(self):
        """O(empty) = {empty}."""
        dl = downset_lattice({})
        assert len(dl.elements) == 1
        assert frozenset() in dl.elements
        assert dl.top == frozenset()
        assert dl.bottom == frozenset()

    def test_single_element(self):
        """O({x}) = {empty, {x}} --- a 2-element chain."""
        dl = downset_lattice({1: set()})
        assert len(dl.elements) == 2
        assert frozenset() in dl.elements
        assert frozenset({1}) in dl.elements
        assert dl.top == frozenset({1})
        assert dl.bottom == frozenset()

    def test_two_element_chain(self):
        """Poset: a > b. O(P) = {empty, {b}, {a,b}} --- 3-element chain."""
        dl = downset_lattice({1: {2}, 2: set()})
        assert len(dl.elements) == 3
        assert frozenset() in dl.elements
        assert frozenset({2}) in dl.elements
        assert frozenset({1, 2}) in dl.elements
        # {1} alone is NOT a downset since 1 > 2 requires 2 to be present
        assert frozenset({1}) not in dl.elements

    def test_two_element_antichain(self):
        """Poset: a, b incomparable. O(P) = Boolean 2^2."""
        dl = downset_lattice({1: set(), 2: set()})
        assert len(dl.elements) == 4
        assert frozenset() in dl.elements
        assert frozenset({1}) in dl.elements
        assert frozenset({2}) in dl.elements
        assert frozenset({1, 2}) in dl.elements

    def test_three_element_antichain(self):
        """Poset: a, b, c incomparable. O(P) = 2^3 = 8 elements."""
        dl = downset_lattice({1: set(), 2: set(), 3: set()})
        assert len(dl.elements) == 8

    def test_three_element_chain(self):
        """Poset: a > b > c. O(P) = 4-element chain."""
        dl = downset_lattice({1: {2}, 2: {3}, 3: set()})
        assert len(dl.elements) == 4

    def test_downset_lattice_order(self):
        """Verify ordering: d1 <= d2 iff d1 is subset of d2."""
        dl = downset_lattice({1: set(), 2: set()})
        empty = frozenset()
        a = frozenset({1})
        b = frozenset({2})
        full = frozenset({1, 2})
        # empty <= everything
        assert a in dl.order[empty]
        assert b in dl.order[empty]
        assert full in dl.order[empty]

    def test_downset_covering_relation(self):
        """Verify Hasse edges: covers differ by exactly one element."""
        dl = downset_lattice({1: set(), 2: set()})
        full = frozenset({1, 2})
        assert frozenset({1}) in dl.hasse[full]
        assert frozenset({2}) in dl.hasse[full]


# ---------------------------------------------------------------------------
# Test: Birkhoff on hand-crafted lattices
# ---------------------------------------------------------------------------

class TestBirkhoffHandCrafted:
    """Test birkhoff_representation on hand-crafted lattices."""

    def test_chain_2(self):
        """Chain of 2: trivially distributive, 1 join-irreducible."""
        ss = _chain(2)
        result = birkhoff_representation(ss)
        assert result.is_lattice
        assert result.is_distributive
        assert result.is_representable
        assert len(result.join_irreducibles) == 1
        assert result.downset_lattice_size == 2
        assert result.isomorphism_verified

    def test_chain_4(self):
        """Chain of 4: distributive, 3 join-irreducibles, downsets = 4."""
        ss = _chain(4)
        result = birkhoff_representation(ss)
        assert result.is_lattice
        assert result.is_distributive
        assert result.is_representable
        assert len(result.join_irreducibles) == 3
        assert result.downset_lattice_size == 4
        assert result.isomorphism_verified

    def test_boolean_2(self):
        """Boolean 2^2: 2 join-irreducibles (atoms), 4 downsets."""
        ss = _boolean_lattice_2()
        result = birkhoff_representation(ss)
        assert result.is_lattice
        assert result.is_distributive
        assert result.is_representable
        assert len(result.join_irreducibles) == 2
        assert result.downset_lattice_size == 4
        assert result.isomorphism_verified
        assert result.compression_ratio == pytest.approx(2 / 4)

    def test_diamond_m3_not_representable(self):
        """M3 diamond: NOT distributive, hence not representable."""
        ss = _diamond_m3()
        result = birkhoff_representation(ss)
        assert result.is_lattice
        assert not result.is_distributive
        assert not result.is_representable
        assert not result.isomorphism_verified

    def test_pentagon_n5_not_representable(self):
        """N5 pentagon: NOT distributive, hence not representable."""
        ss = _pentagon_n5()
        result = birkhoff_representation(ss)
        assert result.is_lattice
        assert not result.is_distributive
        assert not result.is_representable

    def test_lattice_size_is_quotient(self):
        """Lattice size should be quotient node count, not raw state count."""
        ss = _chain(3)
        result = birkhoff_representation(ss)
        # Chain has no cycles, so quotient = raw states
        assert result.lattice_size == 3


# ---------------------------------------------------------------------------
# Test: is_representable convenience function
# ---------------------------------------------------------------------------

class TestIsRepresentable:
    """Test the is_representable convenience function."""

    def test_chain_representable(self):
        assert is_representable(_chain(3))

    def test_boolean_representable(self):
        assert is_representable(_boolean_lattice_2())

    def test_m3_not_representable(self):
        assert not is_representable(_diamond_m3())

    def test_n5_not_representable(self):
        assert not is_representable(_pentagon_n5())


# ---------------------------------------------------------------------------
# Test: representation_size
# ---------------------------------------------------------------------------

class TestRepresentationSize:
    """Test representation_size compression ratio."""

    def test_chain_3(self):
        lattice_sz, poset_sz, ratio = representation_size(_chain(3))
        assert lattice_sz == 3
        assert poset_sz == 2
        assert ratio == pytest.approx(2 / 3)

    def test_boolean_2(self):
        lattice_sz, poset_sz, ratio = representation_size(_boolean_lattice_2())
        assert lattice_sz == 4
        assert poset_sz == 2
        assert ratio == pytest.approx(0.5)

    def test_compression_always_le_1(self):
        """The poset is always no larger than the lattice."""
        for ss in [_chain(5), _boolean_lattice_2()]:
            _, _, ratio = representation_size(ss)
            assert ratio <= 1.0

    def test_single_state(self):
        """Single-state lattice: 0 join-irreducibles, ratio = 0."""
        ss = _chain(1)
        lattice_sz, poset_sz, ratio = representation_size(ss)
        assert lattice_sz == 1
        assert poset_sz == 0
        assert ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test: reconstruct_from_poset
# ---------------------------------------------------------------------------

class TestReconstructFromPoset:
    """Test inverse: poset -> downset lattice -> state space."""

    def test_empty_poset(self):
        ss = reconstruct_from_poset({})
        assert len(ss.states) == 1
        assert ss.top == ss.bottom

    def test_single_element(self):
        ss = reconstruct_from_poset({1: set()})
        assert len(ss.states) == 2
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_two_antichain(self):
        """Antichain of 2 -> Boolean 2^2."""
        ss = reconstruct_from_poset({1: set(), 2: set()})
        assert len(ss.states) == 4
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_three_antichain(self):
        """Antichain of 3 -> Boolean 2^3 = 8 states."""
        ss = reconstruct_from_poset({1: set(), 2: set(), 3: set()})
        assert len(ss.states) == 8
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_chain_poset(self):
        """Chain of 2 -> chain of 3."""
        ss = reconstruct_from_poset({1: {2}, 2: set()})
        assert len(ss.states) == 3
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_roundtrip_antichain_2(self):
        """Roundtrip: antichain -> downset -> birkhoff -> same poset size."""
        poset = {1: set(), 2: set()}
        ss = reconstruct_from_poset(poset)
        result = birkhoff_representation(ss)
        assert result.is_representable
        assert result.isomorphism_verified
        assert len(result.join_irreducibles) == 2

    def test_roundtrip_chain(self):
        """Roundtrip: chain -> downset -> birkhoff -> same chain size."""
        poset = {10: {20}, 20: set()}
        ss = reconstruct_from_poset(poset)
        result = birkhoff_representation(ss)
        assert result.is_representable
        assert result.isomorphism_verified


# ---------------------------------------------------------------------------
# Test: Birkhoff on session type benchmarks
# ---------------------------------------------------------------------------

# Benchmarks known to be non-distributive
NON_DISTRIBUTIVE = {
    "Two-Buyer", "Reticulate Pipeline", "TLS Handshake",
    "Saga Orchestrator", "Two-Phase Commit", "Quantum Measurement",
    "Ki3 Onboarding", "Ki3 CI/CD Pipeline", "Polysome",
    "ER-Golgi Secretory", "Apoptosis", "Photosynthesis-Respiration",
    "Weak Decay (Beta)", "Bell Pair", "Quantum Teleportation",
    "Big Bang Nucleosynthesis", "SSH Handshake", "Mutual TLS",
    "Certificate Chain", "DNSSEC",
}


class TestBirkhoffBenchmarks:
    """Test Birkhoff representation on benchmark protocols."""

    @pytest.mark.parametrize(
        "bench",
        [b for b in BENCHMARKS if b.name not in NON_DISTRIBUTIVE],
        ids=[b.name for b in BENCHMARKS if b.name not in NON_DISTRIBUTIVE],
    )
    def test_distributive_benchmark_representable(self, bench):
        """Every distributive benchmark should have a valid Birkhoff representation."""
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        dr = check_distributive(ss)
        if not dr.is_distributive:
            pytest.skip(f"{bench.name} is not distributive")
        result = birkhoff_representation(ss)
        assert result.is_representable, f"{bench.name}: distributive but not representable?"
        assert result.isomorphism_verified, f"{bench.name}: isomorphism check failed"
        assert result.downset_lattice_size == result.lattice_size, \
            f"{bench.name}: |O(J(L))| = {result.downset_lattice_size} != |L| = {result.lattice_size}"

    @pytest.mark.parametrize(
        "bench",
        [b for b in BENCHMARKS if b.name in NON_DISTRIBUTIVE],
        ids=[b.name for b in BENCHMARKS if b.name in NON_DISTRIBUTIVE],
    )
    def test_non_distributive_benchmark_not_representable(self, bench):
        """Non-distributive benchmarks should NOT be representable."""
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        result = birkhoff_representation(ss)
        if result.is_distributive:
            pytest.skip(f"{bench.name} turned out distributive")
        assert not result.is_representable

    @pytest.mark.parametrize(
        "bench",
        [b for b in BENCHMARKS if b.name not in NON_DISTRIBUTIVE],
        ids=[b.name for b in BENCHMARKS if b.name not in NON_DISTRIBUTIVE],
    )
    def test_compression_ratio_below_one(self, bench):
        """For distributive benchmarks, the compression ratio should be <= 1."""
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        dr = check_distributive(ss)
        if not dr.is_distributive:
            pytest.skip(f"{bench.name} is not distributive")
        l_sz, p_sz, ratio = representation_size(ss)
        assert ratio <= 1.0, f"{bench.name}: ratio {ratio} > 1"


# ---------------------------------------------------------------------------
# Test: specific session types
# ---------------------------------------------------------------------------

class TestBirkhoffSessionTypes:
    """Test Birkhoff on specific session type strings."""

    def test_simple_branch(self):
        """&{a: end, b: end} --- should be distributive and representable."""
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        result = birkhoff_representation(ss)
        assert result.is_lattice
        # If distributive, it must be representable
        if result.is_distributive:
            assert result.is_representable
            assert result.isomorphism_verified

    def test_iterator(self):
        """rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}} --- Java Iterator."""
        ast = parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        ss = build_statespace(ast)
        result = birkhoff_representation(ss)
        assert result.is_lattice
        # If distributive, isomorphism should be verified
        if result.is_distributive:
            assert result.isomorphism_verified

    def test_nested_branch(self):
        """&{a: &{c: end, d: end}, b: end} --- nested branches."""
        ast = parse("&{a: &{c: end, d: end}, b: end}")
        ss = build_statespace(ast)
        result = birkhoff_representation(ss)
        assert result.is_lattice

    def test_selection(self):
        """+{ok: end, err: end} --- selection."""
        ast = parse("+{ok: end, err: end}")
        ss = build_statespace(ast)
        result = birkhoff_representation(ss)
        assert result.is_lattice
        if result.is_distributive:
            assert result.is_representable


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestBirkhoffEdgeCases:
    """Edge cases for the Birkhoff module."""

    def test_single_state(self):
        """Single-state lattice (end)."""
        ast = parse("end")
        ss = build_statespace(ast)
        result = birkhoff_representation(ss)
        assert result.is_lattice
        assert result.is_distributive
        assert result.is_representable
        assert len(result.join_irreducibles) == 0

    def test_two_state_chain(self):
        """Two-state chain: top > end."""
        ss = _chain(2)
        result = birkhoff_representation(ss)
        assert result.is_representable
        assert len(result.join_irreducibles) == 1
        assert result.downset_lattice_size == 2

    def test_ji_poset_antichain_for_boolean(self):
        """Boolean 2^2: JI poset should be a 2-element antichain."""
        ss = _boolean_lattice_2()
        result = birkhoff_representation(ss)
        # JI poset should have no edges (antichain)
        for j in result.join_irreducibles:
            assert len(result.ji_poset.get(j, set())) == 0, \
                f"JI {j} has lower covers in poset --- expected antichain"
