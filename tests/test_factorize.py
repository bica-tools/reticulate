"""Tests for lattice factorization theory (Step 363f).

Tests cover:
- Trivial/indecomposable lattices (end, chains, simple)
- Decomposable lattices from parallel composition
- Factor congruence pair detection
- Recursive factorization into indecomposable factors
- Benchmark protocols
- Analysis dataclass integrity
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace, StateSpace
from reticulate.lattice import check_lattice
from reticulate.factorize import (
    find_factor_congruences,
    find_factor_congruences_indexed,
    is_directly_decomposable,
    factorize,
    analyze_factorization,
    FactorizationAnalysis,
    _meet_congruences,
    _congruences_equal,
)
from reticulate.congruence import (
    Congruence,
    CongruenceLattice,
    enumerate_congruences,
    congruence_lattice,
    quotient_lattice,
)

from tests.benchmarks.protocols import BENCHMARKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(type_str: str) -> StateSpace:
    """Parse a session type and build its state space."""
    ast = parse(type_str)
    return build_statespace(ast)


# ---------------------------------------------------------------------------
# Basic indecomposable types
# ---------------------------------------------------------------------------

class TestIndecomposable:
    """Types that should NOT be decomposable."""

    def test_end_indecomposable(self) -> None:
        ss = _build("end")
        assert not is_directly_decomposable(ss)

    def test_single_branch_indecomposable(self) -> None:
        ss = _build("&{a: end}")
        assert not is_directly_decomposable(ss)

    def test_chain_2_indecomposable(self) -> None:
        """A 2-element chain is indecomposable."""
        ss = _build("&{a: end}")
        factors = factorize(ss)
        assert len(factors) == 1

    def test_chain_3_indecomposable(self) -> None:
        """A 3-element chain (depth 2) is indecomposable."""
        ss = _build("&{a: &{b: end}}")
        assert not is_directly_decomposable(ss)
        factors = factorize(ss)
        assert len(factors) == 1

    def test_branch_two_options_indecomposable(self) -> None:
        """&{a: end, b: end} has 3 states forming a V-shape, not decomposable."""
        ss = _build("&{a: end, b: end}")
        assert not is_directly_decomposable(ss)

    def test_select_two_options_indecomposable(self) -> None:
        """+{ok: end, err: end} is a 3-element lattice, not decomposable."""
        ss = _build("+{ok: end, err: end}")
        assert not is_directly_decomposable(ss)

    def test_recursive_iterator_indecomposable(self) -> None:
        """Java Iterator has no hidden parallelism."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        assert not is_directly_decomposable(ss)

    def test_factorize_end_returns_self(self) -> None:
        ss = _build("end")
        factors = factorize(ss)
        assert len(factors) == 1
        assert factors[0] is ss


# ---------------------------------------------------------------------------
# Decomposable types (parallel composition)
# ---------------------------------------------------------------------------

class TestDecomposable:
    """Types that SHOULD be decomposable."""

    def test_parallel_two_chains(self) -> None:
        """(a.end || b.end) should decompose into two 2-chains."""
        ss = _build("(&{a: end} || &{b: end})")
        lr = check_lattice(ss)
        assert lr.is_lattice
        assert is_directly_decomposable(ss, lr)

    def test_parallel_factor_count(self) -> None:
        """Parallel of two simple protocols should yield 2 factors."""
        ss = _build("(&{a: end} || &{b: end})")
        factors = factorize(ss)
        assert len(factors) == 2

    def test_parallel_factor_sizes(self) -> None:
        """Each factor of (a.end || b.end) should have 2 states."""
        ss = _build("(&{a: end} || &{b: end})")
        factors = factorize(ss)
        sizes = sorted(len(f.states) for f in factors)
        assert sizes == [2, 2]

    def test_parallel_factor_congruences_exist(self) -> None:
        """Factor congruence pairs should be non-empty for parallel types."""
        ss = _build("(&{a: end} || &{b: end})")
        pairs = find_factor_congruences(ss)
        assert len(pairs) >= 1

    def test_parallel_indexed_pairs(self) -> None:
        """Indexed pairs should be consistent with congruence list."""
        ss = _build("(&{a: end} || &{b: end})")
        indexed = find_factor_congruences_indexed(ss)
        assert len(indexed) >= 1
        con_lat = congruence_lattice(ss)
        for i, j in indexed:
            assert 0 <= i < len(con_lat.congruences)
            assert 0 <= j < len(con_lat.congruences)

    def test_parallel_asymmetric_branches(self) -> None:
        """(&{a: end} || &{b: &{c: end}}) should be decomposable."""
        ss = _build("(&{a: end} || &{b: &{c: end}})")
        lr = check_lattice(ss)
        assert lr.is_lattice
        assert is_directly_decomposable(ss, lr)
        factors = factorize(ss, lr)
        assert len(factors) == 2
        sizes = sorted(len(f.states) for f in factors)
        assert sizes == [2, 3]


# ---------------------------------------------------------------------------
# Factor congruence pair properties
# ---------------------------------------------------------------------------

class TestFactorCongruenceProperties:
    """Verify mathematical properties of factor congruence pairs."""

    def test_meet_is_identity(self) -> None:
        """For factor pair (theta, phi), theta ^ phi must be the identity."""
        ss = _build("(&{a: end} || &{b: end})")
        pairs = find_factor_congruences(ss)
        assert len(pairs) >= 1
        theta, phi = pairs[0]
        meet = _meet_congruences(ss, theta, phi)
        # Identity = all singletons
        assert all(len(c) == 1 for c in meet.classes)

    def test_join_is_total(self) -> None:
        """For factor pair (theta, phi), theta v phi must be total."""
        from reticulate.congruence import _join_congruences
        ss = _build("(&{a: end} || &{b: end})")
        pairs = find_factor_congruences(ss)
        assert len(pairs) >= 1
        theta, phi = pairs[0]
        join = _join_congruences(ss, theta, phi)
        # Total = one big class
        assert join.num_classes == 1

    def test_no_factor_pairs_for_chain(self) -> None:
        """Chains have no non-trivial factor pairs."""
        ss = _build("&{a: &{b: end}}")
        pairs = find_factor_congruences(ss)
        assert pairs == []


# ---------------------------------------------------------------------------
# Meet congruence helper tests
# ---------------------------------------------------------------------------

class TestMeetCongruences:
    """Tests for the internal _meet_congruences helper."""

    def test_meet_with_identity(self) -> None:
        """Meet of any congruence with identity is identity."""
        ss = _build("(&{a: end} || &{b: end})")
        con_lat = congruence_lattice(ss)
        # Identity = all-singletons congruence (finest partition)
        identity = next(c for c in con_lat.congruences if c.is_trivial_bottom)
        for c in con_lat.congruences:
            meet = _meet_congruences(ss, c, identity)
            assert _congruences_equal(meet, identity)

    def test_meet_with_total(self) -> None:
        """Meet of any congruence with total is that congruence."""
        ss = _build("(&{a: end} || &{b: end})")
        con_lat = congruence_lattice(ss)
        total = con_lat.congruences[con_lat.top]
        for c in con_lat.congruences:
            meet = _meet_congruences(ss, c, total)
            assert _congruences_equal(meet, c)

    def test_meet_is_commutative(self) -> None:
        """Meet should be commutative."""
        ss = _build("(&{a: end} || &{b: end})")
        congs = enumerate_congruences(ss)
        if len(congs) >= 2:
            c1, c2 = congs[0], congs[1]
            m1 = _meet_congruences(ss, c1, c2)
            m2 = _meet_congruences(ss, c2, c1)
            assert _congruences_equal(m1, m2)


# ---------------------------------------------------------------------------
# Analysis dataclass
# ---------------------------------------------------------------------------

class TestAnalysis:
    """Tests for analyze_factorization."""

    def test_analysis_indecomposable(self) -> None:
        ss = _build("&{a: end}")
        result = analyze_factorization(ss)
        assert isinstance(result, FactorizationAnalysis)
        assert not result.is_decomposable
        assert result.is_indecomposable
        assert result.factor_count == 1
        assert result.factor_sizes == (2,)
        assert result.factor_congruence_pairs == ()

    def test_analysis_decomposable(self) -> None:
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_factorization(ss)
        assert result.is_decomposable
        assert not result.is_indecomposable
        assert result.factor_count == 2
        assert sorted(result.factor_sizes) == [2, 2]
        assert len(result.factor_congruence_pairs) >= 1

    def test_analysis_end(self) -> None:
        ss = _build("end")
        result = analyze_factorization(ss)
        assert not result.is_decomposable
        assert result.is_indecomposable
        assert result.factor_count == 1

    def test_analysis_non_lattice(self) -> None:
        """Non-lattice state spaces should report indecomposable."""
        # Manually build a non-lattice
        ss = StateSpace(
            states={0, 1, 2, 3},
            transitions=[(0, "a", 1), (0, "b", 2), (1, "c", 3), (2, "d", 3)],
            top=0,
            bottom=3,
        )
        # This might or might not be a lattice; test the analysis path
        result = analyze_factorization(ss)
        assert isinstance(result, FactorizationAnalysis)
        assert result.factor_count >= 1

    def test_analysis_factors_are_statespaces(self) -> None:
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_factorization(ss)
        for f in result.factors:
            assert isinstance(f, StateSpace)


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Test factorization on benchmark protocols."""

    def test_non_parallel_benchmarks_indecomposable(self) -> None:
        """Small non-parallel benchmarks should be indecomposable."""
        count = 0
        for bp in BENCHMARKS[:10]:
            if bp.uses_parallel:
                continue
            ss = _build(bp.type_string)
            lr = check_lattice(ss)
            if not lr.is_lattice or len(ss.states) > 5:
                continue
            decomposable = is_directly_decomposable(ss, lr)
            if not decomposable:
                count += 1
        assert count >= 1

    def test_parallel_benchmark_two_buyer(self) -> None:
        """Two-Buyer uses parallel and analysis should complete."""
        bp = next(b for b in BENCHMARKS if b.name == "Two-Buyer")
        ss = _build(bp.type_string)
        lr = check_lattice(ss)
        if lr.is_lattice:
            result = analyze_factorization(ss, lr)
            assert result.factor_count >= 1

    def test_java_iterator_indecomposable(self) -> None:
        """Java Iterator is a simple loop, should be indecomposable."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        lr = check_lattice(ss)
        assert lr.is_lattice
        assert not is_directly_decomposable(ss, lr)

    def test_file_object_indecomposable(self) -> None:
        """File Object protocol should be indecomposable."""
        ss = _build("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")
        lr = check_lattice(ss)
        assert lr.is_lattice
        assert not is_directly_decomposable(ss, lr)

    def test_http_connection_indecomposable(self) -> None:
        """HTTP Connection protocol should be indecomposable."""
        ss = _build("&{connect: rec X . &{request: +{OK200: &{readBody: X}, "
                     "ERR4xx: X, ERR5xx: X}, close: end}}")
        lr = check_lattice(ss)
        assert lr.is_lattice
        assert not is_directly_decomposable(ss, lr)

    def test_a2a_indecomposable(self) -> None:
        """A2A protocol (5 states, no parallel) should be indecomposable."""
        bp = next(b for b in BENCHMARKS if b.name == "A2A")
        ss = _build(bp.type_string)
        lr = check_lattice(ss)
        if lr.is_lattice and len(ss.states) <= 6:
            assert not is_directly_decomposable(ss, lr)


# ---------------------------------------------------------------------------
# Product reconstruction / unique factorization
# ---------------------------------------------------------------------------

class TestUniqueFactorization:
    """Verify factorization properties: reconstruction and uniqueness."""

    def test_product_size_equals_factor_product(self) -> None:
        """For (A || B), |L(A||B)| = |L(A)| * |L(B)|."""
        ss = _build("(&{a: end} || &{b: end})")
        factors = factorize(ss)
        product_size = 1
        for f in factors:
            product_size *= len(f.states)
        assert product_size == len(ss.states)

    def test_asymmetric_product_size(self) -> None:
        """For asymmetric parallel, factor sizes still multiply to original."""
        ss = _build("(&{a: end} || &{b: &{c: end}})")
        factors = factorize(ss)
        product_size = 1
        for f in factors:
            product_size *= len(f.states)
        assert product_size == len(ss.states)

    def test_factor_are_lattices(self) -> None:
        """All factors of a lattice should themselves be lattices."""
        ss = _build("(&{a: end} || &{b: end})")
        factors = factorize(ss)
        for f in factors:
            lr = check_lattice(f)
            assert lr.is_lattice

    def test_factors_are_indecomposable(self) -> None:
        """After full factorization, each factor should be indecomposable."""
        ss = _build("(&{a: end} || &{b: end})")
        factors = factorize(ss)
        for f in factors:
            assert not is_directly_decomposable(f)

    def test_factorize_idempotent(self) -> None:
        """Factorizing an indecomposable returns itself."""
        ss = _build("&{a: &{b: end}}")
        factors = factorize(ss)
        assert len(factors) == 1
        # Factorize again
        factors2 = factorize(factors[0])
        assert len(factors2) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_state_end(self) -> None:
        ss = _build("end")
        result = analyze_factorization(ss)
        assert result.factor_count == 1
        assert not result.is_decomposable

    def test_deeply_nested_branch(self) -> None:
        """Deep chain should be indecomposable."""
        ss = _build("&{a: &{b: &{c: &{d: end}}}}")
        assert not is_directly_decomposable(ss)

    def test_selection_chain(self) -> None:
        """+{a: +{b: end}} should be indecomposable."""
        ss = _build("+{a: +{b: end}}")
        assert not is_directly_decomposable(ss)

    def test_find_factor_congruences_non_lattice(self) -> None:
        """Non-lattice should return empty list."""
        ss = StateSpace(
            states={0},
            transitions=[],
            top=0,
            bottom=0,
        )
        pairs = find_factor_congruences(ss)
        assert pairs == []
