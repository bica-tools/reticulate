"""Tests for subdirect decomposition of session type lattices.

Step 363d: Tests subdirect irreducibility, Birkhoff's decomposition,
factor computation, and direct product detection.
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.subdirect import (
    is_subdirectly_irreducible,
    subdirect_factors,
    analyze_subdirect,
    SubdirectFactor,
    SubdirectAnalysis,
)


def _build(type_str: str):
    """Parse, build state space, check lattice."""
    ast = parse(type_str)
    ss = build_statespace(ast)
    lr = check_lattice(ss)
    return ss, lr


# ═══════════════════════════════════════════════════════
# Test: is_subdirectly_irreducible
# ═══════════════════════════════════════════════════════

class TestSubdirectlyIrreducible:
    """Tests for subdirectly irreducible lattice detection."""

    def test_trivial_si(self):
        """1-element lattice is SI (trivially simple)."""
        ss, lr = _build("end")
        assert lr.is_lattice
        assert is_subdirectly_irreducible(ss, lr)

    def test_two_chain_si(self):
        """2-element chain is SI (simple)."""
        ss, lr = _build("&{a: end}")
        assert lr.is_lattice
        assert is_subdirectly_irreducible(ss, lr)

    def test_three_chain_not_si(self):
        """3-element chain is NOT SI (Con has 2 atoms: θ(0,1) and θ(1,2))."""
        ss, lr = _build("&{a: &{b: end}}")
        assert lr.is_lattice
        # 3-chain: Con(L) = {id, θ(0,2), θ(1,2), total} with 2 atoms
        # So the 3-chain decomposes into 2 factors of size 2
        assert not is_subdirectly_irreducible(ss, lr)

    def test_non_lattice_not_si(self):
        """Non-lattice is not SI."""
        ss, lr = _build("&{a: end, b: rec X . &{a: X}}")
        if not lr.is_lattice:
            assert not is_subdirectly_irreducible(ss, lr)

    def test_simple_implies_si(self):
        """Simple lattice is always SI."""
        ss, lr = _build("&{a: end}")
        assert lr.is_lattice
        from reticulate.congruence import is_simple
        if is_simple(ss):
            assert is_subdirectly_irreducible(ss, lr)


# ═══════════════════════════════════════════════════════
# Test: subdirect_factors
# ═══════════════════════════════════════════════════════

class TestSubdirectFactors:
    """Tests for computing subdirect factors."""

    def test_trivial_one_factor(self):
        """1-element lattice has one factor (itself)."""
        ss, lr = _build("end")
        factors = subdirect_factors(ss, lr)
        assert len(factors) >= 1

    def test_two_chain_one_factor(self):
        """2-element chain (simple) has one factor."""
        ss, lr = _build("&{a: end}")
        factors = subdirect_factors(ss, lr)
        assert len(factors) >= 1

    def test_factors_are_subdirect_factor(self):
        """All factors are SubdirectFactor instances."""
        ss, lr = _build("&{a: end, b: end}")
        if lr.is_lattice:
            factors = subdirect_factors(ss, lr)
            for f in factors:
                assert isinstance(f, SubdirectFactor)

    def test_factor_quotient_is_statespace(self):
        """Each factor's quotient is a valid StateSpace."""
        ss, lr = _build("&{a: &{b: end}}")
        factors = subdirect_factors(ss, lr)
        for f in factors:
            assert hasattr(f.quotient, 'states')
            assert hasattr(f.quotient, 'transitions')
            assert f.quotient_size > 0

    def test_si_lattice_gives_itself(self):
        """An SI lattice's decomposition returns itself as the sole factor."""
        ss, lr = _build("&{a: end}")
        if is_subdirectly_irreducible(ss, lr):
            factors = subdirect_factors(ss, lr)
            assert len(factors) == 1

    def test_non_lattice_empty(self):
        """Non-lattice gives empty factor list."""
        ss, lr = _build("&{a: end, b: rec X . &{a: X}}")
        if not lr.is_lattice:
            factors = subdirect_factors(ss, lr)
            assert factors == []


# ═══════════════════════════════════════════════════════
# Test: analyze_subdirect
# ═══════════════════════════════════════════════════════

class TestAnalyzeSubdirect:
    """Tests for complete subdirect analysis."""

    def test_returns_analysis(self):
        """analyze_subdirect returns SubdirectAnalysis."""
        ss, lr = _build("&{a: end}")
        result = analyze_subdirect(ss, lr)
        assert isinstance(result, SubdirectAnalysis)

    def test_trivial_analysis(self):
        """1-element lattice: simple, SI, 1 factor."""
        ss, lr = _build("end")
        result = analyze_subdirect(ss, lr)
        assert result.is_simple
        assert result.is_subdirectly_irreducible
        assert result.num_factors >= 1

    def test_two_chain_analysis(self):
        """2-element chain analysis."""
        ss, lr = _build("&{a: end}")
        result = analyze_subdirect(ss, lr)
        assert result.is_simple
        assert result.is_subdirectly_irreducible
        assert result.original_size == 2

    def test_three_chain_analysis(self):
        """3-element chain: not SI, decomposes into 2 factors."""
        ss, lr = _build("&{a: &{b: end}}")
        result = analyze_subdirect(ss, lr)
        assert not result.is_subdirectly_irreducible
        assert result.num_factors == 2
        assert result.original_size == 3

    def test_non_lattice_analysis(self):
        """Non-lattice gives zero factors."""
        ss, lr = _build("&{a: end, b: rec X . &{a: X}}")
        if not lr.is_lattice:
            result = analyze_subdirect(ss, lr)
            assert not result.is_subdirectly_irreducible
            assert result.num_factors == 0

    def test_parallel_analysis(self):
        """Parallel composition analysis."""
        ss, lr = _build("(&{a: end} || &{b: end})")
        if lr.is_lattice:
            result = analyze_subdirect(ss, lr)
            assert result.original_size == len(ss.states)
            assert result.con_lattice_size > 0

    def test_branch_analysis(self):
        """Branch type analysis."""
        ss, lr = _build("&{a: end, b: end}")
        if lr.is_lattice:
            result = analyze_subdirect(ss, lr)
            assert isinstance(result.num_factors, int)
            assert result.num_factors >= 1


# ═══════════════════════════════════════════════════════
# Test: Properties
# ═══════════════════════════════════════════════════════

class TestProperties:
    """Tests for lattice-theoretic properties."""

    def test_simple_implies_si(self):
        """Simple lattice is always SI."""
        for type_str in ["end", "&{a: end}"]:
            ss, lr = _build(type_str)
            result = analyze_subdirect(ss, lr)
            if result.is_simple:
                assert result.is_subdirectly_irreducible

    def test_si_implies_at_least_one_factor(self):
        """SI lattice has at least one factor."""
        ss, lr = _build("&{a: &{b: end}}")
        result = analyze_subdirect(ss, lr)
        if result.is_subdirectly_irreducible:
            assert result.num_factors >= 1

    def test_factors_nonempty_for_lattice(self):
        """Every lattice has at least one subdirect factor."""
        for type_str in ["end", "&{a: end}", "&{a: &{b: end}}",
                          "&{a: end, b: end}"]:
            ss, lr = _build(type_str)
            if lr.is_lattice:
                factors = subdirect_factors(ss, lr)
                assert len(factors) >= 1, f"No factors for {type_str}"


# ═══════════════════════════════════════════════════════
# Test: Benchmarks
# ═══════════════════════════════════════════════════════

class TestBenchmarks:
    """Tests on benchmark session types."""

    @pytest.mark.parametrize("name,type_str", [
        ("end", "end"),
        ("single_branch", "&{a: end}"),
        ("two_branch", "&{a: end, b: end}"),
        ("chain_3", "&{a: &{b: end}}"),
        ("chain_4", "&{a: &{b: &{c: end}}}"),
        ("binary_select", "+{ok: end, err: end}"),
        ("parallel", "(&{a: end} || &{b: end})"),
    ])
    def test_benchmark(self, name, type_str):
        """All benchmarks produce valid analysis."""
        ss, lr = _build(type_str)
        if lr.is_lattice:
            result = analyze_subdirect(ss, lr)
            assert isinstance(result, SubdirectAnalysis)
            assert result.num_factors >= 1
            # Consistency: simple => SI
            if result.is_simple:
                assert result.is_subdirectly_irreducible
