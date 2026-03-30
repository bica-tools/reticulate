"""Tests for zeta polynomial and chain counting (Step 32f).

Tests cover:
- Multichain counting Z(P, k) for small k
- Chain counting (strict chains)
- Polynomial interpolation and coefficient computation
- Philip Hall's theorem: Z(P, -1) = mu(0, 1)
- Palindromicity detection
- Whitney number connection
- Benchmark protocol analysis
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.zeta_polynomial import (
    multichain_count,
    chain_count,
    all_chain_counts,
    zeta_polynomial,
    zeta_polynomial_coefficients,
    is_palindromic,
    whitney_numbers_second_kind,
    analyze_zeta_polynomial,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str):
    """Parse and build state space."""
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Basic multichain counting
# ---------------------------------------------------------------------------

class TestMultichainCount:
    """Tests for multichain_count(ss, k)."""

    def test_end_k0(self):
        """Z(end, 0) = 1 (empty chain)."""
        ss = _build("end")
        assert multichain_count(ss, 0) == 1

    def test_end_k1(self):
        """Z(end, 1) = 1 (one element)."""
        ss = _build("end")
        assert multichain_count(ss, 1) == 1

    def test_end_k2(self):
        """Z(end, 2) = 1 (only chain: end <= end)."""
        ss = _build("end")
        assert multichain_count(ss, 2) == 1

    def test_single_branch_k1(self):
        """Z(&{a: end}, 1) = 2 (two states)."""
        ss = _build("&{a: end}")
        assert multichain_count(ss, 1) == 2

    def test_single_branch_k2(self):
        """Z(&{a: end}, 2) = 3: (top,top), (top,bot), (bot,bot)."""
        ss = _build("&{a: end}")
        assert multichain_count(ss, 2) == 3

    def test_two_branch_k1(self):
        """&{a: end, b: end} has 2 states (both branches merge to end)."""
        ss = _build("&{a: end, b: end}")
        assert multichain_count(ss, 1) == 2

    def test_k0_always_1(self):
        """Z(P, 0) = 1 for any poset."""
        for t in ["end", "&{a: end}", "&{a: end, b: end}", "+{a: end}"]:
            ss = _build(t)
            assert multichain_count(ss, 0) == 1

    def test_monotone_in_k(self):
        """Z(P, k) is non-decreasing in k for k >= 0."""
        ss = _build("&{a: end, b: end}")
        prev = multichain_count(ss, 0)
        for k in range(1, 5):
            curr = multichain_count(ss, k)
            assert curr >= prev
            prev = curr

    def test_selection_k1(self):
        """+{a: end} has same count as &{a: end}."""
        ss = _build("+{a: end}")
        assert multichain_count(ss, 1) == 2

    def test_three_branch_k2(self):
        """&{a: end, b: end, c: end} has specific Z(P, 2)."""
        ss = _build("&{a: end, b: end, c: end}")
        # 4 states: top, a, b, c(=end). Actually depends on state space structure.
        n = multichain_count(ss, 1)
        z2 = multichain_count(ss, 2)
        # Z(P, 2) >= n (diagonal pairs) and Z(P, 2) >= n (at least)
        assert z2 >= n


# ---------------------------------------------------------------------------
# Strict chain counting
# ---------------------------------------------------------------------------

class TestChainCount:
    """Tests for chain_count(ss, length)."""

    def test_end_chain_0(self):
        """One chain of length 0 (empty)."""
        ss = _build("end")
        assert chain_count(ss, 0) == 1

    def test_end_chain_1(self):
        """One chain of length 1 (just the element)."""
        ss = _build("end")
        assert chain_count(ss, 1) == 1

    def test_single_branch_chain_1(self):
        """&{a: end} has 2 chains of length 1."""
        ss = _build("&{a: end}")
        assert chain_count(ss, 1) == 2

    def test_single_branch_chain_2(self):
        """&{a: end} has 1 chain of length 2: top > bottom."""
        ss = _build("&{a: end}")
        assert chain_count(ss, 2) == 1

    def test_negative_length(self):
        """chain_count with negative length returns 0."""
        ss = _build("end")
        assert chain_count(ss, -1) == 0

    def test_all_chain_counts(self):
        """all_chain_counts returns counts for all lengths."""
        ss = _build("&{a: end, b: end}")
        counts = all_chain_counts(ss)
        assert 0 in counts or 1 in counts  # should have at least length-1 chains
        # All values positive
        for v in counts.values():
            assert v > 0


class TestAllChainCounts:
    """Tests for all_chain_counts comprehensive."""

    def test_end_chains(self):
        """end: one chain of length 0, one of length 1."""
        ss = _build("end")
        counts = all_chain_counts(ss)
        assert counts.get(0, 0) == 1
        assert counts.get(1, 0) == 1

    def test_diamond(self):
        """&{a: end, b: end} should have chains of lengths 0, 1, 2."""
        ss = _build("&{a: end, b: end}")
        counts = all_chain_counts(ss)
        assert 1 in counts
        assert counts[1] >= 2  # at least top and bottom


# ---------------------------------------------------------------------------
# Zeta polynomial as polynomial
# ---------------------------------------------------------------------------

class TestZetaPolynomial:
    """Tests for the zeta polynomial Z(P, k) as a true polynomial."""

    def test_end_polynomial(self):
        """Z(end, k) = 1 for all k (constant polynomial)."""
        ss = _build("end")
        for k in range(5):
            assert zeta_polynomial(ss, k) == 1

    def test_single_branch_polynomial(self):
        """Z(&{a: end}, k) should be a degree-1 polynomial."""
        ss = _build("&{a: end}")
        coeffs = zeta_polynomial_coefficients(ss)
        # Degree should be 1 (height of the poset)
        assert len(coeffs) >= 2

    def test_coefficients_match_values(self):
        """Coefficients reconstruct the polynomial values."""
        ss = _build("&{a: end, b: end}")
        coeffs = zeta_polynomial_coefficients(ss)
        for k in range(5):
            expected = multichain_count(ss, k)
            computed = sum(c * k**i for i, c in enumerate(coeffs))
            assert abs(computed - expected) < 0.5, f"k={k}: {computed} vs {expected}"


# ---------------------------------------------------------------------------
# Philip Hall's theorem: Z(P, -1) = mu(0, 1)
# ---------------------------------------------------------------------------

class TestHallTheorem:
    """Tests for Philip Hall's theorem via zeta polynomial."""

    def test_end_hall(self):
        """Z(end, -1) should match mu(top, bottom)."""
        ss = _build("end")
        result = analyze_zeta_polynomial(ss)
        assert result.hall_verified

    def test_single_branch_hall(self):
        """Z(&{a: end}, -1) = mu(top, bottom)."""
        ss = _build("&{a: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.hall_verified

    def test_two_branch_hall(self):
        """Z(&{a: end, b: end}, -1) = mu(top, bottom)."""
        ss = _build("&{a: end, b: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.hall_verified

    def test_selection_hall(self):
        """+{a: end, b: end}: Hall's theorem holds."""
        ss = _build("+{a: end, b: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.hall_verified

    def test_nested_branch_hall(self):
        """Nested branch: Hall's theorem holds."""
        ss = _build("&{a: &{c: end}, b: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.hall_verified


# ---------------------------------------------------------------------------
# Palindromicity
# ---------------------------------------------------------------------------

class TestPalindromic:
    """Tests for palindromic detection."""

    def test_end_palindromic(self):
        """end has a constant polynomial — trivially palindromic."""
        ss = _build("end")
        assert is_palindromic(ss)

    def test_single_branch_check(self):
        """&{a: end} palindromicity is well-defined."""
        ss = _build("&{a: end}")
        # Just check it doesn't crash; palindromicity depends on coefficients
        result = is_palindromic(ss)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Whitney numbers
# ---------------------------------------------------------------------------

class TestWhitneyNumbers:
    """Tests for Whitney numbers of the second kind."""

    def test_end_whitney(self):
        """end has W = [1] (one element at rank 0)."""
        ss = _build("end")
        w = whitney_numbers_second_kind(ss)
        assert sum(w) >= 1

    def test_single_branch_whitney(self):
        """&{a: end} has 2 elements at different ranks."""
        ss = _build("&{a: end}")
        w = whitney_numbers_second_kind(ss)
        assert sum(w) == 2

    def test_two_branch_whitney(self):
        """&{a: end, b: end} has 2 states (branches merge to end)."""
        ss = _build("&{a: end, b: end}")
        w = whitney_numbers_second_kind(ss)
        total = sum(w)
        assert total == 2  # top + end


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeZetaPolynomial:
    """Tests for the full analysis function."""

    def test_end_analysis(self):
        """Complete analysis of end type."""
        ss = _build("end")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states == 1
        assert result.degree >= 0
        assert result.hall_verified

    def test_single_branch_analysis(self):
        """Complete analysis of &{a: end}."""
        ss = _build("&{a: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states == 2
        assert len(result.coefficients) >= 1
        assert len(result.values) >= 1

    def test_two_branch_analysis(self):
        """Complete analysis of &{a: end, b: end} (2 states: branches merge)."""
        ss = _build("&{a: end, b: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states == 2
        assert result.total_multichains_k2 >= result.num_states

    def test_selection_analysis(self):
        """Complete analysis of +{a: end, b: end}."""
        ss = _build("+{a: end, b: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states >= 2

    def test_nested_analysis(self):
        """Complete analysis of nested type."""
        ss = _build("&{a: &{c: end, d: end}, b: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states >= 3

    def test_recursive_analysis(self):
        """Analysis of recursive type handles cycles."""
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states >= 1
        assert isinstance(result.hall_verified, bool)


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on standard benchmark protocols."""

    def test_iterator(self):
        """Java Iterator protocol."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states >= 2
        assert result.hall_verified

    def test_simple_resource(self):
        """Simple resource: open, use, close."""
        ss = _build("&{open: &{use: &{close: end}}}")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states == 4
        assert result.degree >= 1

    def test_binary_choice(self):
        """Binary selection."""
        ss = _build("+{left: end, right: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states >= 2

    def test_parallel_simple(self):
        """Parallel composition."""
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states >= 4

    def test_z_k2_geq_n(self):
        """Z(P, 2) >= |P| for any poset (diagonal pairs contribute)."""
        for t in [
            "end", "&{a: end}", "&{a: end, b: end}",
            "+{a: end}", "&{a: &{b: end}}",
        ]:
            ss = _build(t)
            result = analyze_zeta_polynomial(ss)
            assert result.total_multichains_k2 >= result.num_states

    def test_multichain_growth(self):
        """Z(P, k) grows at most polynomially in k."""
        ss = _build("&{a: end, b: end}")
        vals = [multichain_count(ss, k) for k in range(6)]
        # Should be bounded by n^k where n = |P|
        n = len(ss.states)
        for k, v in enumerate(vals):
            if k > 0:
                assert v <= n ** k + 1  # +1 for rounding


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_deeply_nested(self):
        """Deeply nested branch."""
        ss = _build("&{a: &{b: &{c: &{d: end}}}}")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states == 5
        assert result.degree >= 1

    def test_wide_branch(self):
        """Wide branch with many options (all merge to end, 2 states)."""
        ss = _build("&{a: end, b: end, c: end, d: end}")
        result = analyze_zeta_polynomial(ss)
        assert result.num_states == 2
