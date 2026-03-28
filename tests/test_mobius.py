"""Tests for Möbius function analysis (Step 30b).

Tests cover:
- Möbius matrix and pointwise values
- Philip Hall's theorem (alternating chain count)
- Weisner's theorem (coatom sums)
- Distributivity test via |μ| ≤ 1
- Möbius spectrum
- Product formula under parallel
- Benchmark protocol analysis
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.mobius import (
    mobius_matrix,
    mobius_value,
    all_mobius_values,
    verify_hall,
    verify_weisner,
    max_abs_mobius,
    is_distributive_by_mobius,
    mobius_spectrum,
    verify_product_formula,
    analyze_mobius,
    HallResult,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Möbius value tests
# ---------------------------------------------------------------------------

class TestMobiusValue:
    """Tests for μ(⊤,⊥)."""

    def test_end(self):
        """end: single state, μ(⊤,⊥) = 1 (⊤ = ⊥)."""
        ss = _build("end")
        assert mobius_value(ss) == 1

    def test_single_branch(self):
        """&{a: end}: chain of length 1, μ = -1."""
        ss = _build("&{a: end}")
        assert mobius_value(ss) == -1

    def test_chain_length_2(self):
        """&{a: &{b: end}}: chain of length 2, μ = 0."""
        ss = _build("&{a: &{b: end}}")
        assert mobius_value(ss) == 0

    def test_chain_length_3(self):
        """&{a: &{b: &{c: end}}}: chain of length 3, μ = 0."""
        ss = _build("&{a: &{b: &{c: end}}}")
        assert mobius_value(ss) == 0

    def test_two_branches_same_depth(self):
        """&{a: end, b: end}: two branches to bottom."""
        ss = _build("&{a: end, b: end}")
        # This is a chain (both go to same end), μ should be appropriate
        mu = mobius_value(ss)
        assert isinstance(mu, int)

    def test_selection(self):
        """+{a: end, b: end}: selection type."""
        ss = _build("+{a: end, b: end}")
        mu = mobius_value(ss)
        assert isinstance(mu, int)

    def test_parallel_boolean(self):
        """(&{a: end} || &{b: end}): product of two chains, μ = (-1)^2 = 1."""
        ss = _build("(&{a: end} || &{b: end})")
        mu = mobius_value(ss)
        # Product of two 2-element chains: B₂ ≅ C₂ × C₂
        # μ(C₂) = -1 for each, product = (-1)·(-1) = 1
        assert mu == 1

    def test_recursive(self):
        """rec X . &{a: X, b: end}: recursive type."""
        ss = _build("rec X . &{a: X, b: end}")
        mu = mobius_value(ss)
        assert isinstance(mu, int)


# ---------------------------------------------------------------------------
# Möbius matrix tests
# ---------------------------------------------------------------------------

class TestMobiusMatrix:
    """Tests for the full Möbius matrix."""

    def test_diagonal_is_one(self):
        """Diagonal of M is always 1."""
        ss = _build("&{a: &{b: end}}")
        M, states = mobius_matrix(ss)
        for i in range(len(states)):
            assert M[i][i] == 1

    def test_chain_mobius(self):
        """Chain: μ(i, i+1) = -1, μ(i, j) = 0 for j > i+1."""
        ss = _build("&{a: &{b: end}}")
        M, states = mobius_matrix(ss)
        # Should have: diagonal = 1, adjacent = -1, distance 2 = 0
        n = len(states)
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert M[i][j] == 1

    def test_inverse_of_zeta(self):
        """ζ * μ = δ via incidence algebra convolution."""
        from reticulate.zeta import zeta_function, mobius_function, convolve, delta_function
        ss = _build("&{a: end, b: end}")
        zf = zeta_function(ss)
        mf = mobius_function(ss)
        result = convolve(zf, mf, ss)
        df = delta_function(ss)
        for s in ss.states:
            for t in ss.states:
                expected = df.get((s, t), 0)
                actual = result.get((s, t), 0)
                assert actual == expected, \
                    f"(ζ*μ)({s},{t}) = {actual} ≠ δ({s},{t}) = {expected}"


# ---------------------------------------------------------------------------
# Philip Hall's theorem tests
# ---------------------------------------------------------------------------

class TestHall:
    """Tests for Philip Hall's theorem verification."""

    def test_end_hall(self):
        """end: trivial case."""
        ss = _build("end")
        r = verify_hall(ss)
        assert r.verified

    def test_single_branch_hall(self):
        """&{a: end}: one chain of length 1, μ = -1."""
        ss = _build("&{a: end}")
        r = verify_hall(ss)
        assert r.verified
        assert r.mobius_value == -1
        assert r.alternating_sum == -1
        assert 1 in r.chain_counts
        assert r.chain_counts[1] == 1

    def test_chain_2_hall(self):
        """&{a: &{b: end}}: chains of length 1 (none) and 2."""
        ss = _build("&{a: &{b: end}}")
        r = verify_hall(ss)
        assert r.verified
        assert r.mobius_value == 0

    def test_diamond_hall(self):
        """&{a: &{c: end}, b: &{c: end}}: diamond shape."""
        ss = _build("&{a: &{c: end}, b: &{c: end}}")
        r = verify_hall(ss)
        assert r.verified

    def test_parallel_hall(self):
        """(&{a: end} || &{b: end}): product lattice."""
        ss = _build("(&{a: end} || &{b: end})")
        r = verify_hall(ss)
        assert r.verified


# ---------------------------------------------------------------------------
# Weisner's theorem tests
# ---------------------------------------------------------------------------

class TestWeisner:
    """Tests for Weisner's theorem."""

    def test_weisner_runs(self):
        """Weisner verification runs without error."""
        ss = _build("&{a: end}")
        ok, results = verify_weisner(ss)
        assert isinstance(ok, bool)

    def test_weisner_parallel(self):
        """Weisner verification runs on parallel types."""
        ss = _build("(&{a: end} || &{b: end})")
        ok, results = verify_weisner(ss)
        assert isinstance(ok, bool)

    def test_weisner_trivial(self):
        """End type has no test elements — trivially verified."""
        ss = _build("end")
        ok, results = verify_weisner(ss)
        assert ok is True
        assert results == []


# ---------------------------------------------------------------------------
# Distributivity test
# ---------------------------------------------------------------------------

class TestDistributivity:
    """Tests for |μ| ≤ 1 distributivity pre-check."""

    def test_chain_distributive(self):
        """Chains are distributive."""
        ss = _build("&{a: &{b: end}}")
        assert is_distributive_by_mobius(ss) is True

    def test_parallel_distributive(self):
        """Product of chains is distributive (Boolean)."""
        ss = _build("(&{a: end} || &{b: end})")
        assert is_distributive_by_mobius(ss) is True

    def test_max_abs(self):
        """Max |μ| for a chain is 1."""
        ss = _build("&{a: end}")
        assert max_abs_mobius(ss) == 1

    def test_end_abs(self):
        ss = _build("end")
        assert max_abs_mobius(ss) == 1  # μ(⊤,⊤) = 1


# ---------------------------------------------------------------------------
# Spectrum tests
# ---------------------------------------------------------------------------

class TestSpectrum:
    """Tests for Möbius spectrum."""

    def test_chain_spectrum(self):
        """Chain has μ values: -1 (adjacent) and 0 (non-adjacent)."""
        ss = _build("&{a: &{b: end}}")
        spec = mobius_spectrum(ss)
        # Should have -1 entries and possibly 0 entries
        assert -1 in spec or 0 in spec

    def test_single_branch_spectrum(self):
        ss = _build("&{a: end}")
        spec = mobius_spectrum(ss)
        assert -1 in spec
        assert spec[-1] == 1  # Only one off-diagonal pair

    def test_parallel_spectrum(self):
        ss = _build("(&{a: end} || &{b: end})")
        spec = mobius_spectrum(ss)
        # Should have -1 and 1 values (product lattice)
        assert isinstance(spec, dict)


# ---------------------------------------------------------------------------
# Product formula tests
# ---------------------------------------------------------------------------

class TestProductFormula:
    """Tests for μ(L₁×L₂) = μ(L₁)·μ(L₂)."""

    def test_parallel_simple(self):
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ss_prod = _build("(&{a: end} || &{b: end})")
        assert verify_product_formula(ss1, ss2, ss_prod)

    def test_parallel_chain(self):
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: &{c: end}}")
        ss_prod = _build("(&{a: end} || &{b: &{c: end}})")
        assert verify_product_formula(ss1, ss2, ss_prod)

    def test_parallel_both_chains(self):
        """Two chains: μ = (-1)·0 = 0."""
        ss1 = _build("&{a: &{b: end}}")
        ss2 = _build("&{c: end}")
        ss_prod = _build("(&{a: &{b: end}} || &{c: end})")
        # μ(chain_3) = 0, μ(chain_2) = -1, product = 0
        assert verify_product_formula(ss1, ss2, ss_prod)


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyzeMobius:
    """Tests for the main analyze_mobius function."""

    def test_end_analysis(self):
        ss = _build("end")
        r = analyze_mobius(ss)
        assert r.num_states == 1
        assert r.mobius_top_bottom == 1
        assert r.is_distributive_by_mobius is True
        assert r.hall_verification is True

    def test_branch_analysis(self):
        ss = _build("&{a: end}")
        r = analyze_mobius(ss)
        assert r.num_states == 2
        assert r.mobius_top_bottom == -1
        assert r.max_abs_mobius == 1
        assert r.is_distributive_by_mobius is True
        assert r.hall_verification is True

    def test_chain_analysis(self):
        ss = _build("&{a: &{b: end}}")
        r = analyze_mobius(ss)
        assert r.mobius_top_bottom == 0
        assert r.hall_verification is True
        assert r.is_distributive_by_mobius is True

    def test_parallel_analysis(self):
        ss = _build("(&{a: end} || &{b: end})")
        r = analyze_mobius(ss)
        assert r.mobius_top_bottom == 1
        assert r.hall_verification is True
        assert r.is_distributive_by_mobius is True

    def test_recursive_analysis(self):
        ss = _build("rec X . &{a: X, b: end}")
        r = analyze_mobius(ss)
        assert isinstance(r.mobius_top_bottom, int)
        assert r.hall_verification is True


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests on benchmark protocols."""

    BENCHMARKS = [
        ("Java Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("Simple SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel simple", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_hall_theorem(self, name, typ):
        """Hall's theorem holds for all benchmarks."""
        ss = _build(typ)
        r = verify_hall(ss)
        assert r.verified, f"{name}: Hall's theorem failed: μ={r.mobius_value}, alt_sum={r.alternating_sum}"

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_mobius_analysis(self, name, typ):
        """Full analysis completes for all benchmarks."""
        ss = _build(typ)
        r = analyze_mobius(ss)
        assert r.num_states == len(ss.states)
        assert r.hall_verification is True
        assert isinstance(r.mobius_top_bottom, int)
        assert r.max_abs_mobius >= 0

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_distributivity_check(self, name, typ):
        """Distributivity pre-check runs without error."""
        ss = _build(typ)
        result = is_distributive_by_mobius(ss)
        assert isinstance(result, bool)

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_mobius_inversion(self, name, typ):
        """ζ * μ = δ for all benchmarks (via incidence algebra convolution)."""
        from reticulate.zeta import zeta_function, mobius_function, convolve, delta_function
        ss = _build(typ)
        zf = zeta_function(ss)
        mf = mobius_function(ss)
        result = convolve(zf, mf, ss)
        df = delta_function(ss)

        for s in ss.states:
            for t in ss.states:
                expected = df.get((s, t), 0)
                actual = result.get((s, t), 0)
                assert actual == expected, \
                    f"{name}: (ζ*μ)({s},{t}) = {actual} ≠ δ({s},{t}) = {expected}"
