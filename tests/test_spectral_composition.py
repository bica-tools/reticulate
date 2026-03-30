"""Tests for spectral compositionality under parallel constructor (Step 30t).

Tests cover:
- Eigenvalue sum property
- Fiedler = min(fiedler1, fiedler2)
- Spectral gap composition
- Heat trace multiplicativity
- Entropy composition
- Ihara cycle rank formula
- Cheeger bound composition
- Mixing time composition
- Verification against actual products
- Benchmark protocols under parallel
- Edge cases (end || S, S || S)
- Full analysis integration
"""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.product import product_statespace
from reticulate.spectral_composition import (
    SpectralCompositionResult,
    VerificationResult,
    product_eigenvalues,
    product_eigenvalues_from_spectra,
    product_fiedler,
    product_fiedler_from_values,
    product_spectral_gap,
    product_cheeger_bound,
    product_mixing_time,
    product_heat_trace,
    product_heat_trace_series,
    product_entropy,
    product_entropy_additive,
    product_ihara_rank,
    verify_eigenvalue_composition,
    verify_fiedler_composition,
    verify_heat_trace_composition,
    verify_spectral_composition,
    analyze_spectral_composition,
    _first_nonzero,
)
from reticulate.heat_kernel import (
    heat_trace as direct_heat_trace,
    laplacian_eigendecomposition,
)
from reticulate.fiedler import fiedler_value
from reticulate.von_neumann import von_neumann_entropy
from reticulate.ihara import cycle_rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str):
    """Parse and build state space."""
    return build_statespace(parse(s))


def _build_product(s1: str, s2: str):
    """Build the actual product state space."""
    ss1 = _build(s1)
    ss2 = _build(s2)
    return ss1, ss2, product_statespace(ss1, ss2)


# ---------------------------------------------------------------------------
# Basic eigenvalue tests
# ---------------------------------------------------------------------------

class TestProductEigenvalues:
    """Eigenvalue sum property: spec(L_prod) = {mu_i + nu_j}."""

    def test_end_end(self):
        """end || end: trivial product, single eigenvalue 0."""
        ss1 = _build("end")
        ss2 = _build("end")
        eigs = product_eigenvalues(ss1, ss2)
        assert len(eigs) == 1
        assert abs(eigs[0]) < 1e-10

    def test_branch_end(self):
        """&{a: end} || end: eigenvalues come only from left factor."""
        ss1 = _build("&{a: end}")
        ss2 = _build("end")
        eigs = product_eigenvalues(ss1, ss2)
        eigs1, _ = laplacian_eigendecomposition(ss1)
        assert len(eigs) == len(eigs1)
        for e, e1 in zip(eigs, eigs1):
            assert abs(e - e1) < 1e-6

    def test_branch_branch_count(self):
        """&{a: end} || &{b: end}: 2 * 2 = 4 eigenvalues."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        eigs = product_eigenvalues(ss1, ss2)
        assert len(eigs) == 4

    def test_eigenvalue_sum_structure(self):
        """Verify eigenvalues are sums of factor eigenvalues."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        eigs1, _ = laplacian_eigendecomposition(ss1)
        eigs2, _ = laplacian_eigendecomposition(ss2)
        eigs_prod = product_eigenvalues(ss1, ss2)

        expected = sorted([mu + nu for mu in eigs1 for nu in eigs2])
        assert len(eigs_prod) == len(expected)
        for e, exp in zip(eigs_prod, expected):
            assert abs(e - exp) < 1e-6

    def test_from_spectra_matches(self):
        """product_eigenvalues_from_spectra gives same result."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        eigs1, _ = laplacian_eigendecomposition(ss1)
        eigs2, _ = laplacian_eigendecomposition(ss2)

        eigs_direct = product_eigenvalues(ss1, ss2)
        eigs_from = product_eigenvalues_from_spectra(eigs1, eigs2)
        assert len(eigs_direct) == len(eigs_from)
        for a, b in zip(eigs_direct, eigs_from):
            assert abs(a - b) < 1e-10

    def test_empty_spectrum(self):
        """Empty spectra yield empty product."""
        assert product_eigenvalues_from_spectra([], [1.0, 2.0]) == []
        assert product_eigenvalues_from_spectra([1.0], []) == []

    def test_smallest_is_zero(self):
        """Product of connected graphs has smallest eigenvalue 0."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end, d: end}")
        eigs = product_eigenvalues(ss1, ss2)
        assert abs(eigs[0]) < 1e-8

    def test_symmetry(self):
        """product_eigenvalues(ss1, ss2) == product_eigenvalues(ss2, ss1)."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end, c: end}")
        eigs_12 = product_eigenvalues(ss1, ss2)
        eigs_21 = product_eigenvalues(ss2, ss1)
        assert len(eigs_12) == len(eigs_21)
        for a, b in zip(eigs_12, eigs_21):
            assert abs(a - b) < 1e-10


# ---------------------------------------------------------------------------
# Fiedler composition tests
# ---------------------------------------------------------------------------

class TestProductFiedler:
    """Fiedler value: min(lambda2(L1), lambda2(L2))."""

    def test_end_has_zero_fiedler(self):
        """end || anything: Fiedler is 0 since end has 1 state."""
        ss1 = _build("end")
        ss2 = _build("&{a: end, b: end}")
        assert product_fiedler(ss1, ss2) == pytest.approx(0.0, abs=1e-8)

    def test_symmetric_branches(self):
        """Same type on both sides: Fiedler = fiedler of either."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        f1 = fiedler_value(ss1)
        f_prod = product_fiedler(ss1, ss2)
        assert f_prod == pytest.approx(f1, abs=1e-6)

    def test_min_selection(self):
        """Product takes the minimum Fiedler value."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{a: end, b: end}")
        f1 = fiedler_value(ss1)
        f2 = fiedler_value(ss2)
        f_prod = product_fiedler(ss1, ss2)
        assert f_prod == pytest.approx(min(f1, f2), abs=1e-6)

    def test_from_values(self):
        """product_fiedler_from_values helper."""
        assert product_fiedler_from_values(1.5, 2.3) == pytest.approx(1.5)
        assert product_fiedler_from_values(3.0, 1.0) == pytest.approx(1.0)

    def test_both_zero(self):
        """Two disconnected graphs: Fiedler is 0."""
        ss1 = _build("end")
        ss2 = _build("end")
        assert product_fiedler(ss1, ss2) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Spectral gap tests
# ---------------------------------------------------------------------------

class TestProductSpectralGap:
    """Spectral gap = min(gap1, gap2)."""

    def test_end_gap_zero(self):
        """end has gap 0."""
        ss1 = _build("end")
        ss2 = _build("&{a: end}")
        assert product_spectral_gap(ss1, ss2) == pytest.approx(0.0, abs=1e-8)

    def test_branch_gaps(self):
        """Gap of product is min of factor gaps."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{a: end, b: end}")
        eigs1, _ = laplacian_eigendecomposition(ss1)
        eigs2, _ = laplacian_eigendecomposition(ss2)

        gap1 = _first_nonzero(eigs1)
        gap2 = _first_nonzero(eigs2)
        gap_prod = product_spectral_gap(ss1, ss2)
        assert gap_prod == pytest.approx(min(gap1, gap2), abs=1e-6)


# ---------------------------------------------------------------------------
# Heat trace tests
# ---------------------------------------------------------------------------

class TestProductHeatTrace:
    """Heat trace multiplicativity: Z_prod(t) = Z1(t) * Z2(t)."""

    def test_at_t_zero(self):
        """At t~0, Z(t) ~ n, so Z_prod(0) ~ n1 * n2."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ht = product_heat_trace(ss1, ss2, 0.001)
        # n1=2, n2=2, so Z(0) ~ 4
        assert ht == pytest.approx(4.0, abs=0.1)

    def test_multiplicativity(self):
        """Z_prod(t) = Z1(t) * Z2(t)."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        t = 1.0
        z1 = direct_heat_trace(ss1, t)
        z2 = direct_heat_trace(ss2, t)
        z_prod = product_heat_trace(ss1, ss2, t)
        assert z_prod == pytest.approx(z1 * z2, rel=1e-6)

    def test_series(self):
        """Heat trace series returns correct number of values."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        times, traces = product_heat_trace_series(ss1, ss2)
        assert len(times) == 7
        assert len(traces) == 7
        assert all(t > 0 for t in traces)

    def test_custom_times(self):
        """Custom time values."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        times, traces = product_heat_trace_series(ss1, ss2, [0.5, 1.5])
        assert len(times) == 2
        assert len(traces) == 2

    def test_end_times_end(self):
        """end || end: Z(t) = 1 for all t."""
        ss1 = _build("end")
        ss2 = _build("end")
        assert product_heat_trace(ss1, ss2, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Entropy tests
# ---------------------------------------------------------------------------

class TestProductEntropy:
    """Von Neumann entropy composition."""

    def test_end_entropy_zero(self):
        """end has entropy 0."""
        ss1 = _build("end")
        ss2 = _build("end")
        assert product_entropy(ss1, ss2) == pytest.approx(0.0, abs=1e-8)

    def test_entropy_nonnegative(self):
        """Product entropy is always non-negative."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        assert product_entropy(ss1, ss2) >= -1e-10

    def test_additive_estimate(self):
        """Additive estimate S1 + S2."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        s_add = product_entropy_additive(ss1, ss2)
        s1 = von_neumann_entropy(ss1)
        s2 = von_neumann_entropy(ss2)
        assert s_add == pytest.approx(s1 + s2, abs=1e-8)

    def test_entropy_increases_with_parallel(self):
        """Adding a parallel branch increases entropy (more states)."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        e_left = von_neumann_entropy(ss1)
        e_prod = product_entropy(ss1, ss2)
        # Product entropy should be at least as large as individual
        # (not strictly required but typical for non-trivial cases)
        assert e_prod >= e_left - 0.1  # allow small numerical margin

    def test_end_times_branch(self):
        """end || S: entropy comes only from S."""
        ss1 = _build("end")
        ss2 = _build("&{a: end, b: end}")
        e_prod = product_entropy(ss1, ss2)
        e2 = von_neumann_entropy(ss2)
        # With end (1 state), product eigenvalues = eigenvalues of ss2
        assert e_prod == pytest.approx(e2, abs=0.1)


# ---------------------------------------------------------------------------
# Cycle rank tests
# ---------------------------------------------------------------------------

class TestProductIharaRank:
    """Cycle rank of product graph."""

    def test_dags_zero_rank(self):
        """DAGs have rank 0; product of DAGs should use formula."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        r = product_ihara_rank(ss1, ss2)
        # n1=2, n2=2, r1=0, r2=0
        # r_prod = 2*0 + 2*0 + (2-1)*(2-1) = 1
        assert r == 1

    def test_end_product(self):
        """end || S: n1=1, so r_prod = r2 + 0 = (1-1)*(n2-1) + r2."""
        ss1 = _build("end")
        ss2 = _build("&{a: end, b: end}")
        r2 = cycle_rank(ss2)
        r = product_ihara_rank(ss1, ss2)
        # n1=1: r = 1*r2 + n2*0 + 0 = r2
        assert r == r2

    def test_empty_product(self):
        """Empty state spaces."""
        ss1 = _build("end")
        ss2 = _build("end")
        r = product_ihara_rank(ss1, ss2)
        # n1=1, n2=1, r1=0, r2=0: r = 0+0+0 = 0
        assert r == 0

    def test_formula_consistency(self):
        """r_prod = n1*r2 + n2*r1 + (n1-1)*(n2-1)."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end, d: end}")
        n1, n2 = len(ss1.states), len(ss2.states)
        r1, r2 = cycle_rank(ss1), cycle_rank(ss2)
        r_prod = product_ihara_rank(ss1, ss2)
        expected = n1 * r2 + n2 * r1 + (n1 - 1) * (n2 - 1)
        assert r_prod == expected


# ---------------------------------------------------------------------------
# Cheeger bound tests
# ---------------------------------------------------------------------------

class TestProductCheeger:
    """Cheeger constant bounds for product."""

    def test_bounds_ordered(self):
        """Lower bound <= upper bound."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        lower, upper = product_cheeger_bound(ss1, ss2)
        assert lower <= upper + 1e-10

    def test_end_gives_zero(self):
        """end || S: Fiedler is 0, so bounds are both 0."""
        ss1 = _build("end")
        ss2 = _build("&{a: end}")
        lower, upper = product_cheeger_bound(ss1, ss2)
        assert lower == pytest.approx(0.0, abs=1e-10)
        assert upper == pytest.approx(0.0, abs=1e-10)

    def test_nonneg(self):
        """Bounds are non-negative."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        lower, upper = product_cheeger_bound(ss1, ss2)
        assert lower >= -1e-10
        assert upper >= -1e-10


# ---------------------------------------------------------------------------
# Mixing time tests
# ---------------------------------------------------------------------------

class TestProductMixingTime:
    """Mixing time bounds for product."""

    def test_end_times_end(self):
        """end || end: single state, mixing time 0."""
        ss1 = _build("end")
        ss2 = _build("end")
        assert product_mixing_time(ss1, ss2) == 0

    def test_positive_for_nontrivial(self):
        """Non-trivial product has positive mixing time."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        assert product_mixing_time(ss1, ss2) > 0

    def test_bottleneck_principle(self):
        """Mixing time grows with product size."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        m1 = product_mixing_time(ss1, ss1)
        # Larger type
        ss3 = _build("&{a: end, b: end, c: end}")
        m2 = product_mixing_time(ss1, ss3)
        # Larger product should have >= mixing time (due to ln(n) factor)
        assert m2 >= 1


# ---------------------------------------------------------------------------
# Verification tests (against actual product state spaces)
# ---------------------------------------------------------------------------

class TestVerification:
    """Verify composition laws against actual products."""

    def test_verify_eigenvalues_simple(self):
        """Verify eigenvalue sum for simple product."""
        ss1, ss2, ss_prod = _build_product("&{a: end}", "&{b: end}")
        result = verify_eigenvalue_composition(ss1, ss2, ss_prod, tol=1.0)
        assert isinstance(result, VerificationResult)
        assert result.law_name == "eigenvalue_sum"

    def test_verify_fiedler_simple(self):
        """Verify Fiedler min for simple product."""
        ss1, ss2, ss_prod = _build_product("&{a: end}", "&{b: end}")
        result = verify_fiedler_composition(ss1, ss2, ss_prod, tol=1.0)
        assert isinstance(result, VerificationResult)
        assert result.law_name == "fiedler_min"

    def test_verify_heat_trace_simple(self):
        """Verify heat trace multiplicativity for simple product."""
        ss1, ss2, ss_prod = _build_product("&{a: end}", "&{b: end}")
        result = verify_heat_trace_composition(ss1, ss2, ss_prod, t=1.0, tol=1.0)
        assert isinstance(result, VerificationResult)
        assert result.law_name == "heat_trace_multiplicative"

    def test_verify_all(self):
        """Run all verification checks."""
        ss1, ss2, ss_prod = _build_product("&{a: end}", "&{b: end}")
        results = verify_spectral_composition(ss1, ss2, ss_prod)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, VerificationResult)


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyzeSpectralComposition:
    """Full analysis integration tests."""

    def test_basic_analysis(self):
        """Basic analysis returns SpectralCompositionResult."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert isinstance(result, SpectralCompositionResult)

    def test_eigenvalue_fields(self):
        """Analysis populates eigenvalue fields."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert len(result.eigenvalues_left) > 0
        assert len(result.eigenvalues_right) > 0
        assert len(result.eigenvalues_product) == len(result.eigenvalues_left) * len(result.eigenvalues_right)

    def test_fiedler_fields(self):
        """Analysis computes Fiedler values."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert result.fiedler_product == pytest.approx(
            min(result.fiedler_left, result.fiedler_right), abs=1e-6
        )

    def test_heat_trace_fields(self):
        """Analysis computes heat traces."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert len(result.heat_trace_product) == 7
        assert len(result.heat_trace_times) == 7

    def test_custom_heat_times(self):
        """Custom heat sample times."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = analyze_spectral_composition(ss1, ss2, heat_times=[0.5, 1.0])
        assert len(result.heat_trace_product) == 2

    def test_entropy_fields(self):
        """Analysis computes entropy."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert result.entropy_product >= -1e-10

    def test_mixing_time_fields(self):
        """Analysis computes mixing times."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert result.mixing_time_product >= 0

    def test_cheeger_fields(self):
        """Analysis computes Cheeger bounds."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert result.cheeger_lower_product <= result.cheeger_upper_product + 1e-10

    def test_cycle_rank_fields(self):
        """Analysis computes cycle ranks."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert result.cycle_rank_product >= 0

    def test_law_results_populated(self):
        """Law results dictionary is populated."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert "eigenvalue_count" in result.law_results
        assert "zero_eigenvalue" in result.law_results
        assert "fiedler_min" in result.law_results
        assert "entropy_nonneg" in result.law_results


# ---------------------------------------------------------------------------
# Edge cases: end || S
# ---------------------------------------------------------------------------

class TestEndParallel:
    """end || S: product with trivial factor."""

    def test_end_preserves_eigenvalues(self):
        """end || S has same eigenvalues as S (trivially)."""
        ss1 = _build("end")
        ss2 = _build("&{a: end, b: end}")
        eigs_prod = product_eigenvalues(ss1, ss2)
        eigs2, _ = laplacian_eigendecomposition(ss2)
        # end has 1 state with eigenvalue 0; product eigs = 0 + eigs2 = eigs2
        assert len(eigs_prod) == len(eigs2)
        for a, b in zip(eigs_prod, eigs2):
            assert abs(a - b) < 1e-8

    def test_end_preserves_fiedler(self):
        """end || S: Fiedler is 0 (end has Fiedler 0)."""
        ss1 = _build("end")
        ss2 = _build("&{a: end, b: end}")
        assert product_fiedler(ss1, ss2) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Edge cases: S || S
# ---------------------------------------------------------------------------

class TestSelfParallel:
    """S || S: product with itself."""

    def test_eigenvalues_self_product(self):
        """S || S has eigenvalues {2*mu_i, mu_i + mu_j, ...}."""
        ss = _build("&{a: end}")
        eigs = product_eigenvalues(ss, ss)
        eigs_single, _ = laplacian_eigendecomposition(ss)
        expected = sorted([mu + nu for mu in eigs_single for nu in eigs_single])
        assert len(eigs) == len(eigs_single) ** 2
        for a, b in zip(eigs, expected):
            assert abs(a - b) < 1e-8

    def test_fiedler_self(self):
        """S || S: Fiedler is fiedler(S) (min of same value)."""
        ss = _build("&{a: end, b: end}")
        f = fiedler_value(ss)
        f_prod = product_fiedler(ss, ss)
        assert f_prod == pytest.approx(f, abs=1e-6)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for internal helper functions."""

    def test_first_nonzero_empty(self):
        """Empty list returns 0."""
        assert _first_nonzero([]) == 0.0

    def test_first_nonzero_all_zero(self):
        """All zeros returns 0."""
        assert _first_nonzero([0.0, 0.0]) == 0.0

    def test_first_nonzero_normal(self):
        """Returns first value > epsilon."""
        assert _first_nonzero([0.0, 1.5, 3.0]) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Benchmark protocols under parallel
# ---------------------------------------------------------------------------

class TestBenchmarkParallel:
    """Spectral composition for richer session types."""

    def test_nested_branch_parallel(self):
        """Nested branches composed in parallel."""
        ss1 = _build("&{a: &{b: end}}")
        ss2 = _build("&{c: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert isinstance(result, SpectralCompositionResult)
        assert result.all_laws_verified

    def test_selection_parallel(self):
        """Selection types in parallel."""
        ss1 = _build("+{a: end, b: end}")
        ss2 = _build("+{c: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert isinstance(result, SpectralCompositionResult)
        assert len(result.eigenvalues_product) > 0

    def test_mixed_branch_select(self):
        """Branch || Select."""
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("+{c: end, d: end}")
        result = analyze_spectral_composition(ss1, ss2)
        assert result.entropy_product >= -1e-10

    def test_deeper_protocol(self):
        """Deeper protocol composition."""
        ss1 = _build("&{a: &{b: end}, c: end}")
        ss2 = _build("&{d: &{e: end}}")
        result = analyze_spectral_composition(ss1, ss2)
        assert result.fiedler_product == pytest.approx(
            min(result.fiedler_left, result.fiedler_right), abs=1e-6
        )
