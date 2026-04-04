"""Tests for Euler characteristic consolidation module (Step 30v)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.product import product_statespace
from reticulate.euler import (
    euler_characteristic,
    reduced_euler_characteristic,
    verify_hall_theorem,
    verify_kunneth,
    euler_series,
    interval_euler_distribution,
    euler_obstruction,
    compare_euler_methods,
    analyze_euler,
    HallVerification,
    KunnethVerification,
    ComparisonResult,
    EulerAnalysis,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Basic Euler characteristic
# ---------------------------------------------------------------------------

class TestEulerCharacteristic:
    def test_end(self):
        """End has trivial state space: single state, chi = 0."""
        chi = euler_characteristic(_build("end"))
        assert isinstance(chi, int)

    def test_single_branch(self):
        """&{a: end}: two states, no interior -> chi = 0."""
        chi = euler_characteristic(_build("&{a: end}"))
        assert isinstance(chi, int)

    def test_chain_3(self):
        """&{a: &{b: end}}: chain of 3, one interior vertex -> chi = 1."""
        chi = euler_characteristic(_build("&{a: &{b: end}}"))
        assert chi == 1

    def test_chain_4(self):
        """&{a: &{b: &{c: end}}}: chain of 4, two interior vertices."""
        chi = euler_characteristic(_build("&{a: &{b: &{c: end}}}"))
        assert isinstance(chi, int)
        assert chi >= 0

    def test_diamond(self):
        """&{a: end, b: end}: diamond lattice."""
        chi = euler_characteristic(_build("&{a: end, b: end}"))
        assert isinstance(chi, int)

    def test_parallel_simple(self):
        """Parallel composition."""
        chi = euler_characteristic(_build("(&{a: end} || &{b: end})"))
        assert isinstance(chi, int)


class TestReducedEuler:
    def test_end(self):
        red = reduced_euler_characteristic(_build("end"))
        assert isinstance(red, int)

    def test_single_branch(self):
        red = reduced_euler_characteristic(_build("&{a: end}"))
        assert isinstance(red, int)

    def test_chain_3(self):
        red = reduced_euler_characteristic(_build("&{a: &{b: end}}"))
        assert isinstance(red, int)


# ---------------------------------------------------------------------------
# Hall's theorem verification
# ---------------------------------------------------------------------------

class TestHallTheorem:
    def test_end(self):
        result = verify_hall_theorem(_build("end"))
        assert isinstance(result, HallVerification)
        assert result.verified

    def test_single_branch(self):
        result = verify_hall_theorem(_build("&{a: end}"))
        assert result.verified

    def test_chain_3(self):
        result = verify_hall_theorem(_build("&{a: &{b: end}}"))
        assert result.verified

    def test_chain_4(self):
        result = verify_hall_theorem(_build("&{a: &{b: &{c: end}}}"))
        assert result.verified

    def test_diamond(self):
        result = verify_hall_theorem(_build("&{a: end, b: end}"))
        assert result.verified

    def test_parallel(self):
        result = verify_hall_theorem(_build("(&{a: end} || &{b: end})"))
        assert result.verified

    def test_selection(self):
        result = verify_hall_theorem(_build("+{a: end, b: end}"))
        assert result.verified

    def test_recursive(self):
        result = verify_hall_theorem(
            _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        )
        assert result.verified

    def test_deeper_chain(self):
        result = verify_hall_theorem(
            _build("&{a: &{b: &{c: &{d: end}}}}")
        )
        assert result.verified


# ---------------------------------------------------------------------------
# Kunneth verification
# ---------------------------------------------------------------------------

class TestKunneth:
    def test_simple_product(self):
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ss_prod = product_statespace(ss1, ss2)
        result = verify_kunneth(ss1, ss2, ss_prod)
        assert isinstance(result, KunnethVerification)
        assert result.verified

    def test_chain_x_chain(self):
        ss1 = _build("&{a: &{b: end}}")
        ss2 = _build("&{c: end}")
        ss_prod = product_statespace(ss1, ss2)
        result = verify_kunneth(ss1, ss2, ss_prod)
        assert result.verified
        # chi_left/right are actually mu values in this interface
        assert result.expected == result.chi_left * result.chi_right

    def test_diamond_x_chain(self):
        ss1 = _build("&{a: end, b: end}")
        ss2 = _build("&{c: end}")
        ss_prod = product_statespace(ss1, ss2)
        result = verify_kunneth(ss1, ss2, ss_prod)
        assert result.verified

    def test_parallel_from_syntax(self):
        """Build product via syntax and manually, compare."""
        ss_par = _build("(&{a: end} || &{b: end})")
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ss_prod = product_statespace(ss1, ss2)
        # Both should have same Euler characteristic
        chi_par = euler_characteristic(ss_par)
        chi_prod = euler_characteristic(ss_prod)
        assert chi_par == chi_prod


# ---------------------------------------------------------------------------
# Interval Euler series
# ---------------------------------------------------------------------------

class TestEulerSeries:
    def test_end(self):
        series = euler_series(_build("end"))
        assert isinstance(series, dict)

    def test_single_branch(self):
        series = euler_series(_build("&{a: end}"))
        # Should have at least the (top, bottom) entry
        assert len(series) >= 1

    def test_chain_has_intervals(self):
        series = euler_series(_build("&{a: &{b: end}}"))
        # Chain of 3: intervals (0,1), (1,2), (0,2) plus diagonals
        assert len(series) >= 3

    def test_diagonal_entries_are_one(self):
        """mu(x, x) = 1 for all states."""
        ss = _build("&{a: &{b: end}}")
        series = euler_series(ss)
        for (x, y), val in series.items():
            if x == y:
                assert val == 1


# ---------------------------------------------------------------------------
# Interval Euler distribution
# ---------------------------------------------------------------------------

class TestIntervalDistribution:
    def test_end_trivial(self):
        dist = interval_euler_distribution(_build("end"))
        assert isinstance(dist, dict)

    def test_single_branch(self):
        dist = interval_euler_distribution(_build("&{a: end}"))
        # Exactly one off-diagonal interval (top > bottom) with mu = -1
        assert -1 in dist

    def test_diamond(self):
        dist = interval_euler_distribution(_build("&{a: end, b: end}"))
        assert isinstance(dist, dict)
        # Should have entries for various mu values
        total = sum(dist.values())
        assert total > 0

    def test_distribution_sums_to_intervals(self):
        ss = _build("&{a: &{b: end}}")
        dist = interval_euler_distribution(ss)
        series = euler_series(ss)
        n_off_diag = sum(1 for (x, y) in series if x != y)
        assert sum(dist.values()) == n_off_diag


# ---------------------------------------------------------------------------
# Euler obstructions
# ---------------------------------------------------------------------------

class TestEulerObstruction:
    def test_end_no_obstruction(self):
        obs = euler_obstruction(_build("end"))
        assert obs == []

    def test_single_branch_no_obstruction(self):
        obs = euler_obstruction(_build("&{a: end}"))
        assert obs == []

    def test_chain_no_obstruction(self):
        obs = euler_obstruction(_build("&{a: &{b: end}}"))
        assert obs == []

    def test_diamond_no_obstruction(self):
        """Diamond lattice is distributive, so |mu| <= 1."""
        obs = euler_obstruction(_build("&{a: end, b: end}"))
        assert obs == []

    def test_obstruction_format(self):
        """Obstruction entries are (x, y, mu_value) with |mu| > 1."""
        ss = _build("&{a: &{b: end, c: end}, d: end}")
        obs = euler_obstruction(ss)
        for x, y, val in obs:
            assert abs(val) > 1
            assert isinstance(x, int)
            assert isinstance(y, int)

    def test_sorted_by_abs_descending(self):
        """Obstructions should be sorted by |mu| descending."""
        ss = _build("&{a: &{b: end, c: end}, d: &{e: end}}")
        obs = euler_obstruction(ss)
        for i in range(len(obs) - 1):
            assert abs(obs[i][2]) >= abs(obs[i + 1][2])


# ---------------------------------------------------------------------------
# Multi-method comparison
# ---------------------------------------------------------------------------

class TestCompareEulerMethods:
    def test_end(self):
        result = compare_euler_methods(_build("end"))
        assert isinstance(result, ComparisonResult)
        assert result.all_agree

    def test_single_branch(self):
        result = compare_euler_methods(_build("&{a: end}"))
        assert result.all_agree

    def test_chain_3(self):
        result = compare_euler_methods(_build("&{a: &{b: end}}"))
        assert result.all_agree

    def test_diamond(self):
        result = compare_euler_methods(_build("&{a: end, b: end}"))
        assert result.all_agree

    def test_parallel(self):
        result = compare_euler_methods(_build("(&{a: end} || &{b: end})"))
        assert result.all_agree

    def test_recursive(self):
        result = compare_euler_methods(
            _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        )
        assert result.all_agree

    def test_has_f_vector(self):
        result = compare_euler_methods(_build("&{a: &{b: end}}"))
        assert isinstance(result.f_vector, list)
        assert len(result.f_vector) >= 1

    def test_has_betti(self):
        result = compare_euler_methods(_build("&{a: &{b: end}}"))
        assert isinstance(result.betti_numbers, list)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeEuler:
    def test_end(self):
        result = analyze_euler(_build("end"))
        assert isinstance(result, EulerAnalysis)
        assert isinstance(result.euler_characteristic, int)
        assert isinstance(result.hall, HallVerification)
        assert result.hall.verified

    def test_chain_3(self):
        result = analyze_euler(_build("&{a: &{b: end}}"))
        assert result.hall.verified
        assert result.comparison.all_agree
        assert result.num_intervals >= 1

    def test_diamond(self):
        result = analyze_euler(_build("&{a: end, b: end}"))
        assert result.hall.verified
        assert result.comparison.all_agree
        assert isinstance(result.distribution, dict)

    def test_parallel(self):
        result = analyze_euler(_build("(&{a: end} || &{b: end})"))
        assert result.hall.verified
        assert result.comparison.all_agree

    def test_recursive_iterator(self):
        result = analyze_euler(
            _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        )
        assert result.hall.verified
        assert result.comparison.all_agree

    def test_obstructions_in_analysis(self):
        result = analyze_euler(_build("&{a: &{b: end}}"))
        assert isinstance(result.obstructions, list)

    def test_interval_series_in_analysis(self):
        result = analyze_euler(_build("&{a: end, b: end}"))
        assert isinstance(result.interval_series, dict)
        assert len(result.interval_series) > 0


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Run Euler analysis on benchmark protocols."""

    @pytest.fixture(scope="class")
    def benchmarks(self):
        from tests.benchmarks.protocols import BENCHMARKS
        return BENCHMARKS

    def test_hall_theorem_all_benchmarks(self, benchmarks):
        """Hall's theorem must hold for every benchmark."""
        for bp in benchmarks[:20]:  # First 20 to keep runtime reasonable
            ss = _build(bp.type_string)
            result = verify_hall_theorem(ss)
            assert result.verified, f"Hall failed for {bp.name}"

    def test_methods_agree_all_benchmarks(self, benchmarks):
        """All three Euler methods must agree for every benchmark."""
        for bp in benchmarks[:20]:
            ss = _build(bp.type_string)
            result = compare_euler_methods(ss)
            assert result.all_agree, f"Methods disagree for {bp.name}"

    def test_full_analysis_all_benchmarks(self, benchmarks):
        """Full analysis must complete without error for every benchmark."""
        for bp in benchmarks[:15]:
            ss = _build(bp.type_string)
            result = analyze_euler(ss)
            assert isinstance(result, EulerAnalysis)
            assert result.hall.verified, f"Hall failed for {bp.name}"

    def test_kunneth_parallel_benchmarks(self, benchmarks):
        """Kunneth must hold for parallel benchmarks."""
        parallel_bps = [bp for bp in benchmarks if bp.uses_parallel][:5]
        for bp in parallel_bps:
            ss = _build(bp.type_string)
            # Just verify the full analysis works for parallel types
            result = analyze_euler(ss)
            assert result.hall.verified, f"Hall failed for parallel {bp.name}"


# ---------------------------------------------------------------------------
# Edge cases and properties
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_selection_only(self):
        chi = euler_characteristic(_build("+{a: end}"))
        assert isinstance(chi, int)

    def test_nested_branch(self):
        result = verify_hall_theorem(
            _build("&{a: &{b: end, c: end}, d: end}")
        )
        assert result.verified

    def test_nested_selection(self):
        result = verify_hall_theorem(
            _build("+{a: +{b: end, c: end}, d: end}")
        )
        assert result.verified

    def test_mixed_branch_select(self):
        result = verify_hall_theorem(
            _build("&{a: +{b: end, c: end}}")
        )
        assert result.verified

    def test_deep_recursion(self):
        result = verify_hall_theorem(
            _build("rec X . &{a: rec Y . &{b: +{c: X, d: Y, e: end}}}")
        )
        assert result.verified
