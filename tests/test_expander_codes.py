"""Tests for protocol-aware error correction via expander codes (Step 31e)."""

import math
import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.expander_codes import (
    CheegerBounds,
    ExpanderCodeResult,
    analyze_expander_code,
    cheeger_bounds,
    code_distance,
    error_correction_capacity,
    expansion_ratio,
    is_expander,
    parity_check_matrix,
    spectral_gap,
)


def _ss(type_string: str):
    return build_statespace(parse(type_string))


# ---------------------------------------------------------------------------
# Spectral gap
# ---------------------------------------------------------------------------


class TestSpectralGap:
    def test_end_gap(self):
        ss = _ss("end")
        assert spectral_gap(ss) == 0.0

    def test_simple_branch_gap(self):
        ss = _ss("&{a: end}")
        gap = spectral_gap(ss)
        assert gap >= 0.0

    def test_chain_gap(self):
        ss = _ss("&{a: &{b: end}}")
        gap = spectral_gap(ss)
        assert gap > 0.0

    def test_diamond_gap(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        gap = spectral_gap(ss)
        assert gap > 0.0

    def test_parallel_gap(self):
        ss = _ss("(&{a: end} || &{b: end})")
        gap = spectral_gap(ss)
        assert gap >= 0.0


# ---------------------------------------------------------------------------
# Expansion ratio
# ---------------------------------------------------------------------------


class TestExpansionRatio:
    def test_end_expansion(self):
        ss = _ss("end")
        assert expansion_ratio(ss) == 0.0

    def test_simple_branch_expansion(self):
        ss = _ss("&{a: end}")
        ratio = expansion_ratio(ss)
        assert ratio >= 0.0

    def test_chain_expansion(self):
        ss = _ss("&{a: &{b: end}}")
        ratio = expansion_ratio(ss)
        assert ratio >= 0.0

    def test_diamond_expansion(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        ratio = expansion_ratio(ss)
        assert ratio >= 0.0

    def test_parallel_expansion(self):
        ss = _ss("(&{a: end} || &{b: end})")
        ratio = expansion_ratio(ss)
        assert ratio >= 0.0


# ---------------------------------------------------------------------------
# Expander test
# ---------------------------------------------------------------------------


class TestIsExpander:
    def test_end_not_expander(self):
        assert not is_expander(_ss("end"))

    def test_simple_branch_threshold(self):
        ss = _ss("&{a: end}")
        # With default threshold, a 2-node path has gap = 2.0
        result = is_expander(ss)
        assert isinstance(result, bool)

    def test_custom_threshold(self):
        ss = _ss("&{a: &{b: end}}")
        # Very low threshold should be easy to meet
        assert is_expander(ss, threshold=0.01)

    def test_high_threshold(self):
        ss = _ss("&{a: &{b: &{c: end}}}")
        # Very high threshold unlikely
        result = is_expander(ss, threshold=100.0)
        assert not result


# ---------------------------------------------------------------------------
# Cheeger bounds
# ---------------------------------------------------------------------------


class TestCheegerBounds:
    def test_end_cheeger(self):
        ss = _ss("end")
        cb = cheeger_bounds(ss)
        assert isinstance(cb, CheegerBounds)
        assert cb.spectral_gap == 0.0
        assert cb.lower == 0.0

    def test_chain_cheeger_inequality(self):
        ss = _ss("&{a: &{b: end}}")
        cb = cheeger_bounds(ss)
        # Cheeger inequality: lower <= h(G) <= upper
        assert cb.lower <= cb.upper or cb.spectral_gap == 0.0
        assert cb.lower >= 0.0

    def test_diamond_cheeger(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        cb = cheeger_bounds(ss)
        assert cb.max_degree >= 1
        assert cb.spectral_gap > 0.0

    def test_cheeger_lower_is_half_gap(self):
        ss = _ss("&{a: end}")
        cb = cheeger_bounds(ss)
        assert abs(cb.lower - cb.spectral_gap / 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Parity check matrix
# ---------------------------------------------------------------------------


class TestParityCheckMatrix:
    def test_end_empty(self):
        H = parity_check_matrix(_ss("end"))
        # 1 state, 0 edges
        assert len(H) == 1
        assert len(H[0]) == 0

    def test_simple_branch_shape(self):
        ss = _ss("&{a: end}")
        H = parity_check_matrix(ss)
        # 2 states, 1 edge
        assert len(H) == 2
        assert len(H[0]) == 1
        # Each edge column has exactly 2 ones
        col_sum = sum(H[r][0] for r in range(2))
        assert col_sum == 2

    def test_chain_shape(self):
        ss = _ss("&{a: &{b: end}}")
        H = parity_check_matrix(ss)
        # 3 states, 2 edges
        assert len(H) == 3
        assert len(H[0]) == 2

    def test_binary_entries(self):
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        H = parity_check_matrix(ss)
        for row in H:
            for val in row:
                assert val in (0, 1)

    def test_column_weight_two(self):
        """Each column of the incidence matrix has exactly 2 ones."""
        ss = _ss("&{a: &{b: end}}")
        H = parity_check_matrix(ss)
        n_cols = len(H[0])
        for j in range(n_cols):
            col_sum = sum(H[i][j] for i in range(len(H)))
            assert col_sum == 2


# ---------------------------------------------------------------------------
# Code distance
# ---------------------------------------------------------------------------


class TestCodeDistance:
    def test_end_trivial(self):
        assert code_distance(_ss("end")) == 0

    def test_simple_branch_distance(self):
        d = code_distance(_ss("&{a: end}"))
        assert d >= 0

    def test_chain_distance(self):
        d = code_distance(_ss("&{a: &{b: end}}"))
        assert d >= 0

    def test_diamond_distance(self):
        d = code_distance(_ss("&{a: &{c: end}, b: &{c: end}}"))
        assert d >= 0


# ---------------------------------------------------------------------------
# Error correction capacity
# ---------------------------------------------------------------------------


class TestErrorCorrectionCapacity:
    def test_end_zero(self):
        assert error_correction_capacity(_ss("end")) == 0

    def test_non_negative(self):
        ss = _ss("&{a: &{b: end}}")
        assert error_correction_capacity(ss) >= 0

    def test_floor_formula(self):
        """Capacity = floor((d-1)/2)."""
        ss = _ss("&{a: &{c: end}, b: &{c: end}}")
        d = code_distance(ss)
        cap = error_correction_capacity(ss)
        assert cap == max(0, (d - 1) // 2)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


class TestAnalyzeExpanderCode:
    def test_end_analysis(self):
        result = analyze_expander_code(_ss("end"))
        assert isinstance(result, ExpanderCodeResult)
        assert result.num_states == 1
        assert result.spectral_gap == 0.0
        assert not result.is_expander

    def test_simple_branch_analysis(self):
        result = analyze_expander_code(_ss("&{a: end}"))
        assert result.num_states == 2
        assert result.num_edges >= 1

    def test_chain_analysis(self):
        result = analyze_expander_code(_ss("&{a: &{b: end}}"))
        assert result.num_states == 3
        assert result.cheeger.lower <= result.cheeger.upper or result.spectral_gap == 0.0

    def test_diamond_analysis(self):
        result = analyze_expander_code(_ss("&{a: &{c: end}, b: &{c: end}}"))
        assert result.spectral_gap > 0.0
        assert result.expansion_ratio >= 0.0

    def test_parallel_analysis(self):
        result = analyze_expander_code(_ss("(&{a: end} || &{b: end})"))
        assert result.num_states >= 3

    def test_recursive_analysis(self):
        result = analyze_expander_code(_ss("rec X . &{a: X, b: end}"))
        assert isinstance(result, ExpanderCodeResult)

    def test_rate_bounded(self):
        result = analyze_expander_code(_ss("&{a: &{b: end}}"))
        assert 0.0 <= result.rate <= 1.0

    def test_custom_threshold(self):
        ss = _ss("&{a: &{b: end}}")
        r1 = analyze_expander_code(ss, threshold=0.01)
        r2 = analyze_expander_code(ss, threshold=100.0)
        assert r1.is_expander or not r2.is_expander
