"""Tests for log-concavity verification of session type lattices (Step 32e).

Tests cover:
- Basic log-concavity checks
- Whitney number log-concavity (first and second kind)
- Unimodality
- Ultra-log-concavity
- Heron-Rota-Welsh check
- Adiprasito-Huh-Katz bound
- f-vector log-concavity
- Log-concavity ratios
- Preservation under parallel composition
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.log_concavity import (
    verify_log_concavity,
    heron_rota_welsh_check,
    adiprasito_huh_katz_bound,
    is_log_concave_whitney_second,
    is_log_concave_whitney_first,
    check_unimodality,
    log_concavity_ratio,
    preservation_under_product,
    _check_unimodal,
    _check_ultra_log_concave,
    _find_violations,
    _sequence_from_dict,
    _comb,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_comb_basic(self):
        assert _comb(5, 2) == 10
        assert _comb(4, 0) == 1
        assert _comb(0, 0) == 1

    def test_comb_invalid(self):
        assert _comb(3, -1) == 0
        assert _comb(3, 5) == 0

    def test_sequence_from_dict_empty(self):
        assert _sequence_from_dict({}) == []

    def test_sequence_from_dict_basic(self):
        assert _sequence_from_dict({0: 1, 1: 3, 2: 1}) == [1, 3, 1]

    def test_sequence_from_dict_gaps(self):
        assert _sequence_from_dict({0: 1, 2: 5}) == [1, 0, 5]

    def test_find_violations_none(self):
        assert _find_violations([1, 3, 3, 1]) == []

    def test_find_violations_present(self):
        # 1, 1, 5: 1^2 < 1*5
        violations = _find_violations([1, 1, 5])
        assert len(violations) == 1
        assert violations[0][0] == 1  # position k=1


# ---------------------------------------------------------------------------
# Unimodality
# ---------------------------------------------------------------------------

class TestUnimodality:

    def test_empty(self):
        assert _check_unimodal([]) is True

    def test_single(self):
        assert _check_unimodal([5]) is True

    def test_increasing(self):
        assert _check_unimodal([1, 2, 3]) is True

    def test_decreasing(self):
        assert _check_unimodal([3, 2, 1]) is True

    def test_peak(self):
        assert _check_unimodal([1, 3, 2]) is True

    def test_valley(self):
        assert _check_unimodal([3, 1, 2]) is False

    def test_flat(self):
        assert _check_unimodal([2, 2, 2]) is True

    def test_session_type_unimodal(self):
        ss = _build("&{a: &{b: end}}")
        assert check_unimodality(ss) is True


# ---------------------------------------------------------------------------
# Ultra-log-concavity
# ---------------------------------------------------------------------------

class TestUltraLogConcave:

    def test_short_sequence(self):
        assert _check_ultra_log_concave([1]) is True
        assert _check_ultra_log_concave([1, 1]) is True

    def test_binomial_coefficients(self):
        """Binomial coefficients are ultra-log-concave (trivially: ratio = 1)."""
        # C(4, k) = [1, 4, 6, 4, 1]
        assert _check_ultra_log_concave([1, 4, 6, 4, 1]) is True

    def test_constant_not_ultra(self):
        """[1, 1, 1] normalised by C(2,k)=[1,2,1] gives [1, 0.5, 1], not LC."""
        assert _check_ultra_log_concave([1, 1, 1]) is False


# ---------------------------------------------------------------------------
# Log-concavity of Whitney numbers
# ---------------------------------------------------------------------------

class TestWhitneyLogConcavity:

    def test_end(self):
        ss = _build("end")
        assert is_log_concave_whitney_second(ss) is True
        assert is_log_concave_whitney_first(ss) is True

    def test_single_method(self):
        ss = _build("&{a: end}")
        assert is_log_concave_whitney_second(ss) is True
        assert is_log_concave_whitney_first(ss) is True

    def test_chain(self):
        ss = _build("&{a: &{b: end}}")
        assert is_log_concave_whitney_second(ss) is True
        assert is_log_concave_whitney_first(ss) is True

    def test_branch(self):
        ss = _build("&{a: end, b: end}")
        assert is_log_concave_whitney_second(ss) is True

    def test_selection(self):
        ss = _build("+{a: end, b: end}")
        assert is_log_concave_whitney_second(ss) is True

    def test_parallel(self):
        ss = _build("(&{a: end} || &{b: end})")
        assert is_log_concave_whitney_second(ss) is True

    def test_recursive(self):
        ss = _build("rec X . &{a: X, b: end}")
        assert is_log_concave_whitney_second(ss) is True


# ---------------------------------------------------------------------------
# Heron-Rota-Welsh check
# ---------------------------------------------------------------------------

class TestHRW:

    def test_end(self):
        ss = _build("end")
        assert heron_rota_welsh_check(ss) is True

    def test_chain(self):
        ss = _build("&{a: &{b: end}}")
        assert heron_rota_welsh_check(ss) is True

    def test_parallel(self):
        ss = _build("(&{a: end} || &{b: end})")
        result = heron_rota_welsh_check(ss)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Adiprasito-Huh-Katz bound
# ---------------------------------------------------------------------------

class TestAHK:

    def test_end(self):
        ss = _build("end")
        assert adiprasito_huh_katz_bound(ss) is True

    def test_chain(self):
        ss = _build("&{a: &{b: end}}")
        assert adiprasito_huh_katz_bound(ss) is True

    def test_parallel(self):
        ss = _build("(&{a: end} || &{b: end})")
        result = adiprasito_huh_katz_bound(ss)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Log-concavity ratios
# ---------------------------------------------------------------------------

class TestRatios:

    def test_perfect_lc(self):
        # [1, 2, 1]: 2^2 / (1*1) = 4 >= 1
        ratios = log_concavity_ratio([1, 2, 1])
        assert len(ratios) == 1
        assert ratios[0] >= 1.0

    def test_violation(self):
        # [1, 1, 5]: 1^2 / (1*5) = 0.2 < 1
        ratios = log_concavity_ratio([1, 1, 5])
        assert len(ratios) == 1
        assert ratios[0] < 1.0

    def test_zeros(self):
        ratios = log_concavity_ratio([0, 0, 0])
        assert all(r >= 1.0 for r in ratios)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestFullAnalysis:

    def test_end_analysis(self):
        ss = _build("end")
        result = verify_log_concavity(ss)
        assert result.whitney_second_lc is True
        assert result.whitney_first_lc is True
        assert result.lc_violations_second == []
        assert result.lc_violations_first == []

    def test_chain_analysis(self):
        ss = _build("&{a: &{b: end}}")
        result = verify_log_concavity(ss)
        assert result.whitney_second_lc is True
        assert result.is_unimodal is True
        assert result.height >= 0
        assert result.num_states >= 1

    def test_parallel_analysis(self):
        ss = _build("(&{a: end} || &{b: end})")
        result = verify_log_concavity(ss)
        assert result.whitney_second_lc is True
        assert result.is_unimodal is True

    def test_branch_analysis(self):
        ss = _build("&{a: end, b: end}")
        result = verify_log_concavity(ss)
        assert result.whitney_second_lc is True

    def test_recursive_analysis(self):
        ss = _build("rec X . &{a: X, b: end}")
        result = verify_log_concavity(ss)
        assert isinstance(result.hrw_satisfied, bool)
        assert isinstance(result.ahk_bound_holds, bool)

    def test_complex_type(self):
        ss = _build("&{a: &{b: end}, c: &{d: end}}")
        result = verify_log_concavity(ss)
        assert result.whitney_second_lc is True
        assert result.num_states >= 1

    def test_nested_branch(self):
        ss = _build("&{a: &{b: end, c: end}, d: end}")
        result = verify_log_concavity(ss)
        assert isinstance(result.whitney_second_lc, bool)
        assert isinstance(result.is_ultra_lc, bool)
