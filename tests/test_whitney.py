"""Tests for Whitney number analysis (Step 30g).

Tests cover:
- Rank profiles and Whitney numbers
- Unimodality and log-concavity
- Rank symmetry
- Sperner property
- Convolution under parallel
- Benchmark protocols
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.whitney import (
    rank_profile,
    max_rank_level,
    is_unimodal,
    is_rank_symmetric,
    sperner_width,
    is_sperner,
    convolve_profiles,
    verify_profile_convolution,
    analyze_whitney,
    check_log_concave,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Rank profile tests
# ---------------------------------------------------------------------------

class TestRankProfile:

    def test_end(self):
        """end: profile = [1]."""
        assert rank_profile(_build("end")) == [1]

    def test_chain_2(self):
        """&{a: end}: profile = [1, 1]."""
        prof = rank_profile(_build("&{a: end}"))
        assert prof == [1, 1]

    def test_chain_3(self):
        """&{a: &{b: end}}: profile = [1, 1, 1]."""
        prof = rank_profile(_build("&{a: &{b: end}}"))
        assert prof == [1, 1, 1]

    def test_parallel(self):
        """(&{a: end} || &{b: end}): product of two [1,1] = [1,2,1]."""
        prof = rank_profile(_build("(&{a: end} || &{b: end})"))
        assert prof == [1, 2, 1]

    def test_sum_equals_total(self):
        """Sum of profile = number of quotient states."""
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            prof = rank_profile(ss)
            from reticulate.zeta import _compute_sccs
            scc_map, _ = _compute_sccs(ss)
            n_quotient = len(set(scc_map.values()))
            assert sum(prof) == n_quotient


# ---------------------------------------------------------------------------
# Max rank level tests
# ---------------------------------------------------------------------------

class TestMaxRankLevel:

    def test_chain(self):
        """Chain: all levels have 1 element."""
        k, W = max_rank_level(_build("&{a: end}"))
        assert W == 1

    def test_parallel(self):
        """Parallel: middle level has 2 elements."""
        k, W = max_rank_level(_build("(&{a: end} || &{b: end})"))
        assert W == 2
        assert k == 1  # Middle level


# ---------------------------------------------------------------------------
# Unimodality tests
# ---------------------------------------------------------------------------

class TestUnimodality:

    def test_empty(self):
        assert is_unimodal([]) is True

    def test_single(self):
        assert is_unimodal([5]) is True

    def test_increasing(self):
        assert is_unimodal([1, 2, 3]) is True

    def test_decreasing(self):
        assert is_unimodal([3, 2, 1]) is True

    def test_peak(self):
        assert is_unimodal([1, 3, 2]) is True

    def test_valley(self):
        assert is_unimodal([3, 1, 2]) is False

    def test_constant(self):
        assert is_unimodal([2, 2, 2]) is True

    def test_chain_unimodal(self):
        prof = rank_profile(_build("&{a: &{b: end}}"))
        assert is_unimodal(prof)

    def test_parallel_unimodal(self):
        prof = rank_profile(_build("(&{a: end} || &{b: end})"))
        assert is_unimodal(prof)  # [1, 2, 1] is unimodal


# ---------------------------------------------------------------------------
# Log-concavity tests
# ---------------------------------------------------------------------------

class TestLogConcavity:

    def test_chain_log_concave(self):
        prof = rank_profile(_build("&{a: &{b: end}}"))
        assert check_log_concave(prof)

    def test_parallel_log_concave(self):
        prof = rank_profile(_build("(&{a: end} || &{b: end})"))
        assert check_log_concave(prof)  # [1, 2, 1]: 2² ≥ 1·1 ✓


# ---------------------------------------------------------------------------
# Rank symmetry tests
# ---------------------------------------------------------------------------

class TestRankSymmetry:

    def test_chain_symmetric(self):
        """Chains are symmetric: [1, 1, ..., 1]."""
        assert is_rank_symmetric(_build("&{a: end}"))
        assert is_rank_symmetric(_build("&{a: &{b: end}}"))

    def test_parallel_symmetric(self):
        """Product of chains is symmetric: [1, 2, 1]."""
        assert is_rank_symmetric(_build("(&{a: end} || &{b: end})"))


# ---------------------------------------------------------------------------
# Sperner property tests
# ---------------------------------------------------------------------------

class TestSperner:

    def test_chain_sperner(self):
        """Chain has Sperner property (width 1 = max W_k = 1)."""
        assert is_sperner(_build("&{a: end}"))

    def test_parallel_sperner(self):
        """Boolean lattice has Sperner property."""
        assert is_sperner(_build("(&{a: end} || &{b: end})"))

    def test_sperner_width_chain(self):
        assert sperner_width(_build("&{a: end}")) == 1

    def test_sperner_width_parallel(self):
        assert sperner_width(_build("(&{a: end} || &{b: end})")) == 2


# ---------------------------------------------------------------------------
# Convolution tests
# ---------------------------------------------------------------------------

class TestConvolution:

    def test_trivial(self):
        assert convolve_profiles([1], [1]) == [1]

    def test_chain_product(self):
        """[1,1] * [1,1] = [1,2,1]."""
        assert convolve_profiles([1, 1], [1, 1]) == [1, 2, 1]

    def test_parallel_convolution(self):
        """Product lattice profile = convolution of factor profiles."""
        ss1 = _build("&{a: end}")
        ss2 = _build("&{b: end}")
        ss_prod = _build("(&{a: end} || &{b: end})")
        assert verify_profile_convolution(ss1, ss2, ss_prod)


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyze:

    def test_end(self):
        r = analyze_whitney(_build("end"))
        assert r.height == 0
        assert r.rank_profile == [1]
        assert r.is_unimodal is True

    def test_chain(self):
        r = analyze_whitney(_build("&{a: end}"))
        assert r.height == 1
        assert r.rank_profile == [1, 1]
        assert r.is_rank_symmetric is True

    def test_parallel(self):
        r = analyze_whitney(_build("(&{a: end} || &{b: end})"))
        assert r.height == 2
        assert r.rank_profile == [1, 2, 1]
        assert r.is_unimodal is True
        assert r.is_log_concave_second is True
        assert r.is_rank_symmetric is True
        assert r.is_sperner is True


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmarks:
    BENCHMARKS = [
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("Simple SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_whitney_properties(self, name, typ):
        """Whitney analysis runs correctly for all benchmarks."""
        ss = _build(typ)
        r = analyze_whitney(ss)
        assert r.height >= 0
        assert len(r.rank_profile) == r.height + 1
        assert sum(r.rank_profile) == r.total_elements
        assert r.sperner_width >= 1

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_unimodality(self, name, typ):
        """All benchmarks should have unimodal rank profiles."""
        ss = _build(typ)
        prof = rank_profile(ss)
        assert is_unimodal(prof), f"{name}: profile {prof} not unimodal"

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_log_concavity(self, name, typ):
        """All benchmarks should have log-concave rank profiles."""
        ss = _build(typ)
        prof = rank_profile(ss)
        assert check_log_concave(prof), f"{name}: profile {prof} not log-concave"
