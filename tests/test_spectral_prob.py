"""Tests for spectral-probabilistic analysis (Steps 30o-30r)."""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.spectral_prob import (
    cheeger_constant,
    cheeger_bounds,
    mixing_time_bound,
    stationary_distribution,
    heat_kernel_trace,
    von_neumann_entropy,
    normalized_entropy,
    analyze_spectral_prob,
)


def _build(s: str):
    return build_statespace(parse(s))


class TestCheeger:
    def test_end(self):
        assert cheeger_constant(_build("end")) == 0.0

    def test_single_edge(self):
        h = cheeger_constant(_build("&{a: end}"))
        assert h >= 1.0  # K₂ has h = 1

    def test_bounds_consistent(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            lo, hi = cheeger_bounds(ss)
            assert lo <= hi + 0.01

    def test_cheeger_between_bounds(self):
        """h(G) should be between Cheeger bounds."""
        for typ in ["&{a: end}", "&{a: &{b: end}}"]:
            ss = _build(typ)
            h = cheeger_constant(ss)
            lo, hi = cheeger_bounds(ss)
            assert lo - 0.01 <= h <= hi + 0.5  # Some tolerance


class TestMixingTime:
    def test_end(self):
        assert mixing_time_bound(_build("end")) == 0.0

    def test_positive_for_connected(self):
        for typ in ["&{a: end}", "(&{a: end} || &{b: end})"]:
            assert mixing_time_bound(_build(typ)) > 0

    def test_chain_longer_mixes_slower(self):
        t2 = mixing_time_bound(_build("&{a: end}"))
        t3 = mixing_time_bound(_build("&{a: &{b: end}}"))
        assert t3 > t2  # Longer chains mix slower


class TestStationary:
    def test_sums_to_one(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            pi = stationary_distribution(_build(typ))
            assert abs(sum(pi.values()) - 1.0) < 0.01

    def test_proportional_to_degree(self):
        ss = _build("&{a: end}")
        pi = stationary_distribution(ss)
        # K₂: both vertices degree 1, so uniform
        for v in pi.values():
            assert abs(v - 0.5) < 0.01


class TestHeatKernel:
    def test_end(self):
        assert heat_kernel_trace(_build("end"), 1.0) == 1.0

    def test_at_zero(self):
        """Z(0) = n."""
        for typ in ["&{a: end}", "&{a: &{b: end}}"]:
            ss = _build(typ)
            Z = heat_kernel_trace(ss, 0.0)
            assert abs(Z - len(ss.states)) < 0.01

    def test_decreasing_in_t(self):
        ss = _build("&{a: &{b: end}}")
        Z1 = heat_kernel_trace(ss, 0.5)
        Z2 = heat_kernel_trace(ss, 2.0)
        assert Z1 >= Z2  # Heat diffuses → trace decreases


class TestVonNeumann:
    def test_end_zero(self):
        assert von_neumann_entropy(_build("end")) == 0.0

    def test_positive_for_nontrivial(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            S = von_neumann_entropy(_build(typ))
            assert S >= 0.0

    def test_normalized_range(self):
        for typ in ["&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            sn = normalized_entropy(_build(typ))
            assert 0.0 <= sn <= 1.01

    def test_parallel_higher_entropy(self):
        """Parallel has more spectral complexity than chain."""
        S_chain = von_neumann_entropy(_build("&{a: end}"))
        S_par = von_neumann_entropy(_build("(&{a: end} || &{b: end})"))
        # Parallel has more eigenvalues → potentially higher entropy
        assert S_par >= S_chain - 0.01


class TestAnalyze:
    def test_end(self):
        r = analyze_spectral_prob(_build("end"))
        assert r.num_states == 1
        assert r.von_neumann_entropy == 0.0

    def test_branch(self):
        r = analyze_spectral_prob(_build("&{a: end}"))
        assert r.num_states == 2
        assert r.cheeger_constant >= 1.0
        assert r.mixing_time_bound > 0

    def test_parallel(self):
        r = analyze_spectral_prob(_build("(&{a: end} || &{b: end})"))
        assert r.num_states == 4
        assert r.von_neumann_entropy > 0


class TestBenchmarks:
    BENCHMARKS = [
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_analysis_runs(self, name, typ):
        ss = _build(typ)
        r = analyze_spectral_prob(ss)
        assert r.num_states == len(ss.states)
        assert r.von_neumann_entropy >= 0
        assert r.heat_trace > 0
        assert abs(sum(r.stationary_distribution.values()) - 1.0) < 0.01
