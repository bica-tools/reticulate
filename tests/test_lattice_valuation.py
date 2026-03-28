"""Tests for lattice valuations and FKG inequality (Steps 30y-30z)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice_valuation import (
    rank_valuation,
    height_valuation,
    is_modular_valuation,
    is_distributive_lattice,
    verify_fkg,
    analyze_valuations,
)


def _build(s: str):
    return build_statespace(parse(s))


class TestRankValuation:
    def test_end(self):
        v = rank_valuation(_build("end"))
        assert len(v) == 1

    def test_chain(self):
        ss = _build("&{a: end}")
        v = rank_valuation(ss)
        assert v[ss.bottom] == 0
        assert v[ss.top] >= 1

    def test_bottom_always_zero(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            v = rank_valuation(ss)
            assert v[ss.bottom] == 0


class TestHeightValuation:
    def test_top_always_zero(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            v = height_valuation(ss)
            assert v[ss.top] == 0

    def test_chain(self):
        ss = _build("&{a: &{b: end}}")
        v = height_valuation(ss)
        assert v[ss.top] == 0
        assert v[ss.bottom] >= 1


class TestModularity:
    def test_chain_modular(self):
        """Rank is always modular on chains (trivially)."""
        ss = _build("&{a: end}")
        v = rank_valuation(ss)
        assert is_modular_valuation(ss, v)

    def test_parallel_rank(self):
        ss = _build("(&{a: end} || &{b: end})")
        v = rank_valuation(ss)
        result = is_modular_valuation(ss, v)
        assert isinstance(result, bool)


class TestFKG:
    def test_distributive(self):
        """Most session type lattices are distributive."""
        for typ in ["&{a: end}", "&{a: &{b: end}}"]:
            assert is_distributive_lattice(_build(typ))

    def test_fkg_monotone(self):
        """FKG holds for rank as both f and g."""
        ss = _build("&{a: &{b: end}}")
        v = rank_valuation(ss)
        f = lambda s: float(v[s])
        g = lambda s: float(v[s])
        assert verify_fkg(ss, f, g)

    def test_fkg_constant(self):
        """FKG holds trivially for constant functions."""
        ss = _build("&{a: end}")
        assert verify_fkg(ss, lambda s: 1.0, lambda s: 1.0)


class TestAnalyze:
    def test_end(self):
        r = analyze_valuations(_build("end"))
        assert r.num_states == 1

    def test_chain(self):
        r = analyze_valuations(_build("&{a: end}"))
        assert r.num_states == 2
        assert r.fkg_applicable is True

    def test_parallel(self):
        r = analyze_valuations(_build("(&{a: end} || &{b: end})"))
        assert r.num_states == 4


class TestBenchmarks:
    BENCHMARKS = [
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_valuations_run(self, name, typ):
        ss = _build(typ)
        r = analyze_valuations(ss)
        assert r.num_states >= 1
        assert len(r.rank_valuation) == len(ss.states)
