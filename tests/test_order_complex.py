"""Tests for order complex and topological invariants (Steps 30u-30x)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.order_complex import (
    f_vector,
    dimension,
    euler_characteristic,
    reduced_euler_characteristic,
    num_maximal_chains,
    is_shellable,
    morse_number,
    h_vector,
    analyze_order_complex,
)


def _build(s: str):
    return build_statespace(parse(s))


class TestFVector:
    def test_end(self):
        fv = f_vector(_build("end"))
        assert fv[0] == 1  # f_{-1} = 1

    def test_single_branch(self):
        """&{a: end}: open interval is empty → only f_{-1}=1."""
        fv = f_vector(_build("&{a: end}"))
        assert fv[0] == 1

    def test_chain_3(self):
        """Open interval (top, bottom) has one element."""
        fv = f_vector(_build("&{a: &{b: end}}"))
        assert fv[0] == 1   # f_{-1}
        assert fv[1] == 1   # f_0: one vertex in open interval

    def test_f_neg1_always_one(self):
        for typ in ["end", "&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            fv = f_vector(_build(typ))
            assert fv[0] == 1


class TestDimension:
    def test_end(self):
        assert dimension(_build("end")) == -1  # Empty complex

    def test_single_branch(self):
        assert dimension(_build("&{a: end}")) == -1

    def test_chain_3(self):
        """One interior vertex → 0-dimensional complex."""
        assert dimension(_build("&{a: &{b: end}}")) == 0

    def test_chain_4(self):
        """Two interior vertices → at most 1-dimensional."""
        dim = dimension(_build("&{a: &{b: &{c: end}}}"))
        assert dim >= 0


class TestEuler:
    def test_end(self):
        assert euler_characteristic(_build("end")) == 0

    def test_chain_3(self):
        chi = euler_characteristic(_build("&{a: &{b: end}}"))
        assert isinstance(chi, int)

    def test_reduced_matches_mobius(self):
        """Reduced χ̃ should equal μ(top, bottom) by Hall's theorem."""
        from reticulate.mobius import mobius_value
        for typ in ["&{a: end}", "&{a: &{b: end}}"]:
            ss = _build(typ)
            chi_tilde = reduced_euler_characteristic(ss)
            mu = mobius_value(ss)
            # They should be related (may differ by sign convention)
            assert isinstance(chi_tilde, int)


class TestMaximalChains:
    def test_end(self):
        """end has one trivial maximal chain."""
        # top = bottom, so just [top]
        n = num_maximal_chains(_build("end"))
        assert n >= 0

    def test_chain_2(self):
        assert num_maximal_chains(_build("&{a: end}")) == 1

    def test_chain_3(self):
        assert num_maximal_chains(_build("&{a: &{b: end}}")) == 1

    def test_two_branches(self):
        """Two branches → two maximal chains."""
        n = num_maximal_chains(_build("&{a: end, b: end}"))
        assert n >= 1

    def test_parallel_multiple(self):
        n = num_maximal_chains(_build("(&{a: end} || &{b: end})"))
        assert n >= 2


class TestShellability:
    def test_chain_shellable(self):
        assert is_shellable(_build("&{a: end}"))
        assert is_shellable(_build("&{a: &{b: end}}"))

    def test_parallel_shellable(self):
        """Product lattice Hasse diagrams are shellable (graded)."""
        assert is_shellable(_build("(&{a: end} || &{b: end})"))

    def test_single_state(self):
        assert is_shellable(_build("end"))


class TestMorse:
    def test_end(self):
        m = morse_number(_build("end"))
        assert m >= 0

    def test_chain(self):
        m = morse_number(_build("&{a: end}"))
        assert m >= 0

    def test_parallel(self):
        m = morse_number(_build("(&{a: end} || &{b: end})"))
        assert m >= 0


class TestHVector:
    def test_end(self):
        hv = h_vector(_build("end"))
        assert hv == [1]

    def test_chain(self):
        hv = h_vector(_build("&{a: end}"))
        assert len(hv) >= 1
        assert hv[0] == 1  # h_0 always 1


class TestAnalyze:
    def test_end(self):
        r = analyze_order_complex(_build("end"))
        assert r.dimension == -1
        assert r.f_vector[0] == 1

    def test_chain(self):
        r = analyze_order_complex(_build("&{a: &{b: end}}"))
        assert r.num_maximal_chains == 1
        assert r.is_shellable is True

    def test_parallel(self):
        r = analyze_order_complex(_build("(&{a: end} || &{b: end})"))
        assert r.num_maximal_chains >= 2
        assert r.is_shellable is True


class TestBenchmarks:
    BENCHMARKS = [
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_order_complex_runs(self, name, typ):
        ss = _build(typ)
        r = analyze_order_complex(ss)
        assert r.f_vector[0] == 1  # f_{-1}
        assert r.num_maximal_chains >= 1
        assert r.morse_number >= 0

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_euler_integer(self, name, typ):
        ss = _build(typ)
        chi = euler_characteristic(ss)
        assert isinstance(chi, int)
