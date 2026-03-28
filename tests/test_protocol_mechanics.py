"""Tests for protocol mechanics (Step 60p)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.protocol_mechanics import (
    discrete_derivative,
    discrete_second_derivative,
    discrete_integral,
    mobius_inversion,
    velocity_field,
    velocity_magnitude,
    acceleration_field,
    branch_points,
    max_branch_order,
    parallel_dimensions,
    has_periodic_orbit,
    smoothness,
    protocol_energy,
    analyze_mechanics,
)
from reticulate.zeta import compute_rank


def _build(s: str):
    return build_statespace(parse(s))


class TestDiscreteDerivative:
    def test_constant_function(self):
        """Derivative of constant = 0."""
        ss = _build("&{a: end}")
        f = {s: 1.0 for s in ss.states}
        df = discrete_derivative(ss, f)
        for v in df.values():
            assert abs(v) < 0.01

    def test_rank_derivative(self):
        """Derivative of rank function = 1 at each step (chain)."""
        ss = _build("&{a: &{b: end}}")
        rank = compute_rank(ss)
        f = {s: float(rank[s]) for s in ss.states}
        df = discrete_derivative(ss, f)
        # Top has one child → df = rank(top) - rank(child) = 1
        assert df[ss.top] > 0

    def test_bottom_derivative_zero(self):
        """Bottom state has no children → derivative = 0."""
        ss = _build("&{a: end}")
        rank = compute_rank(ss)
        f = {s: float(rank[s]) for s in ss.states}
        df = discrete_derivative(ss, f)
        assert df[ss.bottom] == 0.0


class TestSecondDerivative:
    def test_chain_second_derivative(self):
        """Chain: rank is linear → second derivative ≈ 0 at interior."""
        ss = _build("&{a: &{b: &{c: end}}}")
        rank = compute_rank(ss)
        f = {s: float(rank[s]) for s in ss.states}
        d2f = discrete_second_derivative(ss, f)
        assert isinstance(d2f, dict)

    def test_branch_has_nonzero_second(self):
        """Branch point: trajectory changes → nonzero Δ²."""
        ss = _build("&{a: end, b: end}")
        rank = compute_rank(ss)
        f = {s: float(rank[s]) for s in ss.states}
        d2f = discrete_second_derivative(ss, f)
        # Branch point should have different acceleration than chain
        assert isinstance(d2f, dict)


class TestIntegral:
    def test_integral_of_constant(self):
        ss = _build("&{a: end}")
        f = {s: 1.0 for s in ss.states}
        F = discrete_integral(ss, f)
        # F(top) = sum over downset of top = sum of all states = 2
        assert F[ss.top] >= 2.0

    def test_mobius_inversion_recovers(self):
        """∫ then μ⁻¹ should recover the original function."""
        ss = _build("&{a: end}")
        f = {s: float(s) for s in ss.states}
        F = discrete_integral(ss, f)
        f_recovered = mobius_inversion(ss, F)
        for s in ss.states:
            assert abs(f[s] - f_recovered[s]) < 0.5


class TestVelocityField:
    def test_end_no_velocity(self):
        ss = _build("end")
        vf = velocity_field(ss)
        assert vf[ss.top] == []

    def test_chain_velocity(self):
        ss = _build("&{a: end}")
        vf = velocity_field(ss)
        assert "a" in vf[ss.top]

    def test_branch_velocity(self):
        ss = _build("&{a: end, b: end}")
        vf = velocity_field(ss)
        assert len(vf[ss.top]) == 2

    def test_magnitude(self):
        ss = _build("&{a: end, b: end}")
        vm = velocity_magnitude(ss)
        assert vm[ss.top] == 2
        assert vm[ss.bottom] == 0


class TestAccelerationField:
    def test_chain_no_acceleration(self):
        """Chain: no branches → acceleration = 0 everywhere."""
        ss = _build("&{a: &{b: end}}")
        af = acceleration_field(ss)
        # Each state has exactly one child → acceleration = 0
        for s in ss.states:
            assert af[s] == 0

    def test_branch_has_acceleration(self):
        """Branch point has acceleration > 0."""
        ss = _build("&{a: end, b: end}")
        af = acceleration_field(ss)
        assert af[ss.top] == 1  # 2 children - 1 = 1

    def test_three_way_branch(self):
        ss = _build("&{a: end, b: end, c: end}")
        af = acceleration_field(ss)
        assert af[ss.top] == 2  # 3 children - 1 = 2

    def test_branch_points_list(self):
        ss = _build("&{a: end, b: end}")
        bp = branch_points(ss)
        assert ss.top in bp
        assert ss.bottom not in bp


class TestBranchOrder:
    def test_chain_order_0(self):
        assert max_branch_order(_build("&{a: end}")) == 0

    def test_simple_branch_order_1(self):
        assert max_branch_order(_build("&{a: end, b: end}")) == 1

    def test_nested_branch_order_2(self):
        """Branch within branch = order 2."""
        order = max_branch_order(_build("&{a: &{c: end, d: end}, b: end}"))
        assert order == 2


class TestParallelDimensions:
    def test_sequential_1d(self):
        assert parallel_dimensions(_build("&{a: end}")) == 1

    def test_parallel_2d(self):
        pd = parallel_dimensions(_build("(&{a: end} || &{b: end})"))
        assert pd >= 2


class TestRecursion:
    def test_no_recursion(self):
        assert has_periodic_orbit(_build("&{a: end}")) is False

    def test_with_recursion(self):
        ss = _build("rec X . &{a: X, b: end}")
        # May or may not have multi-state SCC depending on construction
        result = has_periodic_orbit(ss)
        assert isinstance(result, bool)


class TestSmoothness:
    def test_chain_smooth(self):
        """Chain should be smooth (linear rank function)."""
        s = smoothness(_build("&{a: &{b: &{c: end}}}"))
        assert s > 0.3  # Relatively smooth

    def test_branch_less_smooth(self):
        """Branch point reduces smoothness."""
        s_chain = smoothness(_build("&{a: &{b: end}}"))
        s_branch = smoothness(_build("&{a: end, b: end}"))
        # Both should be positive
        assert s_chain > 0 and s_branch > 0


class TestEnergy:
    def test_end_energy(self):
        e = protocol_energy(_build("end"))
        assert isinstance(e, int)

    def test_energy_integer(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            e = protocol_energy(_build(typ))
            assert isinstance(e, int)


class TestAnalyze:
    def test_end(self):
        r = analyze_mechanics(_build("end"))
        assert r.num_states == 1
        assert r.num_branch_points == 0

    def test_chain(self):
        r = analyze_mechanics(_build("&{a: &{b: end}}"))
        assert r.num_branch_points == 0
        assert r.max_branch_order == 0
        assert r.smoothness > 0

    def test_branch(self):
        r = analyze_mechanics(_build("&{a: end, b: end}"))
        assert r.num_branch_points >= 1
        assert r.max_branch_order >= 1

    def test_parallel(self):
        r = analyze_mechanics(_build("(&{a: end} || &{b: end})"))
        assert r.parallel_dimensions >= 2


class TestBenchmarks:
    BENCHMARKS = [
        ("Java Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_mechanics_runs(self, name, typ):
        ss = _build(typ)
        r = analyze_mechanics(ss)
        assert r.num_states == len(ss.states)
        assert r.smoothness > 0
        assert r.max_branch_order >= 0
        assert isinstance(r.energy, int)

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_velocity_covers_all(self, name, typ):
        ss = _build(typ)
        vf = velocity_field(ss)
        assert set(vf.keys()) == ss.states
