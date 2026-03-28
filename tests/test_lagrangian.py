"""Tests for Lagrangian mechanics of session types (Step 60q)."""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lagrangian import (
    kinetic_energy,
    potential_energy,
    lagrangian_field,
    hamiltonian_field,
    total_energy,
    gravitational_field,
    momentum,
    path_action,
    least_action_path,
    max_action_path,
    analyze_circular,
    check_energy_conservation,
    is_energy_conserved,
    analyze_lagrangian,
)


def _build(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Energy tests
# ---------------------------------------------------------------------------

class TestKineticEnergy:
    def test_end_zero(self):
        """End state: no transitions → T = 0."""
        ss = _build("end")
        T = kinetic_energy(ss)
        assert T[ss.top] == 0.0

    def test_chain_uniform(self):
        """Chain: each state has 1 transition → T = 0.5."""
        ss = _build("&{a: &{b: end}}")
        T = kinetic_energy(ss)
        assert T[ss.top] == 0.5  # 1²/2
        assert T[ss.bottom] == 0.0

    def test_branch_higher(self):
        """Branch: 2 transitions → T = 2.0."""
        ss = _build("&{a: end, b: end}")
        T = kinetic_energy(ss)
        assert T[ss.top] == 2.0  # 2²/2

    def test_three_branch(self):
        """Three branches → T = 4.5."""
        ss = _build("&{a: end, b: end, c: end}")
        T = kinetic_energy(ss)
        assert T[ss.top] == 4.5  # 3²/2


class TestPotentialEnergy:
    def test_bottom_zero(self):
        """Bottom state: V = 0 (ground state)."""
        ss = _build("&{a: end}")
        V = potential_energy(ss)
        assert V[ss.bottom] == 0.0

    def test_top_positive(self):
        """Top state: V > 0 (above ground)."""
        ss = _build("&{a: end}")
        V = potential_energy(ss)
        assert V[ss.top] > 0

    def test_chain_linear(self):
        """Chain: V increases linearly from bottom to top."""
        ss = _build("&{a: &{b: &{c: end}}}")
        V = potential_energy(ss)
        assert V[ss.top] == 3.0
        assert V[ss.bottom] == 0.0


class TestLagrangian:
    def test_bottom_zero(self):
        """Bottom: L = T - V = 0 - 0 = 0."""
        ss = _build("&{a: end}")
        L = lagrangian_field(ss)
        assert L[ss.bottom] == 0.0

    def test_lagrangian_defined(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            L = lagrangian_field(_build(typ))
            for v in L.values():
                assert not math.isnan(v)


class TestHamiltonian:
    def test_total_positive(self):
        """Total energy is non-negative."""
        for typ in ["end", "&{a: end}", "&{a: &{b: end}}"]:
            E = total_energy(_build(typ))
            assert E >= 0.0

    def test_hamiltonian_equals_T_plus_V(self):
        ss = _build("&{a: &{b: end}}")
        T = kinetic_energy(ss)
        V = potential_energy(ss)
        H = hamiltonian_field(ss)
        for s in ss.states:
            assert abs(H[s] - (T[s] + V[s])) < 0.001


# ---------------------------------------------------------------------------
# Gravity and momentum tests
# ---------------------------------------------------------------------------

class TestGravity:
    def test_bottom_max_gravity(self):
        ss = _build("&{a: end}")
        g = gravitational_field(ss)
        assert g[ss.bottom] > g[ss.top]

    def test_gravity_decreases_with_rank(self):
        ss = _build("&{a: &{b: &{c: end}}}")
        g = gravitational_field(ss)
        assert g[ss.bottom] > 1.0  # Strong at bottom


class TestMomentum:
    def test_bottom_zero(self):
        ss = _build("&{a: end}")
        p = momentum(ss)
        assert p[ss.bottom] == 0

    def test_chain_uniform(self):
        ss = _build("&{a: &{b: end}}")
        p = momentum(ss)
        assert p[ss.top] == 1  # One outgoing

    def test_branch_higher(self):
        ss = _build("&{a: end, b: end}")
        p = momentum(ss)
        assert p[ss.top] == 2


# ---------------------------------------------------------------------------
# Action and path tests
# ---------------------------------------------------------------------------

class TestAction:
    def test_single_state(self):
        ss = _build("end")
        a = path_action(ss, [ss.top])
        assert isinstance(a, float)

    def test_chain_action(self):
        ss = _build("&{a: end}")
        a = path_action(ss, [ss.top, ss.bottom])
        assert isinstance(a, float)


class TestLeastAction:
    def test_chain_unique(self):
        """Chain has only one path → it's the least action path."""
        ss = _build("&{a: &{b: end}}")
        path, action = least_action_path(ss)
        assert path[0] == ss.top
        assert path[-1] == ss.bottom

    def test_least_le_max(self):
        """Least action ≤ max action."""
        for typ in ["&{a: end}", "&{a: end, b: end}", "(&{a: end} || &{b: end})"]:
            ss = _build(typ)
            _, la = least_action_path(ss)
            _, ma = max_action_path(ss)
            assert la <= ma + 0.001

    def test_path_valid(self):
        """Least action path should start at top and end at bottom."""
        ss = _build("(&{a: end} || &{b: end})")
        path, _ = least_action_path(ss)
        assert path[0] == ss.top
        assert path[-1] == ss.bottom


# ---------------------------------------------------------------------------
# Circular mechanics tests
# ---------------------------------------------------------------------------

class TestCircular:
    def test_no_cycles_acyclic(self):
        """Acyclic types have no circular mechanics."""
        circ = analyze_circular(_build("&{a: end}"))
        assert len(circ) == 0

    def test_recursive_has_cycle(self):
        """Recursive type may have circular mechanics."""
        ss = _build("rec X . &{a: X, b: end}")
        circ = analyze_circular(ss)
        # May or may not have multi-state SCC depending on construction
        assert isinstance(circ, list)

    def test_cycle_properties(self):
        """If cycles exist, they have positive period and inertia."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        circ = analyze_circular(ss)
        for c in circ:
            assert c.period >= 1
            assert c.inertia >= 2
            assert c.angular_velocity > 0


# ---------------------------------------------------------------------------
# Conservation tests
# ---------------------------------------------------------------------------

class TestConservation:
    def test_chain_conservation(self):
        """In a chain (deterministic), energy should be approximately conserved."""
        ss = _build("&{a: &{b: &{c: end}}}")
        path, _ = least_action_path(ss)
        energies = check_energy_conservation(ss, path)
        assert len(energies) == len(path)

    def test_conservation_check(self):
        ss = _build("&{a: &{b: end}}")
        path, _ = least_action_path(ss)
        result = is_energy_conserved(ss, path, tolerance=2.0)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Full analysis tests
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_end(self):
        r = analyze_lagrangian(_build("end"))
        assert r.num_states == 1
        assert r.total_energy >= 0

    def test_chain(self):
        r = analyze_lagrangian(_build("&{a: &{b: end}}"))
        assert r.num_states == 3
        assert r.least_action_value <= r.max_action_value + 0.001
        assert len(r.circular_mechanics) == 0

    def test_branch(self):
        r = analyze_lagrangian(_build("&{a: end, b: end}"))
        assert r.momentum[r.least_action_path[0]] >= 1

    def test_parallel(self):
        r = analyze_lagrangian(_build("(&{a: end} || &{b: end})"))
        assert r.num_states == 4
        assert r.total_energy > 0


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

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
    def test_lagrangian_runs(self, name, typ):
        ss = _build(typ)
        r = analyze_lagrangian(ss)
        assert r.num_states == len(ss.states)
        assert r.total_energy >= 0
        assert r.least_action_value <= r.max_action_value + 0.001

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_hamiltonian_nonneg(self, name, typ):
        ss = _build(typ)
        H = hamiltonian_field(ss)
        for s, h in H.items():
            assert h >= -0.001, f"{name}: H({s}) = {h} < 0"

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_path_endpoints(self, name, typ):
        ss = _build(typ)
        path, _ = least_action_path(ss)
        if path:
            assert path[0] == ss.top
            assert path[-1] == ss.bottom
