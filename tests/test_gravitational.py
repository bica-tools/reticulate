"""Tests for the gravitational session type analyzer (Step 59l)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.gravitational import (
    GravitationalField,
    analyze_gravity,
    detect_black_holes,
    detect_orbits,
    escape_velocity,
    gravitational_force,
    gravitational_potential,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nim3_ss():
    return build_statespace(
        parse("+{take1: &{take1: +{take1: end}, take2: end}, take2: &{take1: end}, take3: end}")
    )


@pytest.fixture
def hex2x2_ss():
    return build_statespace(
        parse(
            "+{a: &{b: +{c: end, d: &{c: end}}, "
            "c: +{b: &{d: end}, d: &{b: end}}, "
            "d: +{b: &{c: end}, c: end}}, "
            "b: &{a: +{c: end, d: end}, "
            "c: +{a: &{d: end}, d: end}, "
            "d: +{a: &{c: end}, c: end}}, "
            "c: &{a: +{b: end, d: &{b: end}}, "
            "b: +{a: end, d: &{a: end}}, "
            "d: +{a: end, b: end}}, "
            "d: &{a: +{b: end, c: &{b: end}}, "
            "b: +{a: &{c: end}, c: &{a: end}}, "
            "c: +{a: &{b: end}, b: end}}}"
        )
    )


@pytest.fixture
def recursive_ss():
    """rec X . &{request: +{response: X}} — infinite orbit."""
    return build_statespace(parse("rec X . &{request: +{response: X}}"))


@pytest.fixture
def iterator_ss():
    """rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}} — orbit with exit."""
    return build_statespace(
        parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
    )


@pytest.fixture
def dominion_ss():
    return build_statespace(
        parse("(+{play_attack: wait} || &{react: end, pass: end}) . &{resolve: end}")
    )


@pytest.fixture
def end_ss():
    return build_statespace(parse("end"))


@pytest.fixture
def simple_branch_ss():
    return build_statespace(parse("&{a: end, b: end}"))


# ---------------------------------------------------------------------------
# gravitational_potential tests
# ---------------------------------------------------------------------------


class TestPotential:
    def test_bottom_has_zero_potential(self, nim3_ss):
        pot = gravitational_potential(nim3_ss)
        assert pot[nim3_ss.bottom] == 0

    def test_top_has_max_potential(self, nim3_ss):
        pot = gravitational_potential(nim3_ss)
        assert pot[nim3_ss.top] == max(v for v in pot.values() if v >= 0)

    def test_nim3_top_potential_is_3(self, nim3_ss):
        """Longest path in Nim(3) is take1->take1->take1 = 3 steps."""
        pot = gravitational_potential(nim3_ss)
        assert pot[nim3_ss.top] == 3

    def test_nim3_all_potentials_nonneg(self, nim3_ss):
        pot = gravitational_potential(nim3_ss)
        assert all(v >= 0 for v in pot.values())

    def test_end_potential_is_zero(self, end_ss):
        pot = gravitational_potential(end_ss)
        assert pot[end_ss.top] == 0

    def test_simple_branch_potential(self, simple_branch_ss):
        pot = gravitational_potential(simple_branch_ss)
        assert pot[simple_branch_ss.top] == 1
        assert pot[simple_branch_ss.bottom] == 0

    def test_hex2x2_top_potential(self, hex2x2_ss):
        """Hex 2x2 has longest path of 4 (all cells filled)."""
        pot = gravitational_potential(hex2x2_ss)
        assert pot[hex2x2_ss.top] == 4

    def test_recursive_potential(self, iterator_ss):
        """Iterator has exit path, so all states should have finite potential."""
        pot = gravitational_potential(iterator_ss)
        assert pot[iterator_ss.bottom] == 0
        # At least some states have positive potential
        assert any(v > 0 for v in pot.values())


# ---------------------------------------------------------------------------
# escape_velocity tests
# ---------------------------------------------------------------------------


class TestEscapeVelocity:
    def test_bottom_escape_zero(self, nim3_ss):
        esc = escape_velocity(nim3_ss)
        assert esc[nim3_ss.bottom] == 0

    def test_nim3_top_escape_is_1(self, nim3_ss):
        """Top can reach bottom in 1 step (take3)."""
        esc = escape_velocity(nim3_ss)
        assert esc[nim3_ss.top] == 1

    def test_escape_leq_potential(self, nim3_ss):
        """Escape velocity <= potential for all states."""
        pot = gravitational_potential(nim3_ss)
        esc = escape_velocity(nim3_ss)
        for s in nim3_ss.states:
            if pot[s] >= 0 and esc[s] >= 0:
                assert esc[s] <= pot[s]

    def test_end_escape_zero(self, end_ss):
        esc = escape_velocity(end_ss)
        assert esc[end_ss.top] == 0

    def test_hex_escape_leq_potential(self, hex2x2_ss):
        pot = gravitational_potential(hex2x2_ss)
        esc = escape_velocity(hex2x2_ss)
        for s in hex2x2_ss.states:
            if pot[s] >= 0 and esc[s] >= 0:
                assert esc[s] <= pot[s]


# ---------------------------------------------------------------------------
# gravitational_force tests
# ---------------------------------------------------------------------------


class TestForce:
    def test_bottom_has_zero_force(self, nim3_ss):
        force = gravitational_force(nim3_ss)
        assert force[nim3_ss.bottom] == 0

    def test_nim3_top_force_is_3(self, nim3_ss):
        """Top has 3 outgoing transitions (take1, take2, take3)."""
        force = gravitational_force(nim3_ss)
        assert force[nim3_ss.top] == 3

    def test_end_has_zero_force(self, end_ss):
        force = gravitational_force(end_ss)
        assert force[end_ss.top] == 0

    def test_simple_branch_force(self, simple_branch_ss):
        force = gravitational_force(simple_branch_ss)
        assert force[simple_branch_ss.top] == 2

    def test_all_forces_nonneg(self, hex2x2_ss):
        force = gravitational_force(hex2x2_ss)
        assert all(f >= 0 for f in force.values())


# ---------------------------------------------------------------------------
# detect_orbits tests
# ---------------------------------------------------------------------------


class TestOrbits:
    def test_nim3_no_orbits(self, nim3_ss):
        """Nim(3) is acyclic — no orbits."""
        orbits = detect_orbits(nim3_ss)
        assert len(orbits) == 0

    def test_hex2x2_no_orbits(self, hex2x2_ss):
        """Hex(2x2) is acyclic — no orbits."""
        orbits = detect_orbits(hex2x2_ss)
        assert len(orbits) == 0

    def test_recursive_has_orbit(self, recursive_ss):
        """rec X . &{req: +{resp: X}} has a 2-state orbit."""
        orbits = detect_orbits(recursive_ss)
        assert len(orbits) >= 1
        total_orbit_states = sum(len(o) for o in orbits)
        assert total_orbit_states >= 2

    def test_iterator_has_orbit(self, iterator_ss):
        """Iterator has a cycle through hasNext->TRUE->next->hasNext."""
        orbits = detect_orbits(iterator_ss)
        assert len(orbits) >= 1

    def test_end_no_orbits(self, end_ss):
        orbits = detect_orbits(end_ss)
        assert len(orbits) == 0


# ---------------------------------------------------------------------------
# detect_black_holes tests
# ---------------------------------------------------------------------------


class TestBlackHoles:
    def test_nim3_no_black_holes(self, nim3_ss):
        """Well-formed types have no black holes."""
        bh = detect_black_holes(nim3_ss)
        assert len(bh) == 0

    def test_hex2x2_no_black_holes(self, hex2x2_ss):
        bh = detect_black_holes(hex2x2_ss)
        assert len(bh) == 0

    def test_end_no_black_holes(self, end_ss):
        bh = detect_black_holes(end_ss)
        assert len(bh) == 0

    def test_iterator_no_black_holes(self, iterator_ss):
        bh = detect_black_holes(iterator_ss)
        assert len(bh) == 0


# ---------------------------------------------------------------------------
# analyze_gravity tests
# ---------------------------------------------------------------------------


class TestAnalyzeGravity:
    def test_nim3_is_lyapunov(self, nim3_ss):
        """Nim(3) is acyclic — potential strictly decreases = Lyapunov."""
        gf = analyze_gravity(nim3_ss)
        assert gf.is_lyapunov is True

    def test_hex2x2_is_lyapunov(self, hex2x2_ss):
        gf = analyze_gravity(hex2x2_ss)
        assert gf.is_lyapunov is True

    def test_nim3_max_potential(self, nim3_ss):
        gf = analyze_gravity(nim3_ss)
        assert gf.max_potential == 3

    def test_nim3_total_energy(self, nim3_ss):
        """Total energy = sum of potentials: 3+2+1+1+0 = 7."""
        gf = analyze_gravity(nim3_ss)
        assert gf.total_energy == 7

    def test_nim3_no_orbits(self, nim3_ss):
        gf = analyze_gravity(nim3_ss)
        assert len(gf.orbits) == 0

    def test_nim3_no_black_holes(self, nim3_ss):
        gf = analyze_gravity(nim3_ss)
        assert len(gf.black_holes) == 0

    def test_nim3_gradient_positive(self, nim3_ss):
        """All gradients positive in acyclic Lyapunov field."""
        gf = analyze_gravity(nim3_ss)
        for (src, label, tgt), drop in gf.gradient.items():
            assert drop > 0, f"Non-positive gradient: {src}--{label}-->{tgt}: {drop}"

    def test_iterator_has_orbits(self, iterator_ss):
        gf = analyze_gravity(iterator_ss)
        assert len(gf.orbits) >= 1

    def test_kinetic_branching_nonneg(self, nim3_ss):
        gf = analyze_gravity(nim3_ss)
        for s, k in gf.kinetic_branching.items():
            assert k >= 0.0

    def test_nim3_kinetic_at_top(self, nim3_ss):
        """Top has 3 successors with potentials 2, 1, 0. Drops: 1, 2, 3. Avg = 2.0."""
        gf = analyze_gravity(nim3_ss)
        assert gf.kinetic_branching[nim3_ss.top] == 2.0

    def test_result_type(self, nim3_ss):
        gf = analyze_gravity(nim3_ss)
        assert isinstance(gf, GravitationalField)

    def test_end_trivial_field(self, end_ss):
        gf = analyze_gravity(end_ss)
        assert gf.max_potential == 0
        assert gf.total_energy == 0
        assert gf.is_lyapunov is True
        assert len(gf.orbits) == 0
        assert len(gf.black_holes) == 0

    def test_dominion_analysis(self, dominion_ss):
        """Dominion should have valid gravitational analysis."""
        gf = analyze_gravity(dominion_ss)
        assert gf.max_potential > 0
        assert len(gf.black_holes) == 0
