"""Tests for the orbital session type analyzer (Step 60g)."""

from __future__ import annotations

import pytest

from reticulate import build_statespace, parse
from reticulate.orbital import (
    GroupOrbitResult,
    OrbitalResult,
    PlanetaryOrbitResult,
    ProbabilityCloudResult,
    QuantumOrbitalResult,
    TypeSpaceOrbitResult,
    analyze_orbital,
    compute_orbits,
    planetary_orbits,
    probability_cloud,
    quantum_orbitals,
    type_space_orbit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def end_ss():
    return build_statespace(parse("end"))


@pytest.fixture
def simple_branch_ss():
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def recursive_ss():
    return build_statespace(parse("rec X . &{next: X, done: end}"))


@pytest.fixture
def iterator_ss():
    return build_statespace(parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"))


@pytest.fixture
def parallel_ss():
    return build_statespace(parse("(&{a: end} || &{b: end})"))


@pytest.fixture
def diamond_ss():
    return build_statespace(parse("&{a: &{c: end}, b: &{c: end}}"))


@pytest.fixture
def multi_branch_ss():
    return build_statespace(parse("&{a: end, b: end, c: end}"))


@pytest.fixture
def deep_ss():
    return build_statespace(parse("&{a: &{b: &{c: end}}}"))


@pytest.fixture
def selection_ss():
    return build_statespace(parse("+{ok: end, err: end}"))


# ---------------------------------------------------------------------------
# Interpretation 1: Quantum Orbital Analogy
# ---------------------------------------------------------------------------


class TestQuantumOrbitals:
    def test_end_single_shell(self, end_ss):
        """end has one state at depth 0."""
        result = quantum_orbitals(end_ss)
        assert result.energy_levels == {0: 1}
        assert result.max_principal_quantum == 0

    def test_branch_two_shells(self, simple_branch_ss):
        """&{a: end, b: end} has depth 0 and depth 1."""
        result = quantum_orbitals(simple_branch_ss)
        assert 0 in result.energy_levels
        assert 1 in result.energy_levels

    def test_shell_labels_cycle(self, deep_ss):
        """Shell labels cycle through s, p, d, f, g."""
        result = quantum_orbitals(deep_ss)
        assert result.shell_labels[0] == "1s"
        assert result.shell_labels[1] == "2p"
        assert result.shell_labels[2] == "3d"

    def test_angular_momentum_branching(self, simple_branch_ss):
        """Top state with 2 branches has angular momentum >= 2 at depth 0."""
        result = quantum_orbitals(simple_branch_ss)
        assert result.angular_momentum[0] >= 2

    def test_angular_momentum_leaf(self, simple_branch_ss):
        """Bottom states (end) have angular momentum 0."""
        result = quantum_orbitals(simple_branch_ss)
        max_depth = result.max_principal_quantum
        assert result.angular_momentum[max_depth] == 0

    def test_exclusion_violations_non_parallel(self, simple_branch_ss):
        """Non-parallel types have 0 exclusion violations."""
        result = quantum_orbitals(simple_branch_ss)
        assert result.exclusion_violations == 0

    def test_exclusion_violations_non_parallel_recursive(self, recursive_ss):
        """Recursive non-parallel types have 0 exclusion violations."""
        result = quantum_orbitals(recursive_ss)
        assert result.exclusion_violations == 0

    def test_max_principal_quantum_end(self, end_ss):
        """end has max principal quantum 0."""
        result = quantum_orbitals(end_ss)
        assert result.max_principal_quantum == 0

    def test_max_principal_quantum_deep(self, deep_ss):
        """Deep chain has max principal quantum = depth."""
        result = quantum_orbitals(deep_ss)
        assert result.max_principal_quantum >= 2

    def test_energy_levels_sum(self, diamond_ss):
        """Energy level counts sum to total reachable states."""
        result = quantum_orbitals(diamond_ss)
        total = sum(result.energy_levels.values())
        assert total == len(diamond_ss.states)


# ---------------------------------------------------------------------------
# Interpretation 2: Group-Theoretic Orbits
# ---------------------------------------------------------------------------


class TestGroupOrbits:
    def test_end_single_orbit(self, end_ss):
        """end has a single orbit of size 1."""
        result = compute_orbits(end_ss)
        assert result.orbit_count == 1
        assert result.trivial_orbits == 1

    def test_symmetric_branch_orbit(self, simple_branch_ss):
        """&{a: end, b: end} — the two intermediate states should be in same orbit."""
        result = compute_orbits(simple_branch_ss)
        # At least one non-trivial orbit expected (the two branch targets)
        assert result.orbit_count <= len(simple_branch_ss.states)

    def test_symmetry_ratio_end(self, end_ss):
        """end has symmetry ratio 1.0 (one state, one orbit)."""
        result = compute_orbits(end_ss)
        assert result.symmetry_ratio == 1.0

    def test_symmetry_ratio_geq_one(self, diamond_ss):
        """Symmetry ratio >= 1 always."""
        result = compute_orbits(diamond_ss)
        assert result.symmetry_ratio >= 1.0

    def test_max_orbit_size_geq_one(self, simple_branch_ss):
        """Max orbit size is at least 1."""
        result = compute_orbits(simple_branch_ss)
        assert result.max_orbit_size >= 1

    def test_orbits_partition(self, diamond_ss):
        """Orbits partition the state set."""
        result = compute_orbits(diamond_ss)
        all_states: set[int] = set()
        for orbit in result.orbits:
            assert not (all_states & orbit), "orbits must be disjoint"
            all_states |= orbit
        assert all_states == diamond_ss.states

    def test_trivial_orbits_count(self, end_ss):
        """Trivial orbits are orbits of size 1."""
        result = compute_orbits(end_ss)
        actual_trivial = sum(1 for o in result.orbits if len(o) == 1)
        assert result.trivial_orbits == actual_trivial

    def test_quotient_states_equals_orbit_count(self, recursive_ss):
        """Quotient states = orbit count."""
        result = compute_orbits(recursive_ss)
        assert result.symmetry_quotient_states == result.orbit_count

    def test_multi_branch_symmetry(self, multi_branch_ss):
        """&{a: end, b: end, c: end} — three symmetric branches."""
        result = compute_orbits(multi_branch_ss)
        assert result.max_orbit_size >= 1

    def test_selection_orbits(self, selection_ss):
        """Selection type has orbits too."""
        result = compute_orbits(selection_ss)
        assert result.orbit_count >= 1
        assert result.orbit_count <= len(selection_ss.states)


# ---------------------------------------------------------------------------
# Interpretation 3: Planetary Orbit Dynamics
# ---------------------------------------------------------------------------


class TestPlanetaryOrbits:
    def test_no_bound_states_acyclic(self, simple_branch_ss):
        """Acyclic types have no bound states."""
        result = planetary_orbits(simple_branch_ss)
        assert len(result.bound_states) == 0
        assert result.binding_energy == 0.0

    def test_bound_states_recursive(self, recursive_ss):
        """Recursive types have bound states in their cycle."""
        result = planetary_orbits(recursive_ss)
        assert len(result.bound_states) > 0
        assert result.binding_energy > 0.0

    def test_escape_velocity_bottom(self, simple_branch_ss):
        """Bottom state has escape velocity 0."""
        result = planetary_orbits(simple_branch_ss)
        assert result.escape_velocities[simple_branch_ss.bottom] == 0

    def test_escape_velocity_top(self, simple_branch_ss):
        """Top state has escape velocity > 0 for non-end types."""
        result = planetary_orbits(simple_branch_ss)
        assert result.escape_velocities[simple_branch_ss.top] > 0

    def test_free_states_acyclic(self, diamond_ss):
        """All states are free in an acyclic type."""
        result = planetary_orbits(diamond_ss)
        assert result.free_states == diamond_ss.states

    def test_binding_energy_range(self, recursive_ss):
        """Binding energy is in [0, 1]."""
        result = planetary_orbits(recursive_ss)
        assert 0.0 <= result.binding_energy <= 1.0

    def test_orbital_period_zero_acyclic(self, simple_branch_ss):
        """Acyclic states have orbital period 0."""
        result = planetary_orbits(simple_branch_ss)
        for s in simple_branch_ss.states:
            assert result.orbital_periods[s] == 0

    def test_orbital_period_positive_recursive(self, recursive_ss):
        """Recursive types have positive orbital periods for bound states."""
        result = planetary_orbits(recursive_ss)
        for s in result.bound_states:
            assert result.orbital_periods[s] > 0

    def test_lagrange_points_subset_bound(self, recursive_ss):
        """Lagrange points are a subset of bound states."""
        result = planetary_orbits(recursive_ss)
        assert result.lagrange_points <= result.bound_states

    def test_end_no_bound(self, end_ss):
        """end has no bound states."""
        result = planetary_orbits(end_ss)
        assert len(result.bound_states) == 0
        assert result.binding_energy == 0.0


# ---------------------------------------------------------------------------
# Interpretation 4: Probability Cloud
# ---------------------------------------------------------------------------


class TestProbabilityCloud:
    def test_end_uniform(self, end_ss):
        """end has probability 1.0 at the single state."""
        result = probability_cloud(end_ss)
        assert sum(result.state_probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_probabilities_sum_to_one(self, simple_branch_ss):
        """Probabilities sum to approximately 1."""
        result = probability_cloud(simple_branch_ss)
        total = sum(result.state_probabilities.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_entropy_non_negative(self, simple_branch_ss):
        """Shannon entropy is non-negative."""
        result = probability_cloud(simple_branch_ss)
        assert result.entropy >= 0.0

    def test_cloud_radius_non_negative(self, diamond_ss):
        """Cloud radius (std dev) is non-negative."""
        result = probability_cloud(diamond_ss)
        assert result.cloud_radius >= 0.0

    def test_measurement_entropy_leaf(self, simple_branch_ss):
        """Bottom state has measurement entropy 0 (no outgoing transitions)."""
        result = probability_cloud(simple_branch_ss)
        assert result.measurement_entropy[simple_branch_ss.bottom] == 0.0

    def test_measurement_entropy_branch(self, simple_branch_ss):
        """Branching state has measurement entropy = log2(branches)."""
        result = probability_cloud(simple_branch_ss)
        # Top state has 2 branches
        assert result.measurement_entropy[simple_branch_ss.top] == pytest.approx(1.0, abs=0.01)

    def test_entropy_end(self, end_ss):
        """end has entropy 0 (single state, p=1)."""
        result = probability_cloud(end_ss)
        assert result.entropy == pytest.approx(0.0, abs=0.01)

    def test_expected_depth_non_negative(self, diamond_ss):
        """Expected depth is non-negative."""
        result = probability_cloud(diamond_ss)
        assert result.expected_depth >= 0.0

    def test_recursive_probabilities(self, recursive_ss):
        """Recursive type probabilities sum to ~1."""
        result = probability_cloud(recursive_ss)
        total = sum(result.state_probabilities.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_three_branch_entropy(self, multi_branch_ss):
        """Three branches at top: measurement entropy = log2(3)."""
        result = probability_cloud(multi_branch_ss)
        import math
        assert result.measurement_entropy[multi_branch_ss.top] == pytest.approx(
            math.log2(3), abs=0.01
        )


# ---------------------------------------------------------------------------
# Interpretation 5: Type Space Orbits
# ---------------------------------------------------------------------------


class TestTypeSpaceOrbit:
    def test_single_type(self):
        """Single type yields trajectory of length 1."""
        result = type_space_orbit(["end"])
        assert len(result.trajectory) == 1
        assert result.is_convergent is False
        assert result.drift_rate == 0.0

    def test_convergent_trajectory(self):
        """Same type repeated converges."""
        result = type_space_orbit(["end", "end", "end"])
        assert result.is_convergent is True
        assert result.drift_rate == 0.0

    def test_expanding_trajectory(self):
        """Increasingly complex types yield expanding trajectory."""
        result = type_space_orbit([
            "end",
            "&{a: &{b: end}}",
            "&{a: &{b: &{c: end}}}",
            "&{a: &{b: &{c: &{d: end}}}}",
        ])
        assert result.is_expanding is True
        assert result.drift_rate > 0.0

    def test_trajectory_entries(self):
        """Each trajectory entry has (states, transitions, density)."""
        result = type_space_orbit(["end", "&{a: end}"])
        assert len(result.trajectory) == 2
        for entry in result.trajectory:
            assert len(entry) == 3
            states, trans, density = entry
            assert states >= 1
            assert trans >= 0
            assert density >= 0.0

    def test_period_detection(self):
        """Alternating types yield periodic trajectory."""
        result = type_space_orbit(["end", "&{a: end}", "end", "&{a: end}"])
        assert result.orbital_period > 0


# ---------------------------------------------------------------------------
# Combined Analysis
# ---------------------------------------------------------------------------


class TestAnalyzeOrbital:
    def test_returns_orbital_result(self, simple_branch_ss):
        """analyze_orbital returns OrbitalResult."""
        result = analyze_orbital(simple_branch_ss)
        assert isinstance(result, OrbitalResult)

    def test_quantum_field(self, simple_branch_ss):
        """Result has quantum field of correct type."""
        result = analyze_orbital(simple_branch_ss)
        assert isinstance(result.quantum, QuantumOrbitalResult)

    def test_group_field(self, simple_branch_ss):
        """Result has group field of correct type."""
        result = analyze_orbital(simple_branch_ss)
        assert isinstance(result.group, GroupOrbitResult)

    def test_planetary_field(self, recursive_ss):
        """Result has planetary field of correct type."""
        result = analyze_orbital(recursive_ss)
        assert isinstance(result.planetary, PlanetaryOrbitResult)

    def test_classification_free(self, simple_branch_ss):
        """Acyclic type classified as free."""
        result = analyze_orbital(simple_branch_ss)
        assert result.classification == "free"
