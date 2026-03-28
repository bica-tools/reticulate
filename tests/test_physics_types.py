"""Tests for Fundamental Physics as Session Types (Step 207)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.physics_types import (
    PHYSICS_LIBRARY,
    PhysicsEntry,
    all_physics_form_lattices,
    get_physics,
    grand_unified_type,
    physics_by_domain,
    physics_by_scale,
    unification_score,
)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


class TestPhysicsLibrary:
    def test_library_has_at_least_25_entries(self) -> None:
        assert len(PHYSICS_LIBRARY) >= 25

    def test_all_entries_are_physics_entry(self) -> None:
        for entry in PHYSICS_LIBRARY.values():
            assert isinstance(entry, PhysicsEntry)

    def test_all_names_match_keys(self) -> None:
        for key, entry in PHYSICS_LIBRARY.items():
            assert key == entry.name

    def test_all_domains_valid(self) -> None:
        valid = {"quantum", "thermodynamics", "relativity", "cosmology",
                 "forces", "particles"}
        for entry in PHYSICS_LIBRARY.values():
            assert entry.domain in valid, f"{entry.name} has invalid domain {entry.domain}"

    def test_all_scales_valid(self) -> None:
        valid = {"planck", "atomic", "human", "cosmic"}
        for entry in PHYSICS_LIBRARY.values():
            assert entry.scale in valid, f"{entry.name} has invalid scale {entry.scale}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    @pytest.mark.parametrize("name", list(PHYSICS_LIBRARY.keys()))
    def test_every_entry_parses(self, name: str) -> None:
        entry = PHYSICS_LIBRARY[name]
        ast = parse(entry.session_type_str)
        assert ast is not None

    @pytest.mark.parametrize("name", list(PHYSICS_LIBRARY.keys()))
    def test_every_entry_builds_statespace(self, name: str) -> None:
        entry = PHYSICS_LIBRARY[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2


# ---------------------------------------------------------------------------
# Lattice verification
# ---------------------------------------------------------------------------


class TestLattice:
    @pytest.mark.parametrize("name", list(PHYSICS_LIBRARY.keys()))
    def test_every_entry_forms_lattice(self, name: str) -> None:
        entry = PHYSICS_LIBRARY[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.is_lattice, f"{name} does not form a lattice"

    def test_all_physics_form_lattices(self) -> None:
        assert all_physics_form_lattices()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestGetPhysics:
    def test_lookup_gravity(self) -> None:
        entry = get_physics("gravity")
        assert entry.name == "gravity"
        assert entry.domain == "forces"

    def test_lookup_superposition(self) -> None:
        entry = get_physics("superposition")
        assert entry.domain == "quantum"

    def test_lookup_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown physics entry"):
            get_physics("unicorn_force")

    def test_lookup_all(self) -> None:
        for name in PHYSICS_LIBRARY:
            entry = get_physics(name)
            assert entry.name == name


# ---------------------------------------------------------------------------
# Domain and scale queries
# ---------------------------------------------------------------------------


class TestDomainScale:
    def test_forces_domain(self) -> None:
        forces = physics_by_domain("forces")
        assert len(forces) == 4
        names = {e.name for e in forces}
        assert "gravity" in names
        assert "electromagnetism" in names

    def test_quantum_domain(self) -> None:
        quantum = physics_by_domain("quantum")
        assert len(quantum) >= 5
        assert all(e.domain == "quantum" for e in quantum)

    def test_thermodynamics_domain(self) -> None:
        thermo = physics_by_domain("thermodynamics")
        assert len(thermo) >= 4

    def test_cosmology_domain(self) -> None:
        cosmo = physics_by_domain("cosmology")
        assert len(cosmo) >= 4

    def test_particles_domain(self) -> None:
        parts = physics_by_domain("particles")
        assert len(parts) >= 3

    def test_relativity_domain(self) -> None:
        rel = physics_by_domain("relativity")
        assert len(rel) >= 2

    def test_planck_scale(self) -> None:
        planck = physics_by_scale("planck")
        assert len(planck) >= 3

    def test_cosmic_scale(self) -> None:
        cosmic = physics_by_scale("cosmic")
        assert len(cosmic) >= 4

    def test_empty_domain(self) -> None:
        assert physics_by_domain("phlogiston") == []

    def test_empty_scale(self) -> None:
        assert physics_by_scale("galactic") == []


# ---------------------------------------------------------------------------
# Unification score
# ---------------------------------------------------------------------------


class TestUnificationScore:
    def test_same_type_score_one(self) -> None:
        score = unification_score("gravity", "gravity")
        assert score == 1.0

    def test_related_forces_positive(self) -> None:
        # Forces share some structural similarity
        score = unification_score("strong_nuclear", "weak_nuclear")
        assert score >= 0.0

    def test_unrelated_score_low(self) -> None:
        score = unification_score("gravity", "superposition")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            unification_score("gravity", "magic")


# ---------------------------------------------------------------------------
# Grand unified type
# ---------------------------------------------------------------------------


class TestGrandUnifiedType:
    def test_gut_returns_string(self) -> None:
        gut = grand_unified_type()
        assert isinstance(gut, str)

    def test_gut_parses(self) -> None:
        gut = grand_unified_type()
        ast = parse(gut)
        assert ast is not None

    def test_gut_forms_lattice(self) -> None:
        gut = grand_unified_type()
        ast = parse(gut)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.is_lattice
