"""Tests for Encyclopedia of Civilization as Session Types (Step 212)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.encyclopedia_civilization import (
    CIVILIZATION_ENCYCLOPEDIA,
    CivilizationEntry,
    all_civilization_form_lattices,
    civilization_by_domain,
)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


class TestCivilizationEncyclopedia:
    def test_encyclopedia_has_at_least_50_entries(self) -> None:
        assert len(CIVILIZATION_ENCYCLOPEDIA) >= 50

    def test_all_entries_are_civilization_entry(self) -> None:
        for entry in CIVILIZATION_ENCYCLOPEDIA.values():
            assert isinstance(entry, CivilizationEntry)

    def test_all_names_match_keys(self) -> None:
        for key, entry in CIVILIZATION_ENCYCLOPEDIA.items():
            assert key == entry.name

    def test_all_domains_valid(self) -> None:
        valid = {"governance", "economics", "education", "medicine", "technology", "justice"}
        for entry in CIVILIZATION_ENCYCLOPEDIA.values():
            assert entry.domain in valid, f"{entry.name} has invalid domain {entry.domain}"

    def test_all_descriptions_nonempty(self) -> None:
        for entry in CIVILIZATION_ENCYCLOPEDIA.values():
            assert len(entry.description) > 10, f"{entry.name} has short description"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    @pytest.mark.parametrize("name", list(CIVILIZATION_ENCYCLOPEDIA.keys()))
    def test_every_entry_parses(self, name: str) -> None:
        entry = CIVILIZATION_ENCYCLOPEDIA[name]
        ast = parse(entry.session_type_str)
        assert ast is not None

    @pytest.mark.parametrize("name", list(CIVILIZATION_ENCYCLOPEDIA.keys()))
    def test_every_entry_builds_statespace(self, name: str) -> None:
        entry = CIVILIZATION_ENCYCLOPEDIA[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2


# ---------------------------------------------------------------------------
# Lattice verification
# ---------------------------------------------------------------------------


class TestLattice:
    @pytest.mark.parametrize("name", list(CIVILIZATION_ENCYCLOPEDIA.keys()))
    def test_every_entry_forms_lattice(self, name: str) -> None:
        entry = CIVILIZATION_ENCYCLOPEDIA[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.is_lattice, f"{name} does not form a lattice"

    def test_all_civilization_form_lattices(self) -> None:
        assert all_civilization_form_lattices()


# ---------------------------------------------------------------------------
# Domain queries
# ---------------------------------------------------------------------------


class TestDomainQueries:
    def test_governance_domain(self) -> None:
        gov = civilization_by_domain("governance")
        assert len(gov) == 10
        names = {e.name for e in gov}
        assert "election" in names
        assert "revolution" in names

    def test_economics_domain(self) -> None:
        econ = civilization_by_domain("economics")
        assert len(econ) == 10
        names = {e.name for e in econ}
        assert "auction" in names
        assert "supply_chain" in names

    def test_education_domain(self) -> None:
        edu = civilization_by_domain("education")
        assert len(edu) == 8
        names = {e.name for e in edu}
        assert "thesis_defense" in names
        assert "apprenticeship" in names

    def test_medicine_domain(self) -> None:
        med = civilization_by_domain("medicine")
        assert len(med) == 8
        names = {e.name for e in med}
        assert "surgery" in names
        assert "clinical_trial" in names

    def test_technology_domain(self) -> None:
        tech = civilization_by_domain("technology")
        assert len(tech) == 8
        names = {e.name for e in tech}
        assert "software_development" in names
        assert "space_launch" in names

    def test_justice_domain(self) -> None:
        jus = civilization_by_domain("justice")
        assert len(jus) == 6
        names = {e.name for e in jus}
        assert "arrest" in names
        assert "mediation" in names

    def test_empty_domain(self) -> None:
        assert civilization_by_domain("religion") == []

    def test_all_domains_covered(self) -> None:
        domains = {e.domain for e in CIVILIZATION_ENCYCLOPEDIA.values()}
        assert domains == {"governance", "economics", "education", "medicine", "technology", "justice"}
