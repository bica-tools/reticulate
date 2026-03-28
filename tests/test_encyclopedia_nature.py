"""Tests for Encyclopedia of Nature as Session Types (Step 210)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.encyclopedia_nature import (
    NATURE_ENCYCLOPEDIA,
    NatureEntry,
    all_nature_form_lattices,
    nature_by_domain,
)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


class TestNatureEncyclopedia:
    def test_encyclopedia_has_at_least_50_entries(self) -> None:
        assert len(NATURE_ENCYCLOPEDIA) >= 50

    def test_all_entries_are_nature_entry(self) -> None:
        for entry in NATURE_ENCYCLOPEDIA.values():
            assert isinstance(entry, NatureEntry)

    def test_all_names_match_keys(self) -> None:
        for key, entry in NATURE_ENCYCLOPEDIA.items():
            assert key == entry.name

    def test_all_domains_valid(self) -> None:
        valid = {"biology", "chemistry", "ecology", "geology", "weather"}
        for entry in NATURE_ENCYCLOPEDIA.values():
            assert entry.domain in valid, f"{entry.name} has invalid domain {entry.domain}"

    def test_all_descriptions_nonempty(self) -> None:
        for entry in NATURE_ENCYCLOPEDIA.values():
            assert len(entry.description) > 10, f"{entry.name} has short description"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    @pytest.mark.parametrize("name", list(NATURE_ENCYCLOPEDIA.keys()))
    def test_every_entry_parses(self, name: str) -> None:
        entry = NATURE_ENCYCLOPEDIA[name]
        ast = parse(entry.session_type_str)
        assert ast is not None

    @pytest.mark.parametrize("name", list(NATURE_ENCYCLOPEDIA.keys()))
    def test_every_entry_builds_statespace(self, name: str) -> None:
        entry = NATURE_ENCYCLOPEDIA[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2


# ---------------------------------------------------------------------------
# Lattice verification
# ---------------------------------------------------------------------------


class TestLattice:
    @pytest.mark.parametrize("name", list(NATURE_ENCYCLOPEDIA.keys()))
    def test_every_entry_forms_lattice(self, name: str) -> None:
        entry = NATURE_ENCYCLOPEDIA[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.is_lattice, f"{name} does not form a lattice"

    def test_all_nature_form_lattices(self) -> None:
        assert all_nature_form_lattices()


# ---------------------------------------------------------------------------
# Domain queries
# ---------------------------------------------------------------------------


class TestDomainQueries:
    def test_biology_domain(self) -> None:
        bio = nature_by_domain("biology")
        assert len(bio) == 15
        names = {e.name for e in bio}
        assert "mitosis" in names
        assert "photosynthesis" in names

    def test_chemistry_domain(self) -> None:
        chem = nature_by_domain("chemistry")
        assert len(chem) == 10
        names = {e.name for e in chem}
        assert "combustion" in names
        assert "nuclear_fission" in names

    def test_ecology_domain(self) -> None:
        eco = nature_by_domain("ecology")
        assert len(eco) == 10
        names = {e.name for e in eco}
        assert "succession" in names
        assert "carbon_cycle" in names

    def test_geology_domain(self) -> None:
        geo = nature_by_domain("geology")
        assert len(geo) == 8
        names = {e.name for e in geo}
        assert "erosion" in names
        assert "earthquake" in names

    def test_weather_domain(self) -> None:
        weather = nature_by_domain("weather")
        assert len(weather) == 7
        names = {e.name for e in weather}
        assert "hurricane_formation" in names
        assert "tornado" in names

    def test_empty_domain(self) -> None:
        assert nature_by_domain("alchemy") == []

    def test_all_domains_covered(self) -> None:
        domains = {e.domain for e in NATURE_ENCYCLOPEDIA.values()}
        assert domains == {"biology", "chemistry", "ecology", "geology", "weather"}
