"""Tests for Encyclopedia of Human Experience as Session Types (Step 211)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.encyclopedia_human import (
    HUMAN_ENCYCLOPEDIA,
    HumanEntry,
    all_human_form_lattices,
    human_by_domain,
)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


class TestHumanEncyclopedia:
    def test_encyclopedia_has_at_least_50_entries(self) -> None:
        assert len(HUMAN_ENCYCLOPEDIA) >= 50

    def test_all_entries_are_human_entry(self) -> None:
        for entry in HUMAN_ENCYCLOPEDIA.values():
            assert isinstance(entry, HumanEntry)

    def test_all_names_match_keys(self) -> None:
        for key, entry in HUMAN_ENCYCLOPEDIA.items():
            assert key == entry.name

    def test_all_domains_valid(self) -> None:
        valid = {"emotions", "relationships", "life_events", "daily_activities", "rituals"}
        for entry in HUMAN_ENCYCLOPEDIA.values():
            assert entry.domain in valid, f"{entry.name} has invalid domain {entry.domain}"

    def test_all_descriptions_nonempty(self) -> None:
        for entry in HUMAN_ENCYCLOPEDIA.values():
            assert len(entry.description) > 10, f"{entry.name} has short description"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    @pytest.mark.parametrize("name", list(HUMAN_ENCYCLOPEDIA.keys()))
    def test_every_entry_parses(self, name: str) -> None:
        entry = HUMAN_ENCYCLOPEDIA[name]
        ast = parse(entry.session_type_str)
        assert ast is not None

    @pytest.mark.parametrize("name", list(HUMAN_ENCYCLOPEDIA.keys()))
    def test_every_entry_builds_statespace(self, name: str) -> None:
        entry = HUMAN_ENCYCLOPEDIA[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2


# ---------------------------------------------------------------------------
# Lattice verification
# ---------------------------------------------------------------------------


class TestLattice:
    @pytest.mark.parametrize("name", list(HUMAN_ENCYCLOPEDIA.keys()))
    def test_every_entry_forms_lattice(self, name: str) -> None:
        entry = HUMAN_ENCYCLOPEDIA[name]
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.is_lattice, f"{name} does not form a lattice"

    def test_all_human_form_lattices(self) -> None:
        assert all_human_form_lattices()


# ---------------------------------------------------------------------------
# Domain queries
# ---------------------------------------------------------------------------


class TestDomainQueries:
    def test_emotions_domain(self) -> None:
        emo = human_by_domain("emotions")
        assert len(emo) == 12
        names = {e.name for e in emo}
        assert "joy" in names
        assert "fear" in names
        assert "hope" in names

    def test_relationships_domain(self) -> None:
        rel = human_by_domain("relationships")
        assert len(rel) == 8
        names = {e.name for e in rel}
        assert "friendship" in names
        assert "romance" in names

    def test_life_events_domain(self) -> None:
        life = human_by_domain("life_events")
        assert len(life) == 10
        names = {e.name for e in life}
        assert "birth" in names
        assert "death" in names

    def test_daily_activities_domain(self) -> None:
        daily = human_by_domain("daily_activities")
        assert len(daily) == 10
        names = {e.name for e in daily}
        assert "waking_up" in names
        assert "dreaming" in names

    def test_rituals_domain(self) -> None:
        rit = human_by_domain("rituals")
        assert len(rit) == 10
        names = {e.name for e in rit}
        assert "prayer" in names
        assert "funeral" in names

    def test_empty_domain(self) -> None:
        assert human_by_domain("supernatural") == []

    def test_all_domains_covered(self) -> None:
        domains = {e.domain for e in HUMAN_ENCYCLOPEDIA.values()}
        assert domains == {"emotions", "relationships", "life_events", "daily_activities", "rituals"}
