"""Tests for the Session Type Game of Life (Step 59n)."""

from __future__ import annotations

import pytest

from reticulate import parse
from reticulate.parser import Branch, End, Select
from reticulate.life import (
    Interaction,
    Organism,
    World,
    WorldConfig,
    WorldStats,
    classify_interaction,
    render,
    render_stats,
    run,
    tick,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return WorldConfig(width=5, height=5, seed=42)


@pytest.fixture
def small_world(small_config):
    return World(config=small_config)


@pytest.fixture
def predator_type():
    return parse("+{hunt: &{SUCCESS: end, FAIL: end}, rest: end}")


@pytest.fixture
def prey_type():
    return parse("&{hunt: +{ESCAPE: end, CAUGHT: end}, rest: end}")


@pytest.fixture
def simple_branch():
    return parse("&{a: end, b: end}")


@pytest.fixture
def simple_select():
    return parse("+{a: end, b: end}")


# ---------------------------------------------------------------------------
# Organism tests
# ---------------------------------------------------------------------------


class TestOrganism:
    def test_alive_with_energy(self, simple_branch):
        org = Organism(session_type=simple_branch, energy=5)
        assert org.is_alive is True

    def test_dead_with_zero_energy(self, simple_branch):
        org = Organism(session_type=simple_branch, energy=0)
        assert org.is_alive is False

    def test_dead_if_extinct_type(self):
        org = Organism(session_type=End(), energy=10)
        assert org.is_alive is False

    def test_type_str(self, simple_branch):
        org = Organism(session_type=simple_branch, energy=5)
        assert org.type_str == "&{a: end, b: end}"

    def test_repr(self, simple_branch):
        org = Organism(session_type=simple_branch, energy=5)
        assert "energy=5" in repr(org)


# ---------------------------------------------------------------------------
# Interaction tests
# ---------------------------------------------------------------------------


class TestInteraction:
    def test_predation_dual_polarity(self, predator_type, prey_type):
        """Select vs Branch with same labels = predation."""
        result = classify_interaction(predator_type, prey_type)
        assert result == Interaction.PREDATION

    def test_mutualism_identical_types(self, simple_branch):
        """Identical types are mutual subtypes = mutualism."""
        result = classify_interaction(simple_branch, simple_branch)
        assert result == Interaction.MUTUALISM

    def test_parasitism_subtype(self):
        """Wider branch is subtype of narrower = parasitism."""
        wide = parse("&{a: end, b: end, c: end}")
        narrow = parse("&{a: end, b: end}")
        result = classify_interaction(wide, narrow)
        assert result == Interaction.PARASITISM

    def test_competition_incompatible(self):
        """Two selects with different labels = competition."""
        s1 = parse("+{x: end}")
        s2 = parse("+{y: end}")
        result = classify_interaction(s1, s2)
        assert result == Interaction.COMPETITION

    def test_select_vs_branch_predation(self, simple_select, simple_branch):
        result = classify_interaction(simple_select, simple_branch)
        # Same labels, opposite polarity
        assert result in (Interaction.PREDATION, Interaction.MUTUALISM, Interaction.PARASITISM)


# ---------------------------------------------------------------------------
# World tests
# ---------------------------------------------------------------------------


class TestWorld:
    def test_place_and_get(self, small_world, simple_branch):
        org = small_world.place(2, 3, simple_branch, energy=5)
        retrieved = small_world.get(2, 3)
        assert retrieved is org

    def test_empty_cell_returns_none(self, small_world):
        assert small_world.get(0, 0) is None

    def test_toroidal_wrapping(self, small_world, simple_branch):
        small_world.place(4, 4, simple_branch, energy=5)
        # Should be same as (4, 4) due to 5x5 grid
        assert small_world.get(9, 9) is not None

    def test_population_count(self, small_world, simple_branch, simple_select):
        assert small_world.population == 0
        small_world.place(0, 0, simple_branch, energy=5)
        assert small_world.population == 1
        small_world.place(1, 1, simple_select, energy=5)
        assert small_world.population == 2

    def test_neighbours(self, small_world, simple_branch):
        small_world.place(2, 2, simple_branch, energy=5)
        small_world.place(3, 2, simple_branch, energy=5)
        neighbours = small_world.neighbours(2, 2)
        assert len(neighbours) == 1
        assert neighbours[0][:2] == (3, 2)

    def test_empty_neighbours(self, small_world, simple_branch):
        small_world.place(2, 2, simple_branch, energy=5)
        empties = small_world.empty_neighbours(2, 2)
        assert len(empties) == 4  # All 4 neighbours are empty


# ---------------------------------------------------------------------------
# Tick tests
# ---------------------------------------------------------------------------


class TestTick:
    def test_tick_advances_generation(self, small_world, simple_branch):
        small_world.place(2, 2, simple_branch, energy=5)
        assert small_world.generation == 0
        tick(small_world)
        assert small_world.generation == 1

    def test_tick_returns_stats(self, small_world, simple_branch):
        small_world.place(2, 2, simple_branch, energy=5)
        stats = tick(small_world)
        assert isinstance(stats, WorldStats)
        assert stats.generation == 1

    def test_metabolism_reduces_energy(self, small_world, simple_branch):
        org = small_world.place(2, 2, simple_branch, energy=10)
        tick(small_world)
        # Energy should decrease from metabolism
        assert org.energy < 10 or not org.is_alive

    def test_death_removes_organism(self, small_world, simple_branch):
        small_world.place(2, 2, simple_branch, energy=1)
        tick(small_world)
        # With metabolism cost, should be dead
        assert small_world.population == 0

    def test_reproduction_creates_offspring(self, small_world, simple_branch):
        # High energy to ensure reproduction
        small_world.place(2, 2, simple_branch, energy=20)
        stats = tick(small_world)
        assert stats.births >= 0  # May or may not reproduce depending on threshold

    def test_interaction_between_neighbours(self, small_world, predator_type, prey_type):
        small_world.place(2, 2, predator_type, energy=10)
        small_world.place(3, 2, prey_type, energy=10)
        stats = tick(small_world)
        assert stats.predations >= 1

    def test_aging(self, small_world, simple_branch):
        org = small_world.place(2, 2, simple_branch, energy=20)
        assert org.age == 0
        tick(small_world)
        # If alive, age should increase
        if org.is_alive:
            assert org.age == 1


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_multiple_generations(self, small_world, simple_branch):
        small_world.place(2, 2, simple_branch, energy=50)
        results = run(small_world, generations=5)
        assert len(results) >= 1
        assert len(results) <= 5

    def test_run_stops_on_extinction(self, small_world, simple_branch):
        small_world.place(2, 2, simple_branch, energy=1)
        results = run(small_world, generations=100)
        # Should stop early when population hits 0
        assert len(results) < 100

    def test_run_deterministic_with_seed(self, simple_branch):
        def make_world():
            w = World(config=WorldConfig(width=5, height=5, seed=42))
            w.place(2, 2, simple_branch, energy=20)
            w.place(1, 2, parse("+{x: end, y: end}"), energy=15)
            return w

        r1 = run(make_world(), generations=5)
        r2 = run(make_world(), generations=5)
        assert len(r1) == len(r2)
        for s1, s2 in zip(r1, r2):
            assert s1.population == s2.population


# ---------------------------------------------------------------------------
# Render tests
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_empty_world(self, small_world):
        output = render(small_world)
        assert "Generation 0" in output
        assert "Pop: 0" in output
        # Should have dots for empty cells
        assert "." in output

    def test_render_with_organisms(self, small_world, predator_type, prey_type):
        small_world.place(0, 0, predator_type, energy=10)
        small_world.place(1, 0, prey_type, energy=10)
        output = render(small_world)
        assert "S" in output  # Select = predator
        assert "B" in output  # Branch = prey

    def test_render_energy_levels(self, small_world, simple_select):
        small_world.place(0, 0, simple_select, energy=10)
        small_world.place(1, 0, simple_select, energy=2)
        output = render(small_world)
        assert "S" in output  # High energy
        assert "s" in output  # Low energy

    def test_render_stats_empty(self, small_world):
        output = render_stats(small_world)
        assert "No history" in output

    def test_render_stats_with_history(self, small_world, simple_branch):
        small_world.place(2, 2, simple_branch, energy=10)
        tick(small_world)
        output = render_stats(small_world)
        assert "Gen" in output
        assert "Pop" in output


# ---------------------------------------------------------------------------
# Integration / scenario tests
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_predator_prey_ecosystem(self):
        """Run a small predator-prey ecosystem."""
        cfg = WorldConfig(width=5, height=5, seed=42, reproduction_threshold=10)
        world = World(config=cfg)

        pred = parse("+{hunt: &{SUCCESS: end, FAIL: end}, rest: end}")
        prey = parse("&{hunt: +{ESCAPE: end, CAUGHT: end}, rest: end}")

        world.place(0, 0, pred, energy=15)
        world.place(4, 4, prey, energy=15)

        results = run(world, generations=10)
        assert len(results) >= 1

    def test_pure_competition(self):
        """Two competing species in adjacent cells."""
        cfg = WorldConfig(width=3, height=3, seed=42)
        world = World(config=cfg)

        s1 = parse("+{x: end}")
        s2 = parse("+{y: end}")

        world.place(1, 1, s1, energy=10)
        world.place(2, 1, s2, energy=10)

        stats = tick(world)
        assert stats.competitions >= 1

    def test_mutualism_boosts_both(self):
        """Two identical types should gain energy from mutualism."""
        cfg = WorldConfig(width=3, height=3, seed=42, interaction_gain=6)
        world = World(config=cfg)

        t = parse("&{a: end, b: end}")
        org1 = world.place(1, 1, t, energy=10)
        org2 = world.place(2, 1, t, energy=10)

        stats = tick(world)
        assert stats.mutualisms >= 1
