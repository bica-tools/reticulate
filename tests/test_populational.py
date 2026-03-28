"""Tests for the populational session type module (Step 59m)."""

from __future__ import annotations

import pytest

from reticulate import parse, pretty, build_statespace, check_lattice
from reticulate.parser import Branch, End, Select, Rec, Var
from reticulate.populational import (
    MutationKind,
    Mutation,
    EvolutionResult,
    add_branch,
    remove_branch,
    swap_polarity,
    deepen_branch,
    widen_branch,
    mutate,
    fitness,
    fitness_detailed,
    is_extinct,
    extinction_risk,
    is_speciation,
    evolve,
    predator_prey,
    sir_model,
    lotka_volterra,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_branch():
    return parse("&{a: end, b: end}")


@pytest.fixture
def simple_select():
    return parse("+{a: end, b: end}")


@pytest.fixture
def nim3():
    return parse("+{take1: &{take1: +{take1: end}, take2: end}, take2: &{take1: end}, take3: end}")


@pytest.fixture
def recursive_type():
    return parse("rec X . &{a: +{b: X, c: end}}")


# ---------------------------------------------------------------------------
# Mutation operator tests
# ---------------------------------------------------------------------------


class TestAddBranch:
    def test_add_to_branch(self, simple_branch):
        result = add_branch(simple_branch, "c")
        assert isinstance(result, Branch)
        labels = [l for l, _ in result.choices]
        assert "c" in labels
        assert len(result.choices) == 3

    def test_add_to_select(self, simple_select):
        result = add_branch(simple_select, "c")
        assert isinstance(result, Select)
        labels = [l for l, _ in result.choices]
        assert "c" in labels

    def test_add_duplicate_label_noop(self, simple_branch):
        result = add_branch(simple_branch, "a")
        assert result == simple_branch

    def test_add_to_end(self):
        result = add_branch(End(), "x")
        assert result == End()

    def test_add_preserves_existing(self, simple_branch):
        result = add_branch(simple_branch, "c")
        labels = [l for l, _ in result.choices]
        assert "a" in labels
        assert "b" in labels


class TestRemoveBranch:
    def test_remove_named(self, simple_branch):
        result = remove_branch(simple_branch, "a")
        assert isinstance(result, Branch)
        labels = [l for l, _ in result.choices]
        assert "a" not in labels
        assert "b" in labels

    def test_remove_last_becomes_end(self):
        single = parse("&{a: end}")
        result = remove_branch(single, "a")
        assert isinstance(result, End)

    def test_remove_unnamed_removes_last(self, simple_branch):
        result = remove_branch(simple_branch)
        assert isinstance(result, Branch)
        assert len(result.choices) == 1

    def test_remove_from_end(self):
        result = remove_branch(End())
        assert result == End()


class TestSwapPolarity:
    def test_branch_to_select(self, simple_branch):
        result = swap_polarity(simple_branch)
        assert isinstance(result, Select)
        assert result.choices == simple_branch.choices

    def test_select_to_branch(self, simple_select):
        result = swap_polarity(simple_select)
        assert isinstance(result, Branch)
        assert result.choices == simple_select.choices

    def test_swap_end_noop(self):
        assert swap_polarity(End()) == End()

    def test_swap_is_involution(self, simple_branch):
        assert swap_polarity(swap_polarity(simple_branch)) == simple_branch


class TestDeepenBranch:
    def test_deepen_existing(self, simple_branch):
        result = deepen_branch(simple_branch, "a", "prep")
        # a should now lead to &{prep: end} instead of end
        a_cont = dict(result.choices)["a"]
        assert isinstance(a_cont, Branch)
        assert a_cont.choices[0][0] == "prep"

    def test_deepen_nonexistent_noop(self, simple_branch):
        result = deepen_branch(simple_branch, "z", "prep")
        assert result == simple_branch


class TestWidenBranch:
    def test_widen_creates_duplicate(self, simple_branch):
        result = widen_branch(simple_branch, "a", "a_copy")
        labels = [l for l, _ in result.choices]
        assert "a_copy" in labels
        assert len(result.choices) == 3
        # a and a_copy should have same continuation
        continuations = dict(result.choices)
        assert continuations["a"] == continuations["a_copy"]

    def test_widen_existing_target_noop(self, simple_branch):
        result = widen_branch(simple_branch, "a", "b")
        assert result == simple_branch


# ---------------------------------------------------------------------------
# Mutate function tests
# ---------------------------------------------------------------------------


class TestMutate:
    def test_mutate_returns_tuple(self, simple_branch):
        result, record = mutate(simple_branch, MutationKind.ADD_BRANCH, label="c")
        assert isinstance(record, Mutation)
        assert record.kind == MutationKind.ADD_BRANCH

    def test_mutation_record_has_strings(self, simple_branch):
        _, record = mutate(simple_branch, MutationKind.SWAP_POLARITY)
        assert isinstance(record.original, str)
        assert isinstance(record.mutant, str)
        assert isinstance(record.site, str)

    def test_all_kinds_work(self, simple_branch):
        for kind in MutationKind:
            kwargs = {}
            if kind == MutationKind.ADD_BRANCH:
                kwargs = {"label": "z"}
            elif kind == MutationKind.REMOVE_BRANCH:
                kwargs = {"label": "a"}
            elif kind == MutationKind.DEEPEN_BRANCH:
                kwargs = {"label": "a"}
            elif kind == MutationKind.WIDEN_BRANCH:
                kwargs = {"source_label": "a", "new_label": "z"}
            result, record = mutate(simple_branch, kind, **kwargs)
            assert record.kind == kind


# ---------------------------------------------------------------------------
# Fitness tests
# ---------------------------------------------------------------------------


class TestFitness:
    def test_end_has_zero_fitness(self):
        assert fitness(End()) == 0

    def test_branch_has_positive_fitness(self, simple_branch):
        assert fitness(simple_branch) > 0

    def test_nim3_fitness(self, nim3):
        assert fitness(nim3) == 3

    def test_adding_branch_increases_or_maintains_fitness(self, simple_branch):
        f_before = fitness(simple_branch)
        mutant = add_branch(simple_branch, "c")
        f_after = fitness(mutant)
        assert f_after >= f_before

    def test_fitness_detailed_keys(self, nim3):
        fd = fitness_detailed(nim3)
        assert "potential" in fd
        assert "is_lattice" in fd
        assert "states" in fd
        assert fd["is_lattice"] is True


# ---------------------------------------------------------------------------
# Extinction tests
# ---------------------------------------------------------------------------


class TestExtinction:
    def test_end_is_extinct(self):
        assert is_extinct(End()) is True

    def test_branch_not_extinct(self, simple_branch):
        assert is_extinct(simple_branch) is False

    def test_extinction_risk_end(self):
        assert extinction_risk(End()) == 1.0

    def test_extinction_risk_decreases_with_fitness(self, simple_branch, nim3):
        r1 = extinction_risk(simple_branch)
        r2 = extinction_risk(nim3)
        # Nim3 has higher fitness, so lower extinction risk
        assert r2 < r1

    def test_extinction_risk_range(self, nim3):
        r = extinction_risk(nim3)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Speciation tests
# ---------------------------------------------------------------------------


class TestSpeciation:
    def test_identical_not_speciation(self, simple_branch):
        assert is_speciation(simple_branch, simple_branch) is False

    def test_added_branch_not_speciation(self, simple_branch):
        """Adding a branch creates a subtype — not speciation."""
        mutant = add_branch(simple_branch, "c")
        # Wider branch = subtype in Gay-Hole subtyping
        assert is_speciation(simple_branch, mutant) is False

    def test_swap_polarity_is_speciation(self, simple_branch):
        """Swapping &/+ creates incompatible types — speciation."""
        mutant = swap_polarity(simple_branch)
        assert is_speciation(simple_branch, mutant) is True


# ---------------------------------------------------------------------------
# Evolution tests
# ---------------------------------------------------------------------------


class TestEvolution:
    def test_evolve_returns_result(self, simple_branch):
        result = evolve([simple_branch], generations=3, seed=42)
        assert isinstance(result, EvolutionResult)
        assert result.generations == 3

    def test_evolve_preserves_population(self, simple_branch):
        result = evolve([simple_branch, simple_branch], generations=2, seed=42)
        assert len(result.final_population) >= 1

    def test_evolve_records_history(self, simple_branch):
        result = evolve([simple_branch], generations=5, seed=42)
        assert len(result.fitness_history) == 5
        for gen, max_f, avg_f in result.fitness_history:
            assert isinstance(gen, int)
            assert isinstance(max_f, int)
            assert isinstance(avg_f, float)

    def test_evolve_empty_population(self):
        result = evolve([], generations=3)
        assert result.generations <= 1
        assert len(result.final_population) == 0

    def test_evolve_with_seed_deterministic(self, simple_branch):
        r1 = evolve([simple_branch], generations=5, seed=123)
        r2 = evolve([simple_branch], generations=5, seed=123)
        assert r1.final_population == r2.final_population
        assert r1.fitness_history == r2.fitness_history


# ---------------------------------------------------------------------------
# Ecological constructors tests
# ---------------------------------------------------------------------------


class TestEcologicalConstructors:
    def test_predator_prey_returns_pair(self):
        pred, prey = predator_prey(2)
        assert isinstance(pred, Select)
        assert isinstance(prey, Branch)

    def test_predator_prey_parseable(self):
        pred, prey = predator_prey(2)
        # Should be valid session types
        ss_pred = build_statespace(pred)
        ss_prey = build_statespace(prey)
        assert len(ss_pred.states) > 1
        assert len(ss_prey.states) > 1

    def test_predator_prey_lattice(self):
        pred, prey = predator_prey(2)
        lr_pred = check_lattice(build_statespace(pred))
        lr_prey = check_lattice(build_statespace(prey))
        assert lr_pred.is_lattice is True
        assert lr_prey.is_lattice is True

    def test_sir_model_valid(self):
        sir = sir_model()
        ss = build_statespace(sir)
        assert len(ss.states) > 1
        lr = check_lattice(ss)
        assert lr.is_lattice is True

    def test_sir_model_top_is_susceptible(self):
        sir = sir_model()
        assert isinstance(sir, Branch)
        labels = [l for l, _ in sir.choices]
        assert "contact_infected" in labels

    def test_lotka_volterra_valid(self):
        lv = lotka_volterra(3, 2)
        ss = build_statespace(lv)
        assert len(ss.states) > 1
        lr = check_lattice(ss)
        assert lr.is_lattice is True

    def test_lotka_volterra_fitness(self):
        lv = lotka_volterra(3, 2)
        f = fitness(lv)
        assert f > 0

    def test_predator_is_dual_of_prey_at_top(self):
        """Predator Select at top, Prey Branch at top — dual polarity."""
        pred, prey = predator_prey(1)
        assert isinstance(pred, Select)
        assert isinstance(prey, Branch)
        # Top-level labels should match
        pred_labels = {l for l, _ in pred.choices}
        prey_labels = {l for l, _ in prey.choices}
        assert pred_labels == prey_labels
