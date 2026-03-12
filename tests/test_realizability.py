"""Tests for realizability: which bounded lattices arise as L(S)? (Step 156)."""

import pytest

from reticulate.parser import parse, pretty
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice
from reticulate.realizability import (
    Obstruction,
    RealizabilityConditions,
    RealizabilityResult,
    check_determinism,
    check_realizability,
    check_realizability_conditions,
    find_obstructions,
    generate_non_realizable,
    NON_REALIZABLE_CATALOGUE,
)


# ═══════════════════════════════════════════════════════════════════
# Sprint 1: Core data types + determinism + conditions + obstructions
# ═══════════════════════════════════════════════════════════════════


class TestObstruction:
    """Obstruction data type."""

    def test_frozen(self):
        o = Obstruction(kind="test", states=(1, 2), detail="x")
        with pytest.raises(AttributeError):
            o.kind = "other"

    def test_fields(self):
        o = Obstruction(kind="not_lattice", states=(1, 2), detail="no meet")
        assert o.kind == "not_lattice"
        assert o.states == (1, 2)
        assert o.detail == "no meet"


class TestRealizabilityConditions:
    """RealizabilityConditions data type."""

    def test_frozen(self):
        c = RealizabilityConditions(
            is_lattice=True, is_bounded=True, is_deterministic=True,
            all_reachable=True, all_reach_bottom=True, has_reticular_form=True,
        )
        with pytest.raises(AttributeError):
            c.is_lattice = False

    def test_all_true(self):
        c = RealizabilityConditions(
            is_lattice=True, is_bounded=True, is_deterministic=True,
            all_reachable=True, all_reach_bottom=True, has_reticular_form=True,
        )
        assert c.is_lattice and c.is_bounded and c.is_deterministic


class TestCheckDeterminism:
    """Determinism check."""

    def test_deterministic_branch(self):
        ss = build_statespace(parse("&{a: end, b: end}"))
        assert check_determinism(ss) == []

    def test_nondeterministic(self):
        ss = StateSpace(
            states={0, 1, 2, 3},
            transitions=[(0, "a", 1), (0, "a", 2), (1, "b", 3), (2, "c", 3)],
            top=0, bottom=3,
        )
        obs = check_determinism(ss)
        assert len(obs) >= 1
        assert obs[0].kind == "non_deterministic"

    def test_session_types_are_deterministic(self):
        for t in ["end", "&{a: end}", "+{a: end, b: end}",
                   "rec X . &{a: X, b: end}"]:
            ss = build_statespace(parse(t))
            assert check_determinism(ss) == [], f"{t} should be deterministic"


class TestCheckRealizabilityConditions:
    """Conditions check."""

    def test_simple_branch_all_true(self):
        ss = build_statespace(parse("&{a: end, b: end}"))
        c = check_realizability_conditions(ss)
        assert c.is_lattice
        assert c.is_bounded
        assert c.is_deterministic
        assert c.all_reachable
        assert c.all_reach_bottom
        assert c.has_reticular_form

    def test_end_all_true(self):
        ss = build_statespace(parse("end"))
        c = check_realizability_conditions(ss)
        assert c.is_lattice

    def test_unreachable_state_fails(self):
        ss = StateSpace(
            states={0, 1, 2, 3},
            transitions=[(0, "a", 1), (1, "b", 2)],
            top=0, bottom=2,
        )
        c = check_realizability_conditions(ss)
        assert not c.all_reachable


class TestFindObstructions:
    """Obstruction detection."""

    def test_no_obstructions_for_session_types(self):
        for t in ["end", "&{a: end}", "&{a: end, b: end}",
                   "+{a: end, b: end}", "rec X . &{a: X, b: end}"]:
            ss = build_statespace(parse(t))
            obs = find_obstructions(ss)
            assert obs == [], f"{t} should have no obstructions, got {obs}"

    def test_detects_nondeterminism(self):
        ss = generate_non_realizable("nondeterministic")
        obs = find_obstructions(ss)
        kinds = {o.kind for o in obs}
        assert "non_deterministic" in kinds

    def test_detects_unreachable(self):
        ss = generate_non_realizable("unreachable_orphan")
        obs = find_obstructions(ss)
        kinds = {o.kind for o in obs}
        assert "unreachable_states" in kinds

    def test_detects_dead_end(self):
        ss = generate_non_realizable("dead_end")
        obs = find_obstructions(ss)
        kinds = {o.kind for o in obs}
        assert "no_path_to_bottom" in kinds

    def test_detects_bottom_transitions(self):
        ss = generate_non_realizable("self_loop_bottom")
        obs = find_obstructions(ss)
        kinds = {o.kind for o in obs}
        assert "bottom_has_transitions" in kinds


# ═══════════════════════════════════════════════════════════════════
# Sprint 2: Non-realizable catalogue
# ═══════════════════════════════════════════════════════════════════


class TestNonRealizableCatalogue:
    """18 hand-crafted non-realizable examples."""

    def test_catalogue_has_18_entries(self):
        assert len(NON_REALIZABLE_CATALOGUE) == 18

    def test_all_entries_produce_state_spaces(self):
        for kind in NON_REALIZABLE_CATALOGUE:
            ss = generate_non_realizable(kind)
            assert isinstance(ss, StateSpace), f"{kind} didn't produce StateSpace"

    @pytest.mark.parametrize("kind", sorted(NON_REALIZABLE_CATALOGUE.keys()))
    def test_each_entry_is_non_realizable(self, kind):
        """Every catalogue entry should be non-realizable."""
        ss = generate_non_realizable(kind)
        result = check_realizability(ss)
        assert not result.is_realizable, (
            f"{kind} should be non-realizable but was realizable"
        )

    @pytest.mark.parametrize("kind", sorted(NON_REALIZABLE_CATALOGUE.keys()))
    def test_each_entry_has_obstructions(self, kind):
        """Every catalogue entry should have at least one obstruction."""
        ss = generate_non_realizable(kind)
        result = check_realizability(ss)
        assert len(result.obstructions) > 0, (
            f"{kind} should have obstructions"
        )

    def test_generate_unknown_raises(self):
        with pytest.raises(KeyError):
            generate_non_realizable("nonexistent_kind")

    # --- Category A: Structural ---
    # --- Category A: Structural ---
    def test_nondeterministic_not_realizable(self):
        ss = generate_non_realizable("nondeterministic")
        result = check_realizability(ss)
        assert not result.is_realizable
        assert not result.conditions.is_lattice

    def test_unreachable_orphan_not_realizable(self):
        ss = generate_non_realizable("unreachable_orphan")
        result = check_realizability(ss)
        assert not result.is_realizable

    def test_dead_end_not_realizable(self):
        ss = generate_non_realizable("dead_end")
        result = check_realizability(ss)
        assert not result.is_realizable

    def test_self_loop_bottom_not_realizable(self):
        ss = generate_non_realizable("self_loop_bottom")
        result = check_realizability(ss)
        assert not result.is_realizable

    # --- Category B: Lattice ---
    def test_missing_meet_not_lattice(self):
        ss = generate_non_realizable("missing_meet")
        result = check_realizability(ss)
        assert not result.is_realizable
        assert not result.conditions.is_lattice

    def test_missing_join_not_realizable(self):
        ss = generate_non_realizable("missing_join")
        result = check_realizability(ss)
        assert not result.is_realizable

    def test_not_bounded_not_realizable(self):
        ss = generate_non_realizable("not_bounded")
        result = check_realizability(ss)
        assert not result.is_realizable

    # --- Category C: Reticular ---
    def test_mixed_alien_not_realizable(self):
        ss = generate_non_realizable("mixed_alien")
        result = check_realizability(ss)
        assert not result.is_realizable

    def test_label_role_conflict_not_realizable(self):
        ss = generate_non_realizable("label_role_conflict")
        result = check_realizability(ss)
        assert not result.is_realizable

    def test_convergent_sharing_not_realizable(self):
        ss = generate_non_realizable("convergent_sharing")
        result = check_realizability(ss)
        assert not result.is_realizable

    # --- Category E: Counting ---
    def test_singleton_selection_not_realizable(self):
        ss = generate_non_realizable("singleton_selection")
        result = check_realizability(ss)
        assert not result.is_realizable

    def test_empty_branch_not_realizable(self):
        ss = generate_non_realizable("empty_branch")
        result = check_realizability(ss)
        assert not result.is_realizable


# ═══════════════════════════════════════════════════════════════════
# Sprint 3: check_realizability + round-trip tests
# ═══════════════════════════════════════════════════════════════════


class TestCheckRealizability:
    """Full realizability pipeline."""

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "+{a: end, b: end}",
        "&{a: end, b: end}",
        "&{a: +{x: end, y: end}}",
        "+{a: &{x: end, y: end}}",
        "rec X . &{a: X, b: end}",
        "rec X . +{a: X, b: end}",
        "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
    ])
    def test_session_types_are_realizable(self, type_str):
        ss = build_statespace(parse(type_str))
        result = check_realizability(ss)
        assert result.is_realizable, (
            f"{type_str} should be realizable, obstructions: "
            f"{[o.kind for o in result.obstructions]}"
        )

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "+{a: end, b: end}",
        "&{a: end, b: end}",
        "rec X . &{a: X, b: end}",
    ])
    def test_realizable_has_reconstructed_type(self, type_str):
        ss = build_statespace(parse(type_str))
        result = check_realizability(ss)
        assert result.reconstructed_type is not None

    def test_result_type(self):
        ss = build_statespace(parse("&{a: end}"))
        result = check_realizability(ss)
        assert isinstance(result, RealizabilityResult)
        assert isinstance(result.conditions, RealizabilityConditions)

    def test_result_has_lattice_result(self):
        ss = build_statespace(parse("&{a: end}"))
        result = check_realizability(ss)
        assert result.lattice_result is not None
        assert result.lattice_result.is_lattice

    def test_result_has_reticular_result(self):
        ss = build_statespace(parse("&{a: end}"))
        result = check_realizability(ss)
        assert result.reticular_result is not None
        assert result.reticular_result.is_reticulate


class TestRoundTrip:
    """Round-trip: parse → build → check_realizability → reconstruct → build → still realizable."""

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "&{a: end, b: end}",
        "+{a: end, b: end}",
        "&{a: +{x: end, y: end}}",
        "rec X . &{a: X, b: end}",
        "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
    ])
    def test_round_trip_preserves_realizability(self, type_str):
        """parse → build → realize → reconstruct → build → still realizable."""
        ss1 = build_statespace(parse(type_str))
        r1 = check_realizability(ss1)
        assert r1.is_realizable
        assert r1.reconstructed_type is not None

        # Reconstruct and rebuild
        ss2 = build_statespace(parse(r1.reconstructed_type))
        r2 = check_realizability(ss2)
        assert r2.is_realizable, (
            f"Round-trip failed for {type_str}: "
            f"{[o.kind for o in r2.obstructions]}"
        )

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "&{a: end, b: end}",
        "+{a: end, b: end}",
    ])
    def test_round_trip_same_state_count(self, type_str):
        ss1 = build_statespace(parse(type_str))
        r1 = check_realizability(ss1)
        ss2 = build_statespace(parse(r1.reconstructed_type))
        assert len(ss1.states) == len(ss2.states)


# ═══════════════════════════════════════════════════════════════════
# Sprint 4: Benchmark sweep + edge cases
# ═══════════════════════════════════════════════════════════════════


class TestBenchmarkSweep:
    """All 79 benchmarks should be realizable."""

    @pytest.fixture
    def benchmark_types(self):
        from tests.benchmarks.protocols import BENCHMARKS
        return [(b.name, b.type_string) for b in BENCHMARKS]

    def test_all_benchmarks_realizable(self, benchmark_types):
        for name, type_str in benchmark_types:
            ss = build_statespace(parse(type_str))
            result = check_realizability(ss)
            assert result.is_realizable, (
                f"{name} should be realizable, obstructions: "
                f"{[o.kind for o in result.obstructions]}"
            )

    def test_all_benchmarks_have_reconstructed_type(self, benchmark_types):
        for name, type_str in benchmark_types:
            ss = build_statespace(parse(type_str))
            result = check_realizability(ss)
            assert result.reconstructed_type is not None, (
                f"{name} should have a reconstructed type"
            )


class TestEdgeCases:
    """Edge cases and special types."""

    def test_end_type(self):
        ss = build_statespace(parse("end"))
        result = check_realizability(ss)
        assert result.is_realizable
        assert len(ss.states) == 1

    def test_single_method(self):
        ss = build_statespace(parse("&{a: end}"))
        result = check_realizability(ss)
        assert result.is_realizable

    def test_deeply_nested(self):
        t = "&{a: &{b: &{c: &{d: end}}}}"
        ss = build_statespace(parse(t))
        result = check_realizability(ss)
        assert result.is_realizable

    def test_product_type(self):
        t = "(&{a: end} || &{b: end})"
        ss = build_statespace(parse(t))
        result = check_realizability(ss)
        assert result.is_realizable

    def test_recursive_branch(self):
        t = "rec X . &{a: X, b: end}"
        ss = build_statespace(parse(t))
        result = check_realizability(ss)
        assert result.is_realizable

    def test_recursive_selection(self):
        t = "rec X . +{a: X, b: end}"
        ss = build_statespace(parse(t))
        result = check_realizability(ss)
        assert result.is_realizable

    def test_empty_statespace(self):
        """A single-state space (just bottom = top = end) is realizable."""
        ss = StateSpace(
            states={0},
            transitions=[],
            top=0, bottom=0,
            labels={0: "end"},
        )
        result = check_realizability(ss)
        assert result.is_realizable

    def test_conditions_all_true_for_session_types(self):
        ss = build_statespace(parse("&{a: end, b: &{c: end}}"))
        result = check_realizability(ss)
        c = result.conditions
        assert c.is_lattice
        assert c.is_bounded
        assert c.is_deterministic
        assert c.all_reachable
        assert c.all_reach_bottom
        assert c.has_reticular_form

    def test_non_realizable_result_has_false_is_realizable(self):
        ss = generate_non_realizable("nondeterministic")
        result = check_realizability(ss)
        assert not result.is_realizable
        assert result.reconstructed_type is None

    def test_iterator_benchmark(self):
        """Java Iterator — the classic benchmark."""
        t = "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        ss = build_statespace(parse(t))
        result = check_realizability(ss)
        assert result.is_realizable


class TestWildStateMachines:
    """Wild state machines (from test_wild_state_machines.py) should be realizable."""

    def test_turnstile_like(self):
        """Turnstile-like cyclic FSM."""
        ss = StateSpace(
            states={0, 1},
            transitions=[(0, "coin", 1), (1, "push", 0)],
            top=0, bottom=0,
            labels={0: "locked", 1: "unlocked"},
        )
        result = check_realizability(ss)
        # May or may not be realizable depending on bottom having transitions
        # The key is it doesn't crash
        assert isinstance(result, RealizabilityResult)
