"""Tests for exhaustive session type enumeration (Step 6: universality)."""

import pytest

from reticulate.enumerate_types import (
    EnumerationConfig,
    UniversalityResult,
    check_universality,
    enumerate_session_types,
)
from reticulate.lattice import check_lattice
from reticulate.parser import Branch, End, Rec, Select, SessionType, Var, pretty
from reticulate.statespace import build_statespace


# ---------------------------------------------------------------------------
# Enumeration tests
# ---------------------------------------------------------------------------


class TestEnumerateTypes:
    """Test that type enumeration generates expected types."""

    def test_depth_0_yields_end(self):
        config = EnumerationConfig(max_depth=0, labels=("a",))
        types = list(enumerate_session_types(config))
        assert End() in types
        assert len(types) == 1  # only End at depth 0

    def test_depth_1_single_label_branch(self):
        config = EnumerationConfig(
            max_depth=1, labels=("a",), max_branch_width=1,
            include_parallel=False, include_recursion=False,
            include_selection=False,
        )
        types = list(enumerate_session_types(config))
        # End + &{a: end}
        assert End() in types
        assert Branch((("a", End()),)) in types

    def test_depth_1_includes_selection(self):
        config = EnumerationConfig(
            max_depth=1, labels=("a",), max_branch_width=1,
            include_parallel=False, include_recursion=False,
            include_selection=True,
        )
        types = list(enumerate_session_types(config))
        assert Select((("a", End()),)) in types

    def test_depth_2_includes_recursion(self):
        config = EnumerationConfig(
            max_depth=2, labels=("a",), max_branch_width=1,
            include_parallel=False, include_recursion=True,
            include_selection=False,
        )
        types = list(enumerate_session_types(config))
        # rec X . &{a: X} needs depth 2 (rec consumes 1, branch consumes 1)
        assert Rec("X", Branch((("a", Var("X")),))) in types

    def test_no_trivial_recursion(self):
        """rec X . end and rec X . X should NOT be generated."""
        config = EnumerationConfig(
            max_depth=2, labels=("a",), max_branch_width=1,
            include_parallel=False, include_recursion=True,
            include_selection=False,
        )
        types = list(enumerate_session_types(config))
        assert Rec("X", End()) not in types
        assert Rec("X", Var("X")) not in types

    def test_enumeration_count_grows_with_depth(self):
        config_d1 = EnumerationConfig(
            max_depth=1, labels=("a",), max_branch_width=1,
            include_parallel=False, include_recursion=False,
            include_selection=False,
        )
        config_d2 = EnumerationConfig(
            max_depth=2, labels=("a",), max_branch_width=1,
            include_parallel=False, include_recursion=False,
            include_selection=False,
        )
        n1 = len(list(enumerate_session_types(config_d1)))
        n2 = len(list(enumerate_session_types(config_d2)))
        assert n2 > n1

    def test_enumeration_count_grows_with_labels(self):
        config_1 = EnumerationConfig(
            max_depth=1, labels=("a",), max_branch_width=1,
            include_parallel=False, include_recursion=False,
            include_selection=False,
        )
        config_2 = EnumerationConfig(
            max_depth=1, labels=("a", "b"), max_branch_width=2,
            include_parallel=False, include_recursion=False,
            include_selection=False,
        )
        n1 = len(list(enumerate_session_types(config_1)))
        n2 = len(list(enumerate_session_types(config_2)))
        assert n2 > n1

    def test_two_labels_width_2_branch_only(self):
        """Branch-only with 2 labels, width up to 2, depth 1."""
        config = EnumerationConfig(
            max_depth=1, labels=("a", "b"), max_branch_width=2,
            include_parallel=False, include_recursion=False,
            include_selection=False,
        )
        types = list(enumerate_session_types(config))
        # end, &{a:end}, &{b:end}, &{a:end, b:end}
        assert End() in types
        assert Branch((("a", End()),)) in types
        assert Branch((("b", End()),)) in types
        assert Branch((("a", End()), ("b", End()))) in types
        assert len(types) == 4


# ---------------------------------------------------------------------------
# Universality tests — small configurations
# ---------------------------------------------------------------------------


class TestUniversalitySmall:
    """Test universality on small type spaces."""

    def test_depth_0(self):
        config = EnumerationConfig(max_depth=0, labels=("a",))
        result = check_universality(config)
        assert result.is_universal
        assert result.total_types == 1  # just End
        assert result.total_lattices == 1

    def test_depth_1_branch_only_1_label(self):
        config = EnumerationConfig(
            max_depth=1, labels=("a",), max_branch_width=1,
            include_parallel=False, include_recursion=False,
            include_selection=False,
        )
        result = check_universality(config)
        assert result.is_universal

    def test_depth_1_branch_only_2_labels(self):
        config = EnumerationConfig(
            max_depth=1, labels=("a", "b"), max_branch_width=2,
            include_parallel=False, include_recursion=False,
            include_selection=False,
        )
        result = check_universality(config)
        assert result.is_universal

    def test_depth_1_with_selection(self):
        config = EnumerationConfig(
            max_depth=1, labels=("a", "b"), max_branch_width=2,
            include_parallel=False, include_recursion=False,
            include_selection=True,
        )
        result = check_universality(config)
        assert result.is_universal

    def test_depth_1_with_recursion(self):
        config = EnumerationConfig(
            max_depth=1, labels=("a", "b"), max_branch_width=2,
            include_parallel=False, include_recursion=True,
            include_selection=False,
        )
        result = check_universality(config)
        assert result.is_universal

    def test_depth_1_with_parallel(self):
        config = EnumerationConfig(
            max_depth=1, labels=("a",), max_branch_width=1,
            include_parallel=True, include_recursion=False,
            include_selection=False,
        )
        result = check_universality(config)
        assert result.is_universal

    def test_depth_1_full(self):
        """Full grammar, depth 1, 2 labels."""
        config = EnumerationConfig(
            max_depth=1, labels=("a", "b"), max_branch_width=2,
            include_parallel=True, include_recursion=True,
            include_selection=True,
        )
        result = check_universality(config)
        assert result.is_universal
        assert result.total_types > 5  # should be a non-trivial number


class TestUniversalityMedium:
    """Test universality on medium-sized configurations."""

    def test_depth_2_branch_only_2_labels(self):
        """Branch-only, depth 2, 2 labels — exercises nested branches."""
        config = EnumerationConfig(
            max_depth=2, labels=("a", "b"), max_branch_width=2,
            include_parallel=False, include_recursion=False,
            include_selection=False,
        )
        result = check_universality(config)
        assert result.is_universal
        assert result.total_types > 10

    def test_depth_2_branch_and_selection(self):
        """Branch + selection, depth 2, 2 labels."""
        config = EnumerationConfig(
            max_depth=2, labels=("a", "b"), max_branch_width=2,
            include_parallel=False, include_recursion=False,
            include_selection=True,
        )
        result = check_universality(config)
        assert result.is_universal

    def test_depth_2_with_recursion(self):
        """Branch + recursion, depth 2, 1 label."""
        config = EnumerationConfig(
            max_depth=2, labels=("a",), max_branch_width=1,
            include_parallel=False, include_recursion=True,
            include_selection=False,
        )
        result = check_universality(config)
        assert result.is_universal

    def test_depth_2_with_parallel(self):
        """Branch + parallel, depth 2, 1 label."""
        config = EnumerationConfig(
            max_depth=2, labels=("a",), max_branch_width=1,
            include_parallel=True, include_recursion=False,
            include_selection=False,
        )
        result = check_universality(config)
        assert result.is_universal


# ---------------------------------------------------------------------------
# Specific known types — regression checks
# ---------------------------------------------------------------------------


class TestKnownTypes:
    """Verify specific types that are known to be lattices."""

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "&{a: end, b: end}",
        "&{a: &{b: end}}",
        "+{a: end}",
        "+{a: end, b: end}",
        "&{a: end, b: &{c: end}}",
        "rec X . &{a: X, b: end}",
        "&{a: +{x: end, y: end}, b: end}",
    ])
    def test_known_lattice(self, type_str: str):
        from reticulate.parser import parse
        ast = parse(type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert result.is_lattice, f"{type_str} should be a lattice but got counterexample: {result.counterexample}"


class TestNonTerminatingCounterexamples:
    """Non-terminating types fail the lattice check — this is expected."""

    @pytest.mark.parametrize("type_str", [
        "&{a: end, b: rec X . &{a: X}}",
        "&{a: end, b: rec X . &{b: X}}",
        "&{a: end, b: rec X . +{a: X}}",
    ])
    def test_nonterminating_is_not_lattice(self, type_str: str):
        """Non-terminating recursive branches create unreachable bottom."""
        from reticulate.parser import parse
        from reticulate.termination import is_terminating
        ast = parse(type_str)
        assert not is_terminating(ast), f"{type_str} should be non-terminating"
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert not result.is_lattice, f"{type_str} should NOT be a lattice"

    def test_simplest_counterexample(self):
        """The simplest counterexample: &{a: end, b: rec X . &{a: X}}."""
        from reticulate.parser import parse
        ast = parse("&{a: end, b: rec X . &{a: X}}")
        ss = build_statespace(ast)
        result = check_lattice(ss)
        assert not result.is_lattice
        assert result.counterexample is not None
        _, _, kind = result.counterexample
        assert kind == "no_meet"


class TestTerminatingUniversality:
    """Key result: all terminating types are lattices (Step 6)."""

    def test_depth_2_full_terminating(self):
        """Depth 2, all constructors, terminating only — should be universal."""
        config = EnumerationConfig(
            max_depth=2, labels=("a", "b"), max_branch_width=2,
            include_parallel=True, include_recursion=True,
            include_selection=True, require_terminating=True,
        )
        result = check_universality(config)
        assert result.is_universal
        assert result.total_types > 100

    def test_depth_3_nonterminating_has_counterexamples(self):
        """Depth 3 allowing non-terminating → counterexamples found.

        The simplest counterexample &{a: end, b: rec X . &{a: X}} needs
        depth 3: outer branch (1) + rec (1) + inner branch (1).
        """
        config = EnumerationConfig(
            max_depth=3, labels=("a", "b"), max_branch_width=2,
            include_parallel=False, include_recursion=True,
            include_selection=False, require_terminating=False,
        )
        result = check_universality(config)
        assert not result.is_universal
        assert len(result.counterexamples) > 0


# ---------------------------------------------------------------------------
# Result type tests
# ---------------------------------------------------------------------------


class TestUniversalityResult:
    """Test UniversalityResult properties."""

    def test_is_universal_true(self):
        result = UniversalityResult(
            total_types=10, total_lattices=10,
            counterexamples=(), config=EnumerationConfig(),
        )
        assert result.is_universal

    def test_is_universal_false(self):
        from reticulate.lattice import LatticeResult
        fake_lr = LatticeResult(
            is_lattice=False, has_top=True, has_bottom=True,
            all_meets_exist=False, all_joins_exist=True,
            num_scc=3, counterexample=(1, 2, "no_meet"), scc_map={},
        )
        result = UniversalityResult(
            total_types=10, total_lattices=9,
            counterexamples=(("fake", fake_lr),),
            config=EnumerationConfig(),
        )
        assert not result.is_universal
