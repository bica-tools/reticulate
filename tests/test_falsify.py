"""Tests for reticulate.falsify — counterexample finder for session type conjectures."""

from __future__ import annotations

import pytest

from reticulate.falsify import (
    FalsifyResult,
    Predicate,
    binary_implies_distributive,
    dual_preserves_lattice,
    falsify,
    falsify_distributivity_conjecture,
    is_always_lattice,
    is_distributive,
)
from reticulate.lattice import LatticeResult, check_lattice
from reticulate.parser import SessionType, pretty
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# 1. M3 counterexample found by falsify_distributivity_conjecture
# ---------------------------------------------------------------------------

class TestFalsifyDistributivityConjecture:
    """The M3 diamond counterexample should be found."""

    def test_finds_counterexample(self) -> None:
        result = falsify_distributivity_conjecture(depth=2, labels=("a", "b", "c"))
        assert result.falsified is True
        assert result.counterexample is not None
        assert result.counterexample_ast is not None

    def test_counterexample_is_non_distributive(self) -> None:
        result = falsify_distributivity_conjecture(depth=2, labels=("a", "b", "c"))
        assert result.falsified
        # Verify the counterexample is indeed non-distributive
        ast = result.counterexample_ast
        assert ast is not None
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice is True
        assert is_distributive(ast, ss, lr) is False

    def test_counterexample_has_no_parallel(self) -> None:
        from reticulate.falsify import _contains_parallel
        result = falsify_distributivity_conjecture(depth=2, labels=("a", "b", "c"))
        assert result.falsified
        assert not _contains_parallel(result.counterexample_ast)

    def test_result_has_correct_property_name(self) -> None:
        result = falsify_distributivity_conjecture(depth=2, labels=("a", "b", "c"))
        assert result.property_name == "no_parallel_no_nesting_implies_distributive"

    def test_checked_count_is_positive(self) -> None:
        result = falsify_distributivity_conjecture(depth=2, labels=("a", "b", "c"))
        assert result.checked > 0

    def test_details_mentions_counterexample(self) -> None:
        result = falsify_distributivity_conjecture(depth=2, labels=("a", "b", "c"))
        assert "Counterexample" in result.details


# ---------------------------------------------------------------------------
# 2. is_always_lattice NOT falsified for terminating types
# ---------------------------------------------------------------------------

class TestIsAlwaysLattice:
    """Universality conjecture: all terminating types produce lattices."""

    def test_not_falsified_depth_2(self) -> None:
        result = falsify(
            is_always_lattice,
            property_name="universality",
            depth=2,
            width=2,
            labels=("a", "b"),
            require_terminating=True,
            include_parallel=False,
            include_recursion=False,
        )
        assert result.falsified is False
        assert result.checked > 0

    def test_not_falsified_with_selection(self) -> None:
        result = falsify(
            is_always_lattice,
            property_name="universality",
            depth=2,
            width=2,
            labels=("a", "b"),
            require_terminating=True,
            include_parallel=False,
            include_recursion=False,
            include_selection=True,
        )
        assert result.falsified is False

    def test_details_mentions_all_hold(self) -> None:
        result = falsify(
            is_always_lattice,
            property_name="universality",
            depth=1,
            width=2,
            labels=("a",),
            require_terminating=True,
            include_parallel=False,
            include_recursion=False,
        )
        assert result.falsified is False
        assert "holds" in result.details


# ---------------------------------------------------------------------------
# 3. dual_preserves_lattice NOT falsified
# ---------------------------------------------------------------------------

class TestDualPreservesLattice:
    """Duality preserves the lattice property."""

    def test_not_falsified_depth_2(self) -> None:
        result = falsify(
            dual_preserves_lattice,
            property_name="dual_preserves_lattice",
            depth=2,
            width=2,
            labels=("a", "b"),
            require_terminating=True,
            include_parallel=False,
            include_recursion=False,
        )
        assert result.falsified is False
        assert result.checked > 0

    def test_not_falsified_with_selection(self) -> None:
        result = falsify(
            dual_preserves_lattice,
            property_name="dual_preserves_lattice",
            depth=2,
            width=2,
            labels=("a", "b"),
            require_terminating=True,
            include_parallel=False,
            include_recursion=False,
            include_selection=True,
        )
        assert result.falsified is False


# ---------------------------------------------------------------------------
# 4. binary_implies_distributive NOT falsified for binary-only types
# ---------------------------------------------------------------------------

class TestBinaryImpliesDistributive:
    """Binary branching (<=2 choices) always yields distributive lattices."""

    def test_not_falsified_depth_2(self) -> None:
        result = falsify(
            binary_implies_distributive,
            property_name="binary_implies_distributive",
            depth=2,
            width=2,
            labels=("a", "b"),
            require_terminating=True,
            include_parallel=False,
            include_recursion=False,
        )
        assert result.falsified is False
        assert result.checked > 0

    def test_falsified_at_depth_3(self) -> None:
        """At depth 3, even binary branching can produce non-distributive lattices.

        The counterexample &{a: &{a: &{a: end}}, b: &{a: end}} has an M3
        sublattice despite all branches having at most 2 choices.
        """
        result = falsify(
            binary_implies_distributive,
            property_name="binary_implies_distributive",
            depth=3,
            width=2,
            labels=("a", "b"),
            require_terminating=True,
            include_parallel=False,
            include_recursion=False,
            include_selection=False,
        )
        assert result.falsified is True
        assert result.counterexample is not None


# ---------------------------------------------------------------------------
# 5. Custom predicate works
# ---------------------------------------------------------------------------

class TestCustomPredicate:
    """Users can pass arbitrary predicates to falsify."""

    def test_trivially_false_predicate(self) -> None:
        """A predicate that always returns False should find a counterexample immediately."""

        def always_false(
            ast: SessionType, ss: StateSpace, lr: LatticeResult
        ) -> bool:
            return False

        result = falsify(
            always_false,
            property_name="always_false",
            depth=1,
            width=1,
            labels=("a",),
            require_terminating=True,
        )
        assert result.falsified is True
        assert result.checked == 1
        assert result.counterexample is not None

    def test_trivially_true_predicate(self) -> None:
        """A predicate that always returns True should find no counterexample."""

        def always_true(
            ast: SessionType, ss: StateSpace, lr: LatticeResult,
        ) -> bool:
            return True

        result = falsify(
            always_true,
            property_name="always_true",
            depth=1,
            width=2,
            labels=("a",),
            require_terminating=True,
        )
        assert result.falsified is False
        assert result.checked > 0

    def test_state_count_predicate(self) -> None:
        """Predicate that checks state count <= 3."""

        def at_most_3_states(
            ast: SessionType, ss: StateSpace, lr: LatticeResult,
        ) -> bool:
            return len(ss.states) <= 3

        result = falsify(
            at_most_3_states,
            property_name="at_most_3_states",
            depth=2,
            width=2,
            labels=("a", "b"),
            require_terminating=True,
            include_parallel=False,
            include_recursion=False,
            include_selection=False,
        )
        # There should be types with > 3 states at depth 2, width 2
        assert result.falsified is True
        assert result.counterexample is not None
        # Verify the counterexample indeed has > 3 states
        ast = result.counterexample_ast
        ss = build_statespace(ast)
        assert len(ss.states) > 3

    def test_predicate_that_raises(self) -> None:
        """A predicate that raises an exception should be treated as falsified."""

        def crashy(
            ast: SessionType, ss: StateSpace, lr: LatticeResult,
        ) -> bool:
            raise RuntimeError("boom")

        result = falsify(
            crashy,
            property_name="crashy",
            depth=1,
            width=1,
            labels=("a",),
            require_terminating=True,
        )
        assert result.falsified is True
        assert "exception" in result.details

    def test_max_checks_limit(self) -> None:
        """max_checks should limit the number of types checked."""

        def always_true(
            ast: SessionType, ss: StateSpace, lr: LatticeResult,
        ) -> bool:
            return True

        result = falsify(
            always_true,
            property_name="limited",
            depth=3,
            width=3,
            labels=("a", "b", "c"),
            require_terminating=True,
            max_checks=5,
        )
        assert result.falsified is False
        assert result.checked <= 5

    def test_custom_property_name(self) -> None:
        """The property name should be recorded in the result."""
        result = falsify(
            is_always_lattice,
            property_name="my_custom_name",
            depth=1,
            width=1,
            labels=("a",),
            require_terminating=True,
        )
        assert result.property_name == "my_custom_name"


# ---------------------------------------------------------------------------
# 6. FalsifyResult dataclass
# ---------------------------------------------------------------------------

class TestFalsifyResult:
    """FalsifyResult is a frozen dataclass with the right fields."""

    def test_frozen(self) -> None:
        r = FalsifyResult(
            property_name="test",
            counterexample=None,
            counterexample_ast=None,
            checked=0,
            falsified=False,
            details="ok",
        )
        with pytest.raises(AttributeError):
            r.falsified = True  # type: ignore[misc]

    def test_fields(self) -> None:
        r = FalsifyResult(
            property_name="test",
            counterexample="end",
            counterexample_ast=None,
            checked=42,
            falsified=True,
            details="found it",
        )
        assert r.property_name == "test"
        assert r.counterexample == "end"
        assert r.checked == 42
        assert r.falsified is True
        assert r.details == "found it"


# ---------------------------------------------------------------------------
# 7. Built-in predicates in isolation
# ---------------------------------------------------------------------------

class TestBuiltinPredicates:
    """Test built-in predicates on known types."""

    def test_end_is_lattice(self) -> None:
        from reticulate.parser import parse
        ast = parse("end")
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert is_always_lattice(ast, ss, lr) is True

    def test_end_is_distributive(self) -> None:
        from reticulate.parser import parse
        ast = parse("end")
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert is_distributive(ast, ss, lr) is True

    def test_m3_is_not_distributive(self) -> None:
        from reticulate.parser import parse
        ast = parse("&{a: &{a: end}, b: &{b: end}, c: &{c: end}}")
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice is True
        assert is_distributive(ast, ss, lr) is False

    def test_dual_preserves_simple(self) -> None:
        from reticulate.parser import parse
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert dual_preserves_lattice(ast, ss, lr) is True

    def test_binary_simple(self) -> None:
        from reticulate.parser import parse
        ast = parse("&{a: end, b: end}")
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert binary_implies_distributive(ast, ss, lr) is True
