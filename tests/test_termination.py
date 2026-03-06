"""Tests for termination checking and WF-Par well-formedness (termination.py)."""

import pytest

from reticulate.parser import (
    Branch,
    Continuation,
    End,
    Parallel,
    Rec,
    Select,
    Sequence,
    Var,
    Wait,
    parse,
)
from reticulate.termination import (
    TerminationResult,
    WFParallelResult,
    check_termination,
    check_wf_parallel,
    is_terminating,
    _free_vars,
    _bound_vars,
    _contains_parallel,
    _has_exit_path,
)


# ===================================================================
# Helpers
# ===================================================================

def _term(source: str) -> TerminationResult:
    """Parse and check termination."""
    return check_termination(parse(source))


def _wf(source: str) -> WFParallelResult:
    """Parse and check WF-Par."""
    return check_wf_parallel(parse(source))


# ===================================================================
# TestTerminatingSimple — non-recursive types (trivially terminating)
# ===================================================================

class TestTerminatingSimple:
    """Non-recursive session types are always terminating."""

    def test_end(self) -> None:
        assert is_terminating(parse("end")) is True

    def test_sequence(self) -> None:
        assert is_terminating(parse("&{a: &{b: end}}")) is True

    def test_branch(self) -> None:
        assert is_terminating(parse("&{m: end}")) is True

    def test_select(self) -> None:
        assert is_terminating(parse("+{OK: end, ERR: end}")) is True

    def test_parallel_simple(self) -> None:
        assert is_terminating(parse("(&{a: end} || &{b: end})")) is True

    def test_branch_multiple(self) -> None:
        assert is_terminating(parse("&{a: end, b: end, c: end}")) is True

    def test_nested_branch_select(self) -> None:
        assert is_terminating(parse("&{m: +{OK: end, ERR: end}}")) is True


# ===================================================================
# TestTerminatingRecursion — recursive types with exit paths
# ===================================================================

class TestTerminatingRecursion:
    """Recursive types that have at least one exit path."""

    def test_rec_with_exit(self) -> None:
        """rec X . &{read: X, done: end} — done->End is the exit."""
        r = _term("rec X . &{read: X, done: end}")
        assert r.is_terminating is True
        assert r.non_terminating_vars == ()

    def test_rec_select_exit(self) -> None:
        """rec X . +{OK: X, DONE: end} — DONE->End is the exit."""
        assert is_terminating(parse("rec X . +{OK: X, DONE: end}")) is True

    def test_nested_rec_both_terminating(self) -> None:
        """rec X . &{a: rec Y . &{b: Y, done: X}, done: end}
        Y exits to X (an outer var), X exits to end."""
        r = _term("rec X . &{a: rec Y . &{b: Y, done: X}, done: end}")
        assert r.is_terminating is True

    def test_rec_with_parallel_exit(self) -> None:
        """rec X . &{go: (&{a: end} || &{b: end}), loop: X}"""
        assert is_terminating(
            parse("rec X . &{go: (&{a: end} || &{b: end}), loop: X}")
        ) is True

    def test_rec_sequence_exit(self) -> None:
        """rec X . &{step: X, done: &{a: &{b: end}}}"""
        assert is_terminating(parse("rec X . &{step: X, done: &{a: &{b: end}}}")) is True

    def test_java_iterator(self) -> None:
        """Classic benchmark: rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"""
        assert is_terminating(
            parse("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        ) is True

    def test_file_object(self) -> None:
        """&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"""
        assert is_terminating(
            parse("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")
        ) is True


# ===================================================================
# TestNonTerminating — recursive types without exit paths
# ===================================================================

class TestNonTerminating:
    """Recursive types where all paths lead back to the recursion variable."""

    def test_trivial_loop(self) -> None:
        """rec X . X — immediate self-reference."""
        r = _term("rec X . X")
        assert r.is_terminating is False
        assert "X" in r.non_terminating_vars

    def test_all_branches_loop(self) -> None:
        """rec X . &{loop: X} — single branch leads back to X."""
        r = _term("rec X . &{loop: X}")
        assert r.is_terminating is False
        assert "X" in r.non_terminating_vars

    def test_multiple_branches_all_loop(self) -> None:
        """rec X . &{a: X, b: X} — all branches lead back to X."""
        r = _term("rec X . &{a: X, b: X}")
        assert r.is_terminating is False

    def test_select_all_loop(self) -> None:
        """rec X . +{OK: X, ERR: X} — all selections loop."""
        r = _term("rec X . +{OK: X, ERR: X}")
        assert r.is_terminating is False

    def test_nested_all_loop(self) -> None:
        """rec X . &{a: +{OK: X, ERR: X}} — all paths loop."""
        r = _term("rec X . &{a: +{OK: X, ERR: X}}")
        assert r.is_terminating is False

    def test_sequence_left_loops(self) -> None:
        """Continuation(Var(X), End) — left=Var(X) fails, AND -> False.

        Note: we build the AST directly because the parser no longer
        supports sequencing sugar ``X . end``.
        """
        node = Rec("X", Continuation(Var("X"), End()))
        r = check_termination(node)
        assert r.is_terminating is False
        assert "X" in r.non_terminating_vars


# ===================================================================
# TestTerminationResult — field inspection
# ===================================================================

class TestTerminationResult:
    """Inspect TerminationResult fields in detail."""

    def test_terminating_result_fields(self) -> None:
        r = check_termination(parse("end"))
        assert r.is_terminating is True
        assert r.non_terminating_vars == ()
        assert isinstance(r, TerminationResult)

    def test_non_terminating_single_var(self) -> None:
        r = check_termination(parse("rec X . X"))
        assert r.is_terminating is False
        assert r.non_terminating_vars == ("X",)

    def test_multiple_non_terminating_vars(self) -> None:
        """Two nested non-terminating Rec nodes."""
        r = check_termination(parse("rec X . &{a: rec Y . Y, b: X}"))
        # X has exit? &{a: rec Y . Y, b: X} — 'a' branch: rec Y.Y,
        # _has_exit_path for X in rec Y.Y checks body Y, which is Var(Y),
        # name != forbidden(X), so returns True. So X IS terminating.
        # But Y has only Y in body -> non-terminating.
        assert "Y" in r.non_terminating_vars
        assert r.is_terminating is False

    def test_two_independent_non_terminating(self) -> None:
        """(rec X . X || rec Y . Y) — both fail."""
        node = Parallel(Rec("X", Var("X")), Rec("Y", Var("Y")))
        r = check_termination(node)
        assert r.is_terminating is False
        assert "X" in r.non_terminating_vars
        assert "Y" in r.non_terminating_vars

    def test_frozen_result(self) -> None:
        r = check_termination(parse("end"))
        with pytest.raises(AttributeError):
            r.is_terminating = False  # type: ignore[misc]


# ===================================================================
# TestWFParallel — well-formedness of parallel composition
# ===================================================================

class TestWFParallel:
    """WF-Par checks: termination, cross-branch vars, nested parallel."""

    def test_simple_ok(self) -> None:
        """(&{a: end} || &{b: end}) — well-formed."""
        r = _wf("(&{a: end} || &{b: end})")
        assert r.is_well_formed is True
        assert r.errors == ()

    def test_recursive_ok(self) -> None:
        """(rec X . &{a: X, done: end} || rec Y . &{b: Y, done: end})"""
        r = _wf(
            "(rec X . &{a: X, done: end} || rec Y . &{b: Y, done: end})"
        )
        assert r.is_well_formed is True

    def test_non_terminating_left(self) -> None:
        """(rec X . X || end) — left branch is non-terminating."""
        r = _wf("(rec X . X || end)")
        assert r.is_well_formed is False
        assert any("non-terminating" in e for e in r.errors)

    def test_non_terminating_right(self) -> None:
        """(end || rec Y . Y) — right branch is non-terminating."""
        r = _wf("(end || rec Y . Y)")
        assert r.is_well_formed is False
        assert any("non-terminating" in e for e in r.errors)

    def test_cross_branch_vars(self) -> None:
        """Free in left, bound in right -> violation."""
        # rec X . (X || rec X . X) — X is free in left, bound in right
        # But this is contrived. Let's build the AST directly.
        node = Parallel(Var("X"), Rec("X", Var("X")))
        r = check_wf_parallel(node)
        assert r.is_well_formed is False
        assert any("cross-branch" in e for e in r.errors)

    def test_nested_parallel_left(self) -> None:
        """((&{a: end} || &{b: end}) || &{c: end}) — left contains nested parallel."""
        r = _wf("((&{a: end} || &{b: end}) || &{c: end})")
        assert r.is_well_formed is False
        assert any("nested" in e for e in r.errors)

    def test_nested_parallel_right(self) -> None:
        """(&{a: end} || (&{b: end} || &{c: end})) — right contains nested parallel."""
        r = _wf("(&{a: end} || (&{b: end} || &{c: end}))")
        assert r.is_well_formed is False
        assert any("nested" in e for e in r.errors)

    def test_no_parallel_is_well_formed(self) -> None:
        """A type with no parallel is trivially well-formed."""
        r = _wf("rec X . &{a: X, done: end}")
        assert r.is_well_formed is True
        assert r.errors == ()

    def test_wf_result_frozen(self) -> None:
        r = _wf("end")
        with pytest.raises(AttributeError):
            r.is_well_formed = False  # type: ignore[misc]


# ===================================================================
# TestBenchmarks — all 15 benchmarks should be terminating
# ===================================================================

class TestBenchmarks:
    """Verify termination for all 15 benchmark protocols."""

    @pytest.fixture
    def benchmarks(self) -> list:
        from tests.benchmarks.protocols import BENCHMARKS
        return BENCHMARKS

    def test_all_benchmarks_terminating(self, benchmarks: list) -> None:
        for bp in benchmarks:
            node = parse(bp.type_string)
            r = check_termination(node)
            assert r.is_terminating is True, (
                f"Benchmark {bp.name!r} should be terminating, "
                f"but non-terminating vars: {r.non_terminating_vars}"
            )

    def test_all_benchmarks_wf_parallel(self, benchmarks: list) -> None:
        """All benchmarks with parallel should be well-formed."""
        for bp in benchmarks:
            if bp.uses_parallel:
                node = parse(bp.type_string)
                r = check_wf_parallel(node)
                assert r.is_well_formed is True, (
                    f"Benchmark {bp.name!r} should be WF-Par, "
                    f"but errors: {r.errors}"
                )


# ===================================================================
# TestHelpers — internal helper functions
# ===================================================================

class TestHelpers:
    """Tests for _free_vars, _bound_vars, _contains_parallel, _has_exit_path."""

    # -- _free_vars --

    def test_free_vars_end(self) -> None:
        assert _free_vars(End()) == set()

    def test_free_vars_var(self) -> None:
        assert _free_vars(Var("X")) == {"X"}

    def test_free_vars_rec_binds(self) -> None:
        """rec X . X -> no free vars (X is bound)."""
        assert _free_vars(Rec("X", Var("X"))) == set()

    def test_free_vars_rec_with_other(self) -> None:
        """rec X . &{a: X, b: Y} -> {Y}."""
        node = Rec("X", Branch((("a", Var("X")), ("b", Var("Y")))))
        assert _free_vars(node) == {"Y"}

    def test_free_vars_parallel(self) -> None:
        node = Parallel(Var("A"), Var("B"))
        assert _free_vars(node) == {"A", "B"}

    def test_free_vars_sequence(self) -> None:
        node = Continuation(Var("X"), End())
        assert _free_vars(node) == {"X"}

    def test_free_vars_wait(self) -> None:
        """Wait has no free vars (terminal node like End)."""
        assert _free_vars(Wait()) == set()

    # -- _bound_vars --

    def test_bound_vars_end(self) -> None:
        assert _bound_vars(End()) == set()

    def test_bound_vars_var(self) -> None:
        assert _bound_vars(Var("X")) == set()

    def test_bound_vars_rec(self) -> None:
        assert _bound_vars(Rec("X", Var("X"))) == {"X"}

    def test_bound_vars_nested_rec(self) -> None:
        node = Rec("X", Rec("Y", Branch((("a", Var("X")), ("b", Var("Y"))))))
        assert _bound_vars(node) == {"X", "Y"}

    def test_bound_vars_parallel(self) -> None:
        node = Parallel(Rec("X", Var("X")), Rec("Y", Var("Y")))
        assert _bound_vars(node) == {"X", "Y"}

    def test_bound_vars_wait(self) -> None:
        """Wait has no bound vars (terminal node like End)."""
        assert _bound_vars(Wait()) == set()

    # -- _contains_parallel --

    def test_contains_parallel_end(self) -> None:
        assert _contains_parallel(End()) is False

    def test_contains_parallel_direct(self) -> None:
        assert _contains_parallel(Parallel(End(), End())) is True

    def test_contains_parallel_nested_in_branch(self) -> None:
        node = Branch((("m", Parallel(End(), End())),))
        assert _contains_parallel(node) is True

    def test_contains_parallel_in_rec(self) -> None:
        node = Rec("X", Parallel(Var("X"), End()))
        assert _contains_parallel(node) is True

    def test_no_parallel(self) -> None:
        node = parse("rec X . &{a: X, done: end}")
        assert _contains_parallel(node) is False

    # -- _has_exit_path edge cases --

    def test_exit_path_end(self) -> None:
        assert _has_exit_path(End(), "X") is True

    def test_exit_path_forbidden_var(self) -> None:
        assert _has_exit_path(Var("X"), "X") is False

    def test_exit_path_other_var(self) -> None:
        assert _has_exit_path(Var("Y"), "X") is True

    def test_exit_path_through_inner_rec(self) -> None:
        """rec Y . &{b: Y, done: end} — checking forbidden=X:
        inner rec, check body; 'done' branch -> End -> True."""
        node = Rec("Y", Branch((("b", Var("Y")), ("done", End()))))
        assert _has_exit_path(node, "X") is True

    def test_exit_path_wait(self) -> None:
        """Wait is a terminal node, so it has an exit path."""
        assert _has_exit_path(Wait(), "X") is True
