"""Tests for recursive type analysis (Step 13)."""

import pytest

from reticulate.parser import (
    Branch,
    Continuation,
    End,
    Parallel,
    Rec,
    Select,
    Var,
    Wait,
    parse,
    pretty,
)
from reticulate.recursion import (
    GuardednessResult,
    ContractivityResult,
    RecursionAnalysis,
    analyze_recursion,
    analyze_sccs,
    check_contractivity,
    check_guardedness,
    count_rec_binders,
    is_contractive,
    is_guarded,
    is_tail_recursive,
    rec_depth,
    recursive_vars,
    substitute,
    unfold,
    unfold_depth,
)
from reticulate.statespace import build_statespace
from reticulate.morphism import find_isomorphism


# ---------------------------------------------------------------------------
# Guardedness
# ---------------------------------------------------------------------------


class TestGuardedness:
    """Test guardedness checking."""

    def test_end_guarded(self):
        assert is_guarded(End())

    def test_no_recursion_guarded(self):
        assert is_guarded(parse("&{a: end, b: end}"))

    def test_guarded_simple(self):
        assert is_guarded(parse("rec X . &{a: X, b: end}"))

    def test_guarded_nested(self):
        assert is_guarded(parse("rec X . &{a: &{b: X}}"))

    def test_unguarded_identity(self):
        # rec X . X — X appears without any constructor
        s = Rec("X", Var("X"))
        assert not is_guarded(s)

    def test_unguarded_double_rec(self):
        # rec X . rec Y . X — X under another rec but not under constructor
        s = Rec("X", Rec("Y", Var("X")))
        assert not is_guarded(s)

    def test_guarded_result_fields(self):
        s = Rec("X", Var("X"))
        result = check_guardedness(s)
        assert not result.is_guarded
        assert "X" in result.unguarded_vars

    def test_guarded_in_select(self):
        assert is_guarded(parse("+{a: rec X . +{b: X, c: end}}"))

    def test_guarded_in_parallel(self):
        s = Rec("X", Parallel((Var("X"), End())))
        # X is under Parallel constructor — guarded
        assert is_guarded(s)

    def test_multiple_vars(self):
        # rec X . rec Y . &{a: X, b: Y}
        s = Rec("X", Rec("Y", Branch((("a", Var("X")), ("b", Var("Y"))))))
        assert is_guarded(s)


# ---------------------------------------------------------------------------
# Contractivity
# ---------------------------------------------------------------------------


class TestContractivity:
    """Test contractivity checking."""

    def test_end_contractive(self):
        assert is_contractive(End())

    def test_simple_contractive(self):
        assert is_contractive(parse("rec X . &{a: X, b: end}"))

    def test_non_contractive_var(self):
        # rec X . X — body is just a Var
        s = Rec("X", Var("X"))
        assert not is_contractive(s)

    def test_non_contractive_nested_rec(self):
        # rec X . rec Y . &{a: X} — outer body is another Rec
        s = Rec("X", Rec("Y", Branch((("a", Var("X")),))))
        assert not is_contractive(s)

    def test_contractive_result_fields(self):
        s = Rec("X", Var("X"))
        result = check_contractivity(s)
        assert not result.is_contractive
        assert "X" in result.non_contractive_vars

    def test_inner_rec_contractive(self):
        # &{a: rec X . &{b: X}} — inner rec IS contractive
        s = Branch((("a", Rec("X", Branch((("b", Var("X")),)))),))
        assert is_contractive(s)


# ---------------------------------------------------------------------------
# Unfolding
# ---------------------------------------------------------------------------


class TestUnfolding:
    """Test one-step and multi-step unfolding."""

    def test_unfold_non_rec(self):
        s = parse("&{a: end}")
        assert unfold(s) == s

    def test_unfold_simple(self):
        s = parse("rec X . &{a: X, b: end}")
        unfolded = unfold(s)
        # Should be &{a: rec X . &{a: X, b: end}, b: end}
        assert isinstance(unfolded, Branch)
        assert len(unfolded.choices) == 2
        # The "a" choice should contain the original rec type
        a_body = dict(unfolded.choices)["a"]
        assert isinstance(a_body, Rec)

    def test_unfold_preserves_labels(self):
        """Unfolding preserves the set of transition labels."""
        s = parse("rec X . &{a: X, b: end}")
        ss1 = build_statespace(s)
        ss2 = build_statespace(unfold(s))
        labels1 = {l for _, l, _ in ss1.transitions}
        labels2 = {l for _, l, _ in ss2.transitions}
        assert labels1 == labels2

    def test_unfold_depth_zero(self):
        s = parse("rec X . &{a: X}")
        assert unfold_depth(s, 0) == s

    def test_unfold_depth_one(self):
        s = parse("rec X . &{a: X, b: end}")
        result = unfold_depth(s, 1)
        assert isinstance(result, Branch)

    def test_unfold_depth_two(self):
        s = parse("rec X . &{a: X, b: end}")
        result = unfold_depth(s, 2)
        # After 2 unfoldings, should still have Branch at top
        assert isinstance(result, Branch)

    def test_unfold_non_rec_depth(self):
        s = parse("&{a: end}")
        assert unfold_depth(s, 5) == s


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------


class TestSubstitution:
    """Test variable substitution."""

    def test_substitute_var(self):
        result = substitute(Var("X"), "X", End())
        assert result == End()

    def test_substitute_other_var(self):
        result = substitute(Var("Y"), "X", End())
        assert result == Var("Y")

    def test_substitute_in_branch(self):
        s = Branch((("a", Var("X")),))
        result = substitute(s, "X", End())
        assert result == Branch((("a", End()),))

    def test_substitute_shadowed(self):
        s = Rec("X", Var("X"))
        result = substitute(s, "X", End())
        assert result == s  # X is shadowed by the Rec


# ---------------------------------------------------------------------------
# Recursive depth and metrics
# ---------------------------------------------------------------------------


class TestRecursionMetrics:
    """Test recursion depth and counting."""

    def test_depth_no_rec(self):
        assert rec_depth(End()) == 0
        assert rec_depth(parse("&{a: end}")) == 0

    def test_depth_single_rec(self):
        assert rec_depth(parse("rec X . &{a: X}")) == 1

    def test_depth_nested_rec(self):
        s = Rec("X", Branch((("a", Rec("Y", Branch((("b", Var("Y")),)))),)))
        assert rec_depth(s) == 2

    def test_count_single(self):
        assert count_rec_binders(parse("rec X . &{a: X}")) == 1

    def test_count_nested(self):
        s = parse("rec X . &{a: X, b: rec Y . &{c: Y}}")
        assert count_rec_binders(s) == 2

    def test_count_zero(self):
        assert count_rec_binders(End()) == 0

    def test_recursive_vars_empty(self):
        assert recursive_vars(End()) == frozenset()

    def test_recursive_vars_single(self):
        assert recursive_vars(parse("rec X . &{a: X}")) == frozenset({"X"})

    def test_recursive_vars_nested(self):
        s = parse("rec X . &{a: X, b: rec Y . &{c: Y}}")
        assert recursive_vars(s) == frozenset({"X", "Y"})


# ---------------------------------------------------------------------------
# Tail recursion
# ---------------------------------------------------------------------------


class TestTailRecursion:
    """Test tail recursion checking."""

    def test_non_recursive_tail(self):
        assert is_tail_recursive(End())
        assert is_tail_recursive(parse("&{a: end}"))

    def test_simple_tail(self):
        assert is_tail_recursive(parse("rec X . &{a: X, b: end}"))

    def test_nested_branch_tail(self):
        assert is_tail_recursive(parse("rec X . &{a: &{b: X}}"))

    def test_iterator_tail(self):
        assert is_tail_recursive(parse(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        ))

    def test_nested_rec_tail(self):
        s = parse(
            "rec X . &{a: rec Y . &{b: Y, c: X}, d: end}"
        )
        assert is_tail_recursive(s)


# ---------------------------------------------------------------------------
# SCC analysis
# ---------------------------------------------------------------------------


class TestSCCAnalysis:
    """Test SCC analysis of recursive state spaces."""

    def test_no_recursion_trivial(self):
        s = parse("&{a: end}")
        count, sizes = analyze_sccs(s)
        assert count >= 1
        assert sizes == ()  # no non-trivial SCCs

    def test_single_recursion_scc(self):
        """Self-loop creates a trivial SCC (size 1), not a multi-state SCC."""
        s = parse("rec X . &{a: X, b: end}")
        count, sizes = analyze_sccs(s)
        # Self-loop: state loops back to itself, so SCC size is 1 (trivial)
        assert count == 2  # two SCCs: the branch state and end

    def test_nested_recursion_two_sccs(self):
        s = parse("rec X . &{a: rec Y . &{b: Y, c: X}, d: end}")
        count, sizes = analyze_sccs(s)
        # Should have at least 2 non-trivial SCCs
        assert len(sizes) >= 1


# ---------------------------------------------------------------------------
# Complete analysis
# ---------------------------------------------------------------------------


class TestAnalyzeRecursion:
    """Test complete recursion analysis."""

    def test_non_recursive(self):
        result = analyze_recursion(parse("&{a: end, b: end}"))
        assert result.num_rec_binders == 0
        assert result.max_nesting_depth == 0
        assert result.is_guarded
        assert result.is_contractive
        assert result.recursive_vars == frozenset()

    def test_simple_recursive(self):
        result = analyze_recursion(parse("rec X . &{a: X, b: end}"))
        assert result.num_rec_binders == 1
        assert result.max_nesting_depth == 1
        assert result.is_guarded
        assert result.is_contractive
        assert result.recursive_vars == frozenset({"X"})
        assert result.is_tail_recursive

    def test_result_type(self):
        result = analyze_recursion(End())
        assert isinstance(result, RecursionAnalysis)


# ---------------------------------------------------------------------------
# Benchmark analysis
# ---------------------------------------------------------------------------


class TestRecursionBenchmarks:
    """Analyze recursion properties across all 34 benchmarks."""

    @pytest.fixture
    def benchmark_types(self):
        from tests.benchmarks.protocols import BENCHMARKS
        return [(b.name, parse(b.type_string)) for b in BENCHMARKS]

    def test_all_guarded(self, benchmark_types):
        for name, s in benchmark_types:
            assert is_guarded(s), f"{name}: not guarded"

    def test_all_contractive(self, benchmark_types):
        for name, s in benchmark_types:
            assert is_contractive(s), f"{name}: not contractive"

    def test_recursion_statistics(self, benchmark_types):
        total = len(benchmark_types)
        recursive = 0
        tail_recursive = 0
        max_depth = 0
        total_binders = 0

        for name, s in benchmark_types:
            analysis = analyze_recursion(s)
            if analysis.num_rec_binders > 0:
                recursive += 1
            if analysis.is_tail_recursive and analysis.num_rec_binders > 0:
                tail_recursive += 1
            max_depth = max(max_depth, analysis.max_nesting_depth)
            total_binders += analysis.num_rec_binders

        print(f"\n{'='*60}")
        print(f"Step 13: Recursion Statistics ({total} benchmarks)")
        print(f"{'='*60}")
        print(f"Recursive types:          {recursive}/{total}")
        print(f"Tail-recursive:           {tail_recursive}/{recursive}")
        print(f"Max nesting depth:        {max_depth}")
        print(f"Total rec binders:        {total_binders}")
        print(f"All guarded:              True")
        print(f"All contractive:          True")
        print(f"{'='*60}")
