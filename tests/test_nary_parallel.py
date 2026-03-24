"""Tests for nary_parallel.py — Binary vs N-ary Parallel Composition (Step 5c)."""

from __future__ import annotations

import pytest

from reticulate.parser import Branch, Continuation, End, Parallel, Rec, Select, Var, Wait, parse
from reticulate.statespace import build_statespace
from reticulate.morphism import find_isomorphism
from reticulate.nary_parallel import (
    NaryParallelResult,
    all_nestings,
    catalan,
    check_nesting_invariance,
    flatten_parallel,
    is_flat_parallel,
    nest_parallel,
    nest_parallel_left,
)


# ---------------------------------------------------------------------------
# Catalan numbers
# ---------------------------------------------------------------------------

class TestCatalan:
    def test_catalan_0(self) -> None:
        assert catalan(0) == 1

    def test_catalan_1(self) -> None:
        assert catalan(1) == 1

    def test_catalan_2(self) -> None:
        assert catalan(2) == 2

    def test_catalan_3(self) -> None:
        assert catalan(3) == 5

    def test_catalan_4(self) -> None:
        assert catalan(4) == 14

    def test_catalan_5(self) -> None:
        assert catalan(5) == 42

    def test_catalan_negative(self) -> None:
        assert catalan(-1) == 0


# ---------------------------------------------------------------------------
# is_flat_parallel
# ---------------------------------------------------------------------------

class TestIsFlatParallel:
    def test_end_is_flat(self) -> None:
        assert is_flat_parallel(End()) is True

    def test_var_is_flat(self) -> None:
        assert is_flat_parallel(Var("X")) is True

    def test_binary_parallel_flat(self) -> None:
        p = Parallel((End(), End()))
        assert is_flat_parallel(p) is True

    def test_nary_flat(self) -> None:
        p = Parallel((End(), End(), End()))
        assert is_flat_parallel(p) is True

    def test_nested_not_flat(self) -> None:
        inner = Parallel((End(), End()))
        outer = Parallel((inner, End()))
        assert is_flat_parallel(outer) is False

    def test_branch_with_parallel_is_flat(self) -> None:
        """Branch containing Parallel — the Branch itself is flat (non-Parallel)."""
        ast = Branch((("a", Parallel((End(), End()))),))
        assert is_flat_parallel(ast) is True


# ---------------------------------------------------------------------------
# flatten_parallel
# ---------------------------------------------------------------------------

class TestFlattenParallel:
    def test_end_unchanged(self) -> None:
        assert flatten_parallel(End()) == End()

    def test_wait_unchanged(self) -> None:
        assert flatten_parallel(Wait()) == Wait()

    def test_var_unchanged(self) -> None:
        assert flatten_parallel(Var("X")) == Var("X")

    def test_binary_unchanged(self) -> None:
        p = Parallel((End(), End()))
        assert flatten_parallel(p) == p

    def test_right_nested_to_flat(self) -> None:
        """(S1 || (S2 || S3)) -> (S1 || S2 || S3)"""
        s1, s2, s3 = Branch((("a", End()),)), Branch((("b", End()),)), Branch((("c", End()),))
        nested = Parallel((s1, Parallel((s2, s3))))
        flat = flatten_parallel(nested)
        assert isinstance(flat, Parallel)
        assert len(flat.branches) == 3
        assert flat.branches == (s1, s2, s3)

    def test_left_nested_to_flat(self) -> None:
        """((S1 || S2) || S3) -> (S1 || S2 || S3)"""
        s1, s2, s3 = Branch((("a", End()),)), Branch((("b", End()),)), Branch((("c", End()),))
        nested = Parallel((Parallel((s1, s2)), s3))
        flat = flatten_parallel(nested)
        assert isinstance(flat, Parallel)
        assert len(flat.branches) == 3
        assert flat.branches == (s1, s2, s3)

    def test_deep_nested(self) -> None:
        """(((S1 || S2) || S3) || S4) -> (S1 || S2 || S3 || S4)"""
        bs = [Branch(((f"l{i}", End()),)) for i in range(4)]
        nested = Parallel((Parallel((Parallel((bs[0], bs[1])), bs[2])), bs[3]))
        flat = flatten_parallel(nested)
        assert isinstance(flat, Parallel)
        assert len(flat.branches) == 4

    def test_flatten_inside_branch(self) -> None:
        """Flatten descends into Branch children."""
        s1, s2, s3 = End(), End(), End()
        inner = Parallel((s1, Parallel((s2, s3))))
        ast = Branch((("a", inner),))
        result = flatten_parallel(ast)
        assert isinstance(result, Branch)
        par = result.choices[0][1]
        assert isinstance(par, Parallel)
        assert len(par.branches) == 3

    def test_flatten_inside_select(self) -> None:
        s1, s2, s3 = End(), End(), End()
        inner = Parallel((Parallel((s1, s2)), s3))
        ast = Select((("a", inner),))
        result = flatten_parallel(ast)
        par = result.choices[0][1]
        assert isinstance(par, Parallel)
        assert len(par.branches) == 3

    def test_flatten_inside_rec(self) -> None:
        s1, s2 = End(), End()
        inner = Parallel((s1, Parallel((s2, End()))))
        ast = Rec("X", inner)
        result = flatten_parallel(ast)
        assert isinstance(result, Rec)
        par = result.body
        assert isinstance(par, Parallel)
        assert len(par.branches) == 3

    def test_flatten_inside_continuation(self) -> None:
        s1, s2, s3 = End(), End(), End()
        inner = Parallel((s1, Parallel((s2, s3))))
        ast = Continuation(inner, End())
        result = flatten_parallel(ast)
        assert isinstance(result, Continuation)
        par = result.left
        assert isinstance(par, Parallel)
        assert len(par.branches) == 3

    def test_single_branch_degenerates(self) -> None:
        """Parallel with one branch after flattening returns the branch itself."""
        p = Parallel((End(),))
        # Flattening a single-branch parallel should just return End
        assert flatten_parallel(p) == End()


# ---------------------------------------------------------------------------
# nest_parallel (right-associative)
# ---------------------------------------------------------------------------

class TestNestParallel:
    def test_binary_unchanged(self) -> None:
        p = Parallel((End(), End()))
        assert nest_parallel(p) == p

    def test_ternary_right_nested(self) -> None:
        s1, s2, s3 = Branch((("a", End()),)), Branch((("b", End()),)), Branch((("c", End()),))
        flat = Parallel((s1, s2, s3))
        result = nest_parallel(flat)
        # Expect: (s1 || (s2 || s3))
        assert isinstance(result, Parallel)
        assert len(result.branches) == 2
        assert result.branches[0] == s1
        inner = result.branches[1]
        assert isinstance(inner, Parallel)
        assert inner.branches == (s2, s3)

    def test_4ary_right_nested(self) -> None:
        bs = [Branch(((f"l{i}", End()),)) for i in range(4)]
        flat = Parallel(tuple(bs))
        result = nest_parallel(flat)
        # (b0 || (b1 || (b2 || b3)))
        assert isinstance(result, Parallel)
        assert result.branches[0] == bs[0]
        mid = result.branches[1]
        assert isinstance(mid, Parallel)
        assert mid.branches[0] == bs[1]
        inner = mid.branches[1]
        assert isinstance(inner, Parallel)
        assert inner.branches == (bs[2], bs[3])


# ---------------------------------------------------------------------------
# nest_parallel_left
# ---------------------------------------------------------------------------

class TestNestParallelLeft:
    def test_binary_unchanged(self) -> None:
        p = Parallel((End(), End()))
        assert nest_parallel_left(p) == p

    def test_ternary_left_nested(self) -> None:
        s1, s2, s3 = Branch((("a", End()),)), Branch((("b", End()),)), Branch((("c", End()),))
        flat = Parallel((s1, s2, s3))
        result = nest_parallel_left(flat)
        # Expect: ((s1 || s2) || s3)
        assert isinstance(result, Parallel)
        assert len(result.branches) == 2
        assert result.branches[1] == s3
        inner = result.branches[0]
        assert isinstance(inner, Parallel)
        assert inner.branches == (s1, s2)

    def test_4ary_left_nested(self) -> None:
        bs = [Branch(((f"l{i}", End()),)) for i in range(4)]
        flat = Parallel(tuple(bs))
        result = nest_parallel_left(flat)
        # ((( b0 || b1 ) || b2 ) || b3)
        assert isinstance(result, Parallel)
        assert result.branches[1] == bs[3]
        mid = result.branches[0]
        assert isinstance(mid, Parallel)
        assert mid.branches[1] == bs[2]
        inner = mid.branches[0]
        assert isinstance(inner, Parallel)
        assert inner.branches == (bs[0], bs[1])


# ---------------------------------------------------------------------------
# Roundtrip: flatten(nest(S)) == S
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_flatten_of_right_nest(self) -> None:
        s1, s2, s3 = Branch((("a", End()),)), Branch((("b", End()),)), Branch((("c", End()),))
        flat = Parallel((s1, s2, s3))
        assert flatten_parallel(nest_parallel(flat)) == flat

    def test_flatten_of_left_nest(self) -> None:
        s1, s2, s3 = Branch((("a", End()),)), Branch((("b", End()),)), Branch((("c", End()),))
        flat = Parallel((s1, s2, s3))
        assert flatten_parallel(nest_parallel_left(flat)) == flat

    def test_roundtrip_4ary(self) -> None:
        bs = tuple(Branch(((f"l{i}", End()),)) for i in range(4))
        flat = Parallel(bs)
        assert flatten_parallel(nest_parallel(flat)) == flat
        assert flatten_parallel(nest_parallel_left(flat)) == flat

    def test_roundtrip_5ary(self) -> None:
        bs = tuple(Branch(((f"l{i}", End()),)) for i in range(5))
        flat = Parallel(bs)
        assert flatten_parallel(nest_parallel(flat)) == flat
        assert flatten_parallel(nest_parallel_left(flat)) == flat


# ---------------------------------------------------------------------------
# all_nestings
# ---------------------------------------------------------------------------

class TestAllNestings:
    def test_2_branches_1_nesting(self) -> None:
        bs = (End(), End())
        nestings = all_nestings(bs)
        assert len(nestings) == catalan(1)  # C(1) = 1

    def test_3_branches_2_nestings(self) -> None:
        bs = (Branch((("a", End()),)), Branch((("b", End()),)), Branch((("c", End()),)))
        nestings = all_nestings(bs)
        assert len(nestings) == catalan(2)  # C(2) = 2

    def test_4_branches_5_nestings(self) -> None:
        bs = tuple(Branch(((f"l{i}", End()),)) for i in range(4))
        nestings = all_nestings(bs)
        assert len(nestings) == catalan(3)  # C(3) = 5

    def test_5_branches_14_nestings(self) -> None:
        bs = tuple(Branch(((f"l{i}", End()),)) for i in range(5))
        nestings = all_nestings(bs)
        assert len(nestings) == catalan(4)  # C(4) = 14

    def test_too_few_branches(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            all_nestings((End(),))

    def test_3_branches_are_left_and_right(self) -> None:
        s1, s2, s3 = Branch((("a", End()),)), Branch((("b", End()),)), Branch((("c", End()),))
        nestings = all_nestings((s1, s2, s3))
        # Two nestings: (s1 || (s2 || s3)) and ((s1 || s2) || s3)
        assert len(nestings) == 2
        # Check they are the same as nest_parallel and nest_parallel_left
        right = nest_parallel(Parallel((s1, s2, s3)))
        left = nest_parallel_left(Parallel((s1, s2, s3)))
        assert right in nestings
        assert left in nestings

    def test_all_nestings_flatten_back(self) -> None:
        """Every nesting flattens back to the original branches."""
        bs = tuple(Branch(((f"l{i}", End()),)) for i in range(4))
        flat = Parallel(bs)
        for nesting in all_nestings(bs):
            assert flatten_parallel(nesting) == flat


# ---------------------------------------------------------------------------
# Isomorphism verification across nestings
# ---------------------------------------------------------------------------

class TestNestingIsomorphism:
    def test_2_branch_isomorphic(self) -> None:
        """Binary parallel: only 1 nesting, trivially isomorphic."""
        ast = parse("(&{a: wait} || &{b: wait})")
        result = check_nesting_invariance(ast)
        assert result.all_nestings_isomorphic is True
        assert result.num_branches == 2
        assert result.catalan_count == 1

    def test_3_branch_isomorphic(self) -> None:
        """3-ary parallel: 2 nestings, both isomorphic."""
        s1 = Branch((("a", Wait()),))
        s2 = Branch((("b", Wait()),))
        s3 = Branch((("c", Wait()),))
        ast = Parallel((s1, s2, s3))
        result = check_nesting_invariance(ast)
        assert result.all_nestings_isomorphic is True
        assert result.num_branches == 3
        assert result.catalan_count == 2
        assert result.nestings_checked == 2

    def test_4_branch_isomorphic(self) -> None:
        """4-ary parallel: 5 nestings, all isomorphic."""
        bs = tuple(Branch(((f"l{i}", Wait()),)) for i in range(4))
        ast = Parallel(bs)
        result = check_nesting_invariance(ast)
        assert result.all_nestings_isomorphic is True
        assert result.num_branches == 4
        assert result.catalan_count == 5
        assert result.nestings_checked == 5

    def test_non_parallel_trivial(self) -> None:
        result = check_nesting_invariance(End())
        assert result.is_flat is True
        assert result.num_branches == 0
        assert result.all_nestings_isomorphic is True

    def test_nested_parallel_flattened_before_check(self) -> None:
        """A nested binary parallel is flattened then checked."""
        s1 = Branch((("a", Wait()),))
        s2 = Branch((("b", Wait()),))
        s3 = Branch((("c", Wait()),))
        ast = Parallel((Parallel((s1, s2)), s3))
        result = check_nesting_invariance(ast)
        assert result.is_flat is False
        assert result.num_branches == 3
        assert result.all_nestings_isomorphic is True

    def test_right_vs_left_nesting_isomorphic(self) -> None:
        """Directly compare right-nested vs left-nested state spaces."""
        s1 = Branch((("a", Wait()),))
        s2 = Branch((("b", Wait()),))
        s3 = Branch((("c", Wait()),))
        flat = Parallel((s1, s2, s3))
        right = nest_parallel(flat)
        left = nest_parallel_left(flat)
        ss_right = build_statespace(right)
        ss_left = build_statespace(left)
        iso = find_isomorphism(ss_right, ss_left)
        assert iso is not None
        assert iso.kind == "isomorphism"


# ---------------------------------------------------------------------------
# Invariant equality across nestings
# ---------------------------------------------------------------------------

class TestInvariantEquality:
    def test_state_count_equal(self) -> None:
        """All nestings have the same number of states."""
        bs = tuple(Branch(((f"l{i}", Wait()),)) for i in range(4))
        nestings = all_nestings(bs)
        sizes = set()
        for nesting in nestings:
            ss = build_statespace(nesting)
            sizes.add(len(ss.states))
        assert len(sizes) == 1

    def test_transition_count_equal(self) -> None:
        """All nestings have the same number of transitions."""
        bs = tuple(Branch(((f"l{i}", Wait()),)) for i in range(3))
        nestings = all_nestings(bs)
        counts = set()
        for nesting in nestings:
            ss = build_statespace(nesting)
            counts.add(len(ss.transitions))
        assert len(counts) == 1


# ---------------------------------------------------------------------------
# Benchmark protocols with parallel
# ---------------------------------------------------------------------------

class TestBenchmarkProtocols:
    def test_two_buyer_parallel(self) -> None:
        """Two-buyer-like: (&{a: wait} || &{b: wait})"""
        ast = parse("(&{a: wait} || &{b: wait})")
        result = check_nesting_invariance(ast)
        assert result.all_nestings_isomorphic is True

    def test_read_write_parallel(self) -> None:
        """(read.wait || write.wait) — parsed from AST directly."""
        s1 = Branch((("read", Wait()),))
        s2 = Branch((("write", Wait()),))
        ast = Parallel((s1, s2))
        result = check_nesting_invariance(ast)
        assert result.num_branches == 2
        assert result.all_nestings_isomorphic is True

    def test_three_way_fork(self) -> None:
        """Three independent activities in parallel."""
        s1 = Branch((("upload", Wait()),))
        s2 = Branch((("validate", Wait()),))
        s3 = Branch((("notify", Wait()),))
        ast = Parallel((s1, s2, s3))
        result = check_nesting_invariance(ast)
        assert result.num_branches == 3
        assert result.all_nestings_isomorphic is True
        assert result.catalan_count == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_too_many_branches_raises(self) -> None:
        bs = tuple(Branch(((f"l{i}", Wait()),)) for i in range(7))
        ast = Parallel(bs)
        with pytest.raises(ValueError, match="too many branches"):
            check_nesting_invariance(ast)

    def test_flat_flag_correct(self) -> None:
        s1 = Branch((("a", Wait()),))
        s2 = Branch((("b", Wait()),))
        flat = Parallel((s1, s2))
        result = check_nesting_invariance(flat)
        assert result.is_flat is True

    def test_mixed_branch_select(self) -> None:
        """Parallel of Branch and Select branches."""
        s1 = Branch((("a", Wait()),))
        s2 = Select((("b", Wait()),))
        s3 = Branch((("c", Wait()),))
        ast = Parallel((s1, s2, s3))
        result = check_nesting_invariance(ast)
        assert result.all_nestings_isomorphic is True
