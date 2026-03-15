"""Tests for subtyping (Step 7 + Step 155a: Gay–Hole subtyping with parallel)."""

import pytest

from reticulate.parser import Branch, End, Rec, Select, Var, parse, pretty
from reticulate.subtyping import (
    SubtypingResult,
    WidthSubtypingResult,
    check_subtype,
    check_width_embedding,
    is_subtype,
)


# ---------------------------------------------------------------------------
# Basic subtyping: end
# ---------------------------------------------------------------------------


class TestSubtypeEnd:
    """end ≡ &{} — the empty branch, not a separate constructor.

    Since end is the branch with zero methods:
      - Any branch type is a subtype of end (∅ ⊆ {m₁,...,mₙ})
      - end is NOT a subtype of any non-empty branch ({m,...} ⊄ ∅)
      - end and selection types remain incomparable (different families)
    """

    def test_end_subtype_end(self):
        assert is_subtype(End(), End())

    def test_branch_is_subtype_of_end(self):
        """&{a: end} ≤_GH end — by Sub-Branch: ∅ ⊆ {a}."""
        assert is_subtype(parse("&{a: end}"), End())

    def test_end_not_subtype_of_branch(self):
        """end ≤_GH &{a: end} — FAILS: {a} ⊄ ∅."""
        assert not is_subtype(End(), parse("&{a: end}"))

    def test_wider_branch_subtype_of_end(self):
        """&{a: end, b: end, c: end} ≤_GH end — ∅ ⊆ {a,b,c}."""
        assert is_subtype(parse("&{a: end, b: end, c: end}"), End())

    def test_end_incomparable_with_selection(self):
        """end (≡ &{}) and +{a: end} are incomparable (branch vs select)."""
        assert not is_subtype(End(), parse("+{a: end}"))
        assert not is_subtype(parse("+{a: end}"), End())

    def test_nested_branch_subtype_of_end(self):
        """&{a: &{b: end}} ≤_GH end — ∅ ⊆ {a}."""
        assert is_subtype(parse("&{a: &{b: end}}"), End())

    def test_recursive_subtype_of_end(self):
        """rec X . &{a: X, b: end} ≤_GH end — ∅ ⊆ {a,b}."""
        assert is_subtype(parse("rec X . &{a: X, b: end}"), End())


# ---------------------------------------------------------------------------
# Branch subtyping (width + depth)
# ---------------------------------------------------------------------------


class TestSubtypeBranch:
    """Branch subtyping: more methods = subtype, covariant continuations."""

    def test_same_branch(self):
        s = parse("&{a: end}")
        assert is_subtype(s, s)

    def test_wider_is_subtype(self):
        """&{a: end, b: end} ≤ &{a: end} — more methods is subtype."""
        wide = parse("&{a: end, b: end}")
        narrow = parse("&{a: end}")
        assert is_subtype(wide, narrow)

    def test_narrower_not_subtype(self):
        """&{a: end} ≤ &{a: end, b: end} — FAILS: missing method b."""
        narrow = parse("&{a: end}")
        wide = parse("&{a: end, b: end}")
        assert not is_subtype(narrow, wide)

    def test_covariant_depth_with_end(self):
        """&{a: &{b: end}} ≤ &{a: end} — TRUE because end ≡ &{}.

        The continuation &{b: end} ≤_GH end ≡ &{} holds by Sub-Branch
        (∅ ⊆ {b}).  So the whole relation holds.
        """
        deep = parse("&{a: &{b: end}}")
        shallow = parse("&{a: end}")
        assert is_subtype(deep, shallow)

    def test_same_structure_different_depth(self):
        """&{a: &{b: end, c: end}} ≤ &{a: &{b: end}} — covariant depth."""
        deep = parse("&{a: &{b: end, c: end}}")
        shallow = parse("&{a: &{b: end}}")
        assert is_subtype(deep, shallow)

    def test_width_and_depth(self):
        """&{a: &{b: end, c: end}, d: end} ≤ &{a: &{b: end}} — both wider and deeper."""
        wide_deep = parse("&{a: &{b: end, c: end}, d: end}")
        narrow_shallow = parse("&{a: &{b: end}}")
        assert is_subtype(wide_deep, narrow_shallow)

    def test_three_methods_chain(self):
        a = parse("&{a: end, b: end, c: end}")
        b = parse("&{a: end, b: end}")
        c = parse("&{a: end}")
        assert is_subtype(a, b)
        assert is_subtype(b, c)
        assert is_subtype(a, c)  # transitivity

    def test_disjoint_methods_incomparable(self):
        """&{a: end} and &{b: end} are incomparable."""
        a = parse("&{a: end}")
        b = parse("&{b: end}")
        assert not is_subtype(a, b)
        assert not is_subtype(b, a)


# ---------------------------------------------------------------------------
# Selection subtyping (contravariant width, covariant depth)
# ---------------------------------------------------------------------------


class TestSubtypeSelection:
    """Selection subtyping: fewer labels = subtype, covariant continuations."""

    def test_same_selection(self):
        s = parse("+{a: end}")
        assert is_subtype(s, s)

    def test_fewer_is_subtype(self):
        """+{a: end} ≤ +{a: end, b: end} — fewer selections is subtype."""
        few = parse("+{a: end}")
        many = parse("+{a: end, b: end}")
        assert is_subtype(few, many)

    def test_more_not_subtype(self):
        """+{a: end, b: end} ≤ +{a: end} — FAILS: extra selection b."""
        many = parse("+{a: end, b: end}")
        few = parse("+{a: end}")
        assert not is_subtype(many, few)

    def test_covariant_depth(self):
        """+{a: &{b: end, c: end}} ≤ +{a: &{b: end}} — covariant in continuation."""
        deep = parse("+{a: &{b: end, c: end}}")
        shallow = parse("+{a: &{b: end}}")
        assert is_subtype(deep, shallow)

    def test_selection_chain(self):
        a = parse("+{a: end}")
        ab = parse("+{a: end, b: end}")
        abc = parse("+{a: end, b: end, c: end}")
        assert is_subtype(a, ab)
        assert is_subtype(ab, abc)
        assert is_subtype(a, abc)

    def test_disjoint_selections_incomparable(self):
        a = parse("+{a: end}")
        b = parse("+{b: end}")
        assert not is_subtype(a, b)
        assert not is_subtype(b, a)


# ---------------------------------------------------------------------------
# Mixed constructor subtyping
# ---------------------------------------------------------------------------


class TestSubtypeMixed:
    """Branch and selection are incomparable constructors."""

    def test_branch_not_subtype_of_selection(self):
        assert not is_subtype(parse("&{a: end}"), parse("+{a: end}"))

    def test_selection_not_subtype_of_branch(self):
        assert not is_subtype(parse("+{a: end}"), parse("&{a: end}"))

    def test_nested_mixed(self):
        """&{a: +{x: end}} ≤ &{a: +{x: end, y: end}} — selection inside branch.

        The branch method 'a' has a selection continuation.
        +{x: end} ≤ +{x: end, y: end} (fewer selections = subtype).
        So &{a: +{x: end}} ≤ &{a: +{x: end, y: end}} holds by depth.
        """
        sub = parse("&{a: +{x: end}}")
        sup = parse("&{a: +{x: end, y: end}}")
        assert is_subtype(sub, sup)

    def test_nested_mixed_reverse(self):
        """&{a: +{x: end, y: end}} is NOT ≤ &{a: +{x: end}} — extra selection."""
        many = parse("&{a: +{x: end, y: end}}")
        few = parse("&{a: +{x: end}}")
        assert not is_subtype(many, few)

    def test_branch_with_wider_and_selection_with_fewer(self):
        """&{a: +{x: end}, b: +{x: end}} ≤ &{a: +{x: end, y: end}}."""
        sub = parse("&{a: +{x: end}, b: +{x: end}}")
        sup = parse("&{a: +{x: end, y: end}}")
        assert is_subtype(sub, sup)


# ---------------------------------------------------------------------------
# Recursive subtyping
# ---------------------------------------------------------------------------


class TestSubtypeRecursion:
    """Recursive types use coinductive subtyping."""

    def test_same_recursive(self):
        s = parse("rec X . &{a: X, b: end}")
        assert is_subtype(s, s)

    def test_recursive_wider_branch(self):
        """rec X . &{a: X, b: end, c: end} ≤ rec X . &{a: X, b: end}."""
        wide = parse("rec X . &{a: X, b: end, c: end}")
        narrow = parse("rec X . &{a: X, b: end}")
        assert is_subtype(wide, narrow)

    def test_recursive_narrower_not_subtype(self):
        narrow = parse("rec X . &{a: X}")
        wide = parse("rec X . &{a: X, b: end}")
        assert not is_subtype(narrow, wide)

    def test_recursive_same_structure(self):
        """Two structurally identical recursive types are subtypes."""
        s1 = parse("rec X . &{a: X, b: end}")
        s2 = parse("rec Y . &{a: Y, b: end}")
        assert is_subtype(s1, s2)
        assert is_subtype(s2, s1)

    def test_recursive_incomparable(self):
        """rec X . &{a: X} and rec X . &{b: X} are incomparable."""
        s1 = parse("rec X . &{a: X}")
        s2 = parse("rec X . &{b: X}")
        assert not is_subtype(s1, s2)
        assert not is_subtype(s2, s1)


# ---------------------------------------------------------------------------
# check_subtype result object
# ---------------------------------------------------------------------------


class TestCheckSubtype:
    """Test the check_subtype function returns proper results."""

    def test_subtype_result_true(self):
        result = check_subtype(parse("&{a: end, b: end}"), parse("&{a: end}"))
        assert isinstance(result, SubtypingResult)
        assert result.is_subtype
        assert result.reason is None

    def test_subtype_result_false_missing_methods(self):
        result = check_subtype(parse("&{a: end}"), parse("&{a: end, b: end}"))
        assert not result.is_subtype
        assert "missing" in result.reason

    def test_subtype_result_false_extra_selection(self):
        result = check_subtype(parse("+{a: end, b: end}"), parse("+{a: end}"))
        assert not result.is_subtype
        assert "select" in result.reason.lower() or "label" in result.reason.lower()

    def test_subtype_result_false_constructors(self):
        result = check_subtype(parse("&{a: end}"), parse("+{a: end}"))
        assert not result.is_subtype
        assert "incompatible" in result.reason.lower()

    def test_subtype_result_lhs_rhs(self):
        s1 = parse("&{a: end}")
        s2 = parse("&{b: end}")
        result = check_subtype(s1, s2)
        assert result.lhs == pretty(s1)
        assert result.rhs == pretty(s2)


# ---------------------------------------------------------------------------
# Reflexivity and transitivity
# ---------------------------------------------------------------------------


class TestSubtypeProperties:
    """Subtyping should be reflexive and transitive (a preorder)."""

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "&{a: end, b: end}",
        "+{a: end}",
        "+{a: end, b: end}",
        "&{a: +{x: end}}",
        "rec X . &{a: X, b: end}",
        "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
    ])
    def test_reflexivity(self, type_str: str):
        s = parse(type_str)
        assert is_subtype(s, s)

    def test_transitivity_branch(self):
        a = parse("&{a: end, b: end, c: end}")
        b = parse("&{a: end, b: end}")
        c = parse("&{a: end}")
        assert is_subtype(a, b) and is_subtype(b, c) and is_subtype(a, c)

    def test_transitivity_branch_to_end(self):
        """Full chain: &{a,b,c} ≤ &{a,b} ≤ &{a} ≤ end."""
        a = parse("&{a: end, b: end, c: end}")
        b = parse("&{a: end, b: end}")
        c = parse("&{a: end}")
        d = End()
        assert is_subtype(a, b) and is_subtype(b, c)
        assert is_subtype(c, d) and is_subtype(a, d)

    def test_transitivity_selection(self):
        a = parse("+{a: end}")
        b = parse("+{a: end, b: end}")
        c = parse("+{a: end, b: end, c: end}")
        assert is_subtype(a, b) and is_subtype(b, c) and is_subtype(a, c)

    def test_antisymmetry_up_to_equivalence(self):
        """S₁ ≤ S₂ and S₂ ≤ S₁ implies structural equivalence."""
        s1 = parse("rec X . &{a: X, b: end}")
        s2 = parse("rec Y . &{a: Y, b: end}")
        assert is_subtype(s1, s2) and is_subtype(s2, s1)


# ---------------------------------------------------------------------------
# Width subtyping ↔ embedding correspondence
# ---------------------------------------------------------------------------


class TestWidthEmbedding:
    """Width subtyping should correspond to state-space embedding."""

    def test_branch_width_embedding(self):
        """&{a: end, b: end} ≤ &{a: end} ↔ L(&{a: end}) embeds in L(&{a:end, b:end})."""
        result = check_width_embedding(
            parse("&{a: end, b: end}"), parse("&{a: end}")
        )
        assert result.is_subtype
        assert result.has_embedding
        assert result.coincides

    def test_branch_not_subtype_no_embedding(self):
        """&{a: end} NOT ≤ &{a: end, b: end} and no embedding."""
        result = check_width_embedding(
            parse("&{a: end}"), parse("&{a: end, b: end}")
        )
        # Not a subtype
        assert not result.is_subtype

    def test_same_type_embedding(self):
        result = check_width_embedding(
            parse("&{a: end}"), parse("&{a: end}")
        )
        assert result.is_subtype
        assert result.has_embedding
        assert result.coincides

    def test_recursive_width_embedding(self):
        """rec X . &{a: X, b: end, c: end} ≤ rec X . &{a: X, b: end}."""
        result = check_width_embedding(
            parse("rec X . &{a: X, b: end, c: end}"),
            parse("rec X . &{a: X, b: end}"),
        )
        assert result.is_subtype
        assert result.has_embedding
        assert result.coincides

    def test_selection_width_embedding(self):
        """+{a: end} ≤ +{a: end, b: end} ↔ L(+{a:end, b:end}) embeds in L(+{a:end})."""
        result = check_width_embedding(
            parse("+{a: end}"), parse("+{a: end, b: end}")
        )
        assert result.is_subtype
        assert result.has_embedding
        assert result.coincides

    def test_three_methods_width(self):
        result = check_width_embedding(
            parse("&{a: end, b: end, c: end}"), parse("&{a: end}")
        )
        assert result.is_subtype
        assert result.has_embedding
        assert result.coincides


# ---------------------------------------------------------------------------
# Result type tests
# ---------------------------------------------------------------------------


class TestResultTypes:
    """Test result dataclass properties."""

    def test_subtyping_result_fields(self):
        result = SubtypingResult(lhs="end", rhs="end", is_subtype=True)
        assert result.lhs == "end"
        assert result.rhs == "end"
        assert result.is_subtype
        assert result.reason is None

    def test_width_subtyping_result_fields(self):
        result = WidthSubtypingResult(
            type1="&{a: end}", type2="&{a: end}",
            is_subtype=True, has_embedding=True, coincides=True,
        )
        assert result.coincides


# ---------------------------------------------------------------------------
# Step 155a: Parallel subtyping (componentwise)
# ---------------------------------------------------------------------------


class TestParallelSubtyping:
    """Step 155a: (S₁ || S₂) ≤ (T₁ || T₂) iff S₁ ≤ T₁ and S₂ ≤ T₂.

    The parallel constructor is covariant in both components.  This yields
    a product of embeddings: if S₁ embeds in T₁ and S₂ embeds in T₂,
    then L(S₁)×L(S₂) embeds in L(T₁)×L(T₂).
    """

    def test_identical_parallel(self):
        """(A || B) ≤ (A || B)."""
        s = parse("(&{a: end} || &{b: end})")
        assert is_subtype(s, s)

    def test_wider_left_branch(self):
        """(&{a: end, c: end} || &{b: end}) ≤ (&{a: end} || &{b: end})."""
        s1 = parse("(&{a: end, c: end} || &{b: end})")
        s2 = parse("(&{a: end} || &{b: end})")
        assert is_subtype(s1, s2)

    def test_wider_right_branch(self):
        """(&{a: end} || &{b: end, c: end}) ≤ (&{a: end} || &{b: end})."""
        s1 = parse("(&{a: end} || &{b: end, c: end})")
        s2 = parse("(&{a: end} || &{b: end})")
        assert is_subtype(s1, s2)

    def test_wider_both_branches(self):
        """(&{a: end, c: end} || &{b: end, d: end}) ≤ (&{a: end} || &{b: end})."""
        s1 = parse("(&{a: end, c: end} || &{b: end, d: end})")
        s2 = parse("(&{a: end} || &{b: end})")
        assert is_subtype(s1, s2)

    def test_narrower_left_fails(self):
        """(&{a: end} || &{b: end}) ≤ (&{a: end, c: end} || &{b: end}) — FAILS."""
        s1 = parse("(&{a: end} || &{b: end})")
        s2 = parse("(&{a: end, c: end} || &{b: end})")
        assert not is_subtype(s1, s2)

    def test_narrower_right_fails(self):
        """(&{a: end} || &{b: end}) ≤ (&{a: end} || &{b: end, c: end}) — FAILS."""
        s1 = parse("(&{a: end} || &{b: end})")
        s2 = parse("(&{a: end} || &{b: end, c: end})")
        assert not is_subtype(s1, s2)

    def test_parallel_vs_branch_incomparable(self):
        """(A || B) and &{a: end} are incomparable constructors."""
        s1 = parse("(&{a: end} || &{b: end})")
        s2 = parse("&{a: end}")
        assert not is_subtype(s1, s2)
        assert not is_subtype(s2, s1)

    def test_parallel_vs_select_incomparable(self):
        """(A || B) and +{a: end} are incomparable constructors."""
        s1 = parse("(&{a: end} || &{b: end})")
        s2 = parse("+{a: end}")
        assert not is_subtype(s1, s2)
        assert not is_subtype(s2, s1)

    def test_parallel_selection_componentwise(self):
        """(+{A: end} || +{B: end}) ≤ (+{A: end, C: end} || +{B: end, D: end})."""
        s1 = parse("(+{A: end} || +{B: end})")
        s2 = parse("(+{A: end, C: end} || +{B: end, D: end})")
        assert is_subtype(s1, s2)

    def test_parallel_mixed_constructors(self):
        """(&{a: end, b: end} || +{X: end}) ≤ (&{a: end} || +{X: end, Y: end})."""
        s1 = parse("(&{a: end, b: end} || +{X: end})")
        s2 = parse("(&{a: end} || +{X: end, Y: end})")
        assert is_subtype(s1, s2)

    def test_parallel_mixed_one_component_fails(self):
        """(&{a: end} || +{X: end, Y: end}) ≤ (&{a: end, b: end} || +{X: end}) — FAILS.

        Left component: &{a:end} not ≤ &{a:end, b:end} (missing b).
        """
        s1 = parse("(&{a: end} || +{X: end, Y: end})")
        s2 = parse("(&{a: end, b: end} || +{X: end})")
        assert not is_subtype(s1, s2)

    def test_parallel_with_continuation_subtype(self):
        """(A || B) . C ≤ (A || B) . C — reflexivity."""
        s = parse("(&{a: end} || &{b: end}) . &{c: end}")
        assert is_subtype(s, s)

    def test_parallel_with_wider_continuation(self):
        """(A || B) . &{c: end, d: end} ≤ (A || B) . &{c: end}."""
        s1 = parse("(&{a: end} || &{b: end}) . &{c: end, d: end}")
        s2 = parse("(&{a: end} || &{b: end}) . &{c: end}")
        assert is_subtype(s1, s2)

    def test_parallel_with_narrower_continuation_fails(self):
        """(A || B) . &{c: end} ≤ (A || B) . &{c: end, d: end} — FAILS."""
        s1 = parse("(&{a: end} || &{b: end}) . &{c: end}")
        s2 = parse("(&{a: end} || &{b: end}) . &{c: end, d: end}")
        assert not is_subtype(s1, s2)

    def test_parallel_embedding_coincides(self):
        """Width embedding correspondence for parallel types."""
        s1 = parse("(&{a: end, c: end} || &{b: end})")
        s2 = parse("(&{a: end} || &{b: end})")
        result = check_width_embedding(s1, s2)
        assert result.is_subtype
        assert result.has_embedding
        assert result.coincides

    def test_parallel_non_subtype_may_have_embedding(self):
        """Subtyping ⇒ embedding, but not converse (functor faithful, not full).

        (&{a:end} || &{b:end}) has 4 states; (&{a:end,c:end} || &{b:end})
        has 6 states.  The smaller embeds into the larger, but the subtyping
        direction is reversed (narrower branch is NOT a subtype of wider).
        """
        s1 = parse("(&{a: end} || &{b: end})")
        s2 = parse("(&{a: end, c: end} || &{b: end})")
        result = check_width_embedding(s1, s2)
        assert not result.is_subtype
        # Embedding may still exist (faithful but not full)
        assert result.has_embedding is True

    def test_recursive_parallel_subtype(self):
        """Parallel with recursive components: wider rec branch ≤ narrower."""
        s1 = parse("(rec X . &{a: X, b: end} || &{c: end})")
        s2 = parse("(rec X . &{a: X} || &{c: end})")
        assert is_subtype(s1, s2)

    def test_recursive_parallel_not_subtype(self):
        """Parallel with recursive components: narrower rec branch ≤ wider — FAILS."""
        s1 = parse("(rec X . &{a: X} || &{c: end})")
        s2 = parse("(rec X . &{a: X, b: end} || &{c: end})")
        assert not is_subtype(s1, s2)

    def test_nested_parallel_subtype(self):
        """Nested parallel: wider inner components."""
        s1 = parse("(&{a: end, x: end} || &{b: end})")
        s2 = parse("(&{a: end} || &{b: end})")
        # Wrap in continuation
        from reticulate.parser import Continuation
        outer1 = Continuation(s1, parse("&{c: end, d: end}"))
        outer2 = Continuation(s2, parse("&{c: end}"))
        assert is_subtype(outer1, outer2)

    def test_check_subtype_parallel_result(self):
        """check_subtype returns proper SubtypingResult for parallel."""
        s1 = parse("(&{a: end, c: end} || &{b: end})")
        s2 = parse("(&{a: end} || &{b: end})")
        result = check_subtype(s1, s2)
        assert result.is_subtype
        assert result.reason is None

    def test_check_subtype_parallel_failure_reason(self):
        """check_subtype explains parallel failure."""
        s1 = parse("(&{a: end} || &{b: end})")
        s2 = parse("(&{a: end, c: end} || &{b: end})")
        result = check_subtype(s1, s2)
        assert not result.is_subtype
        assert result.reason is not None

    # -- N-ary parallel subtyping (Step 155a extension) --

    def test_ternary_parallel_subtype(self):
        """3-way parallel: componentwise wider ≤ narrower."""
        s1 = parse("(&{a: end, x: end} || &{b: end, y: end} || &{c: end, z: end})")
        s2 = parse("(&{a: end} || &{b: end} || &{c: end})")
        assert is_subtype(s1, s2)

    def test_ternary_parallel_not_subtype(self):
        """3-way parallel: one narrower component breaks subtyping."""
        s1 = parse("(&{a: end} || &{b: end} || &{c: end})")
        s2 = parse("(&{a: end, x: end} || &{b: end} || &{c: end})")
        assert not is_subtype(s1, s2)

    def test_different_arity_incomparable(self):
        """2-way vs 3-way parallel: different arity → not subtypes."""
        s2 = parse("(&{a: end} || &{b: end})")
        s3 = parse("(&{a: end} || &{b: end} || &{c: end})")
        assert not is_subtype(s2, s3)
        assert not is_subtype(s3, s2)

    def test_ternary_parallel_reflexive(self):
        """3-way parallel: reflexive subtyping."""
        s = parse("(&{a: end} || &{b: end} || &{c: end})")
        assert is_subtype(s, s)
