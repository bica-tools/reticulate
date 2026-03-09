"""Tests for subtyping (Step 7: Gay–Hole subtyping on session type ASTs)."""

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
    """end ≤ end is the only relation involving end."""

    def test_end_subtype_end(self):
        assert is_subtype(End(), End())

    def test_end_incomparable_with_branch(self):
        """end and &{a: end} are incomparable (different constructors)."""
        assert not is_subtype(End(), parse("&{a: end}"))
        assert not is_subtype(parse("&{a: end}"), End())

    def test_end_incomparable_with_selection(self):
        assert not is_subtype(End(), parse("+{a: end}"))
        assert not is_subtype(parse("+{a: end}"), End())


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

    def test_covariant_depth(self):
        """&{a: &{b: end}} ≤ &{a: end} — deeper continuation is NOT subtype.

        The continuation &{b: end} is incomparable with end (different
        constructors), so the depth subtyping check fails.
        """
        deep = parse("&{a: &{b: end}}")
        shallow = parse("&{a: end}")
        assert not is_subtype(deep, shallow)

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
