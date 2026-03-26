"""Tests for the language hierarchy module (Step 400a).

Tests classification, separation witnesses, containment, and benchmark
analysis across language levels L0 through L8.
"""

from __future__ import annotations

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
)
from reticulate.extensions.language_hierarchy import (
    ContainmentResult,
    LanguageLevel,
    LevelProperty,
    check_containment,
    classify_benchmark,
    classify_level,
    hierarchy_summary,
    is_expressible_at,
    level_constructors,
    level_properties,
    level_statistics,
    separation_witness,
    verify_strict_hierarchy,
)


# ---------------------------------------------------------------------------
# Basic classification: each constructor maps to expected level
# ---------------------------------------------------------------------------

class TestClassifyLevel:
    """Test classify_level for each constructor type."""

    def test_end_is_l0(self) -> None:
        assert classify_level(End()) == LanguageLevel.L0

    def test_wait_is_l0(self) -> None:
        assert classify_level(Wait()) == LanguageLevel.L0

    def test_branch_only_is_l1(self) -> None:
        s = parse("&{a: end}")
        assert classify_level(s) == LanguageLevel.L1

    def test_branch_multiple_is_l1(self) -> None:
        s = parse("&{a: end, b: end, c: end}")
        assert classify_level(s) == LanguageLevel.L1

    def test_nested_branch_is_l1(self) -> None:
        s = parse("&{a: &{b: end}}")
        assert classify_level(s) == LanguageLevel.L1

    def test_select_is_l2(self) -> None:
        s = parse("+{a: end}")
        assert classify_level(s) == LanguageLevel.L2

    def test_select_multiple_is_l2(self) -> None:
        s = parse("+{ok: end, err: end}")
        assert classify_level(s) == LanguageLevel.L2

    def test_branch_then_select_is_l2(self) -> None:
        s = parse("&{a: +{ok: end, err: end}}")
        assert classify_level(s) == LanguageLevel.L2

    def test_parallel_is_l3(self) -> None:
        s = parse("(&{a: end} || &{b: end})")
        assert classify_level(s) == LanguageLevel.L3

    def test_parallel_with_select_is_l3(self) -> None:
        s = parse("(+{a: end} || &{b: end})")
        assert classify_level(s) == LanguageLevel.L3

    def test_recursion_is_l4(self) -> None:
        s = parse("rec X . &{a: X, b: end}")
        assert classify_level(s) == LanguageLevel.L4

    def test_recursion_with_select_is_l4(self) -> None:
        s = parse("rec X . &{a: +{ok: X, err: end}}")
        assert classify_level(s) == LanguageLevel.L4

    def test_continuation_is_l5(self) -> None:
        s = parse("(&{a: wait} || &{b: wait}) . &{c: end}")
        assert classify_level(s) == LanguageLevel.L5

    def test_nested_parallel_is_l6(self) -> None:
        s = parse("(&{a: end} || (&{b: end} || &{c: end}))")
        assert classify_level(s) == LanguageLevel.L6

    def test_rec_containing_parallel_is_l7(self) -> None:
        s = parse("rec X . (&{a: end} || &{b: X})")
        assert classify_level(s) == LanguageLevel.L7

    def test_mutual_recursion_is_l8(self) -> None:
        s = parse("rec X . &{a: rec Y . &{b: X, c: Y}, d: end}")
        assert classify_level(s) == LanguageLevel.L8


# ---------------------------------------------------------------------------
# is_expressible_at
# ---------------------------------------------------------------------------

class TestIsExpressibleAt:
    """Test is_expressible_at correctness."""

    def test_end_expressible_at_l0(self) -> None:
        assert is_expressible_at(End(), LanguageLevel.L0)

    def test_end_expressible_at_l4(self) -> None:
        assert is_expressible_at(End(), LanguageLevel.L4)

    def test_branch_not_expressible_at_l0(self) -> None:
        s = parse("&{a: end}")
        assert not is_expressible_at(s, LanguageLevel.L0)

    def test_branch_expressible_at_l1(self) -> None:
        s = parse("&{a: end}")
        assert is_expressible_at(s, LanguageLevel.L1)

    def test_select_not_expressible_at_l1(self) -> None:
        s = parse("+{a: end}")
        assert not is_expressible_at(s, LanguageLevel.L1)

    def test_select_expressible_at_l2(self) -> None:
        s = parse("+{a: end}")
        assert is_expressible_at(s, LanguageLevel.L2)

    def test_parallel_not_expressible_at_l2(self) -> None:
        s = parse("(&{a: end} || &{b: end})")
        assert not is_expressible_at(s, LanguageLevel.L2)

    def test_rec_expressible_at_l4_and_above(self) -> None:
        s = parse("rec X . &{a: X, b: end}")
        assert not is_expressible_at(s, LanguageLevel.L3)
        assert is_expressible_at(s, LanguageLevel.L4)
        assert is_expressible_at(s, LanguageLevel.L8)


# ---------------------------------------------------------------------------
# Separation witnesses
# ---------------------------------------------------------------------------

class TestSeparationWitness:
    """Each consecutive level pair has a proper witness."""

    @pytest.mark.parametrize("lower,upper", [
        (LanguageLevel.L0, LanguageLevel.L1),
        (LanguageLevel.L1, LanguageLevel.L2),
        (LanguageLevel.L2, LanguageLevel.L3),
        (LanguageLevel.L3, LanguageLevel.L4),
        (LanguageLevel.L4, LanguageLevel.L5),
        (LanguageLevel.L5, LanguageLevel.L6),
        (LanguageLevel.L6, LanguageLevel.L7),
        (LanguageLevel.L7, LanguageLevel.L8),
    ])
    def test_witness_in_upper_not_lower(
        self, lower: LanguageLevel, upper: LanguageLevel
    ) -> None:
        wit = separation_witness(lower, upper)
        assert classify_level(wit) == upper, (
            f"Witness should be exactly at {upper.name}, "
            f"got {classify_level(wit).name}"
        )
        assert not is_expressible_at(wit, lower)
        assert is_expressible_at(wit, upper)

    def test_non_consecutive_witness(self) -> None:
        """Witness for (L0, L4) should be in L4 but not L0."""
        wit = separation_witness(LanguageLevel.L0, LanguageLevel.L4)
        assert is_expressible_at(wit, LanguageLevel.L4)
        assert not is_expressible_at(wit, LanguageLevel.L0)

    def test_invalid_lower_geq_upper(self) -> None:
        with pytest.raises(ValueError, match="Cannot construct"):
            separation_witness(LanguageLevel.L3, LanguageLevel.L1)

    def test_equal_levels_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot construct"):
            separation_witness(LanguageLevel.L2, LanguageLevel.L2)

    def test_l9_witness_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            separation_witness(LanguageLevel.L8, LanguageLevel.L9)


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------

class TestContainment:
    """Test containment checking."""

    def test_strict_containment_l0_l1(self) -> None:
        result = check_containment(LanguageLevel.L0, LanguageLevel.L1)
        assert result.is_strict_subset
        assert result.witness is not None

    def test_not_strict_when_equal(self) -> None:
        result = check_containment(LanguageLevel.L3, LanguageLevel.L3)
        assert not result.is_strict_subset
        assert result.witness is None

    def test_not_strict_when_reversed(self) -> None:
        result = check_containment(LanguageLevel.L4, LanguageLevel.L2)
        assert not result.is_strict_subset

    def test_full_hierarchy_strict(self) -> None:
        results = verify_strict_hierarchy()
        assert len(results) >= 8  # L0<L1, L1<L2, ..., L7<L8
        for r in results:
            assert r.is_strict_subset, (
                f"{r.lower.name} < {r.upper.name} should be strict"
            )


# ---------------------------------------------------------------------------
# Level constructors
# ---------------------------------------------------------------------------

class TestLevelConstructors:
    """Test constructor sets per level."""

    def test_l0_has_end_wait(self) -> None:
        ctors = level_constructors(LanguageLevel.L0)
        assert ctors == {"End", "Wait"}

    def test_l1_adds_branch(self) -> None:
        ctors = level_constructors(LanguageLevel.L1)
        assert "Branch" in ctors
        assert "End" in ctors

    def test_l2_adds_select(self) -> None:
        ctors = level_constructors(LanguageLevel.L2)
        assert "Select" in ctors
        assert "Branch" in ctors

    def test_l4_adds_rec_var(self) -> None:
        ctors = level_constructors(LanguageLevel.L4)
        assert "Rec" in ctors
        assert "Var" in ctors

    def test_monotonic_inclusion(self) -> None:
        """Each level's constructors include all of the previous level's."""
        levels = sorted(LanguageLevel)
        for i in range(1, len(levels)):
            prev = level_constructors(levels[i - 1])
            curr = level_constructors(levels[i])
            assert prev <= curr, (
                f"{levels[i].name} should include all of "
                f"{levels[i-1].name}'s constructors"
            )


# ---------------------------------------------------------------------------
# Level properties (lattice-theoretic)
# ---------------------------------------------------------------------------

class TestLevelProperties:
    """Test lattice properties per level."""

    def test_l0_is_lattice(self) -> None:
        prop = level_properties(LanguageLevel.L0)
        assert prop.is_lattice
        assert prop.has_top and prop.has_bottom

    def test_l1_is_meet_semilattice_not_lattice(self) -> None:
        prop = level_properties(LanguageLevel.L1)
        assert prop.is_meet_semilattice
        assert not prop.is_lattice

    def test_l2_is_full_lattice(self) -> None:
        prop = level_properties(LanguageLevel.L2)
        assert prop.is_lattice
        assert prop.is_meet_semilattice
        assert prop.is_join_semilattice

    def test_l3_is_lattice(self) -> None:
        prop = level_properties(LanguageLevel.L3)
        assert prop.is_lattice

    def test_all_levels_have_top_bottom(self) -> None:
        for level in LanguageLevel:
            prop = level_properties(level)
            assert prop.has_top
            assert prop.has_bottom


# ---------------------------------------------------------------------------
# Benchmark classification
# ---------------------------------------------------------------------------

class TestBenchmarkClassification:
    """Test classifying benchmarks."""

    def test_classify_iterator(self) -> None:
        level = classify_benchmark(
            "Java Iterator",
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
        )
        assert level == LanguageLevel.L4

    def test_classify_simple_branch(self) -> None:
        level = classify_benchmark("Simple", "&{a: end, b: end}")
        assert level == LanguageLevel.L1

    def test_classify_parallel_protocol(self) -> None:
        level = classify_benchmark(
            "Two-Buyer",
            "&{lookup: &{getPrice: (&{proposeA: end} || "
            "&{proposeB: +{ACCEPT: &{pay: end}, REJECT: end}})}}",
        )
        assert level == LanguageLevel.L3

    def test_classify_all_benchmarks_runs(self) -> None:
        """All 105 benchmarks should classify without error."""
        from tests.benchmarks.protocols import BENCHMARKS

        for bp in BENCHMARKS:
            level = classify_benchmark(bp.name, bp.type_string)
            assert isinstance(level, LanguageLevel)

    def test_classify_all_benchmarks_dict(self) -> None:
        from reticulate.extensions.language_hierarchy import classify_all_benchmarks
        result = classify_all_benchmarks()
        assert len(result) > 30

    def test_level_statistics_shape(self) -> None:
        stats = level_statistics()
        assert "level_counts" in stats
        assert "total" in stats
        assert stats["total"] > 30


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for classification."""

    def test_var_alone_is_l4(self) -> None:
        """A bare Var needs rec context; classified as L4."""
        assert classify_level(Var("X")) == LanguageLevel.L4

    def test_deeply_nested_branch(self) -> None:
        s = parse("&{a: &{b: &{c: &{d: end}}}}")
        assert classify_level(s) == LanguageLevel.L1

    def test_mixed_branch_select(self) -> None:
        s = parse("&{a: +{ok: end}, b: +{err: end}}")
        assert classify_level(s) == LanguageLevel.L2

    def test_empty_branch_is_l1(self) -> None:
        """Empty branch &{} — still uses Branch constructor."""
        s = Branch(choices=())
        assert classify_level(s) == LanguageLevel.L1

    def test_description_attribute(self) -> None:
        assert "branch" in LanguageLevel.L1.description.lower()

    def test_hierarchy_summary_runs(self) -> None:
        summary = hierarchy_summary()
        assert "Session Type Language Hierarchy" in summary
        assert "STRICT" in summary
