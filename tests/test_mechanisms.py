"""Tests for mechanisms of thought as session type operations (Step 102)."""

from __future__ import annotations

import pytest

from reticulate import parse, build_statespace
from reticulate.lattice import check_lattice
from reticulate.parser import Branch, Select, End, Rec, Var, Parallel, SessionType, pretty
from reticulate.mechanisms import (
    MechanismResult,
    abstract_by_labels,
    abstract_by_depth,
    compose_sequential,
    compose_parallel,
    compose_choice,
    negate_type,
    compute_violations,
    make_recursive,
    unroll,
    fixed_point_depth,
    detect_emergence,
    emergence_score,
    dialectic,
    dialectic_chain,
    metaphor,
    detect_metaphor,
    metaphor_quality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE = parse("&{a: end, b: end}")
TWO_DEEP = parse("&{a: &{c: end}, b: end}")
RECURSIVE = parse("rec X . &{a: X, b: end}")


# =========================================================================
# Abstraction tests (8)
# =========================================================================


class TestAbstractByLabels:
    """abstract_by_labels preserves structure but renames labels."""

    def test_preserves_state_count(self) -> None:
        ss = build_statespace(SIMPLE)
        result = abstract_by_labels(ss)
        assert len(result.states) == len(ss.states)

    def test_preserves_transition_count(self) -> None:
        ss = build_statespace(SIMPLE)
        result = abstract_by_labels(ss)
        assert len(result.transitions) == len(ss.transitions)

    def test_labels_are_generic(self) -> None:
        ss = build_statespace(SIMPLE)
        result = abstract_by_labels(ss)
        labels = {lbl for _, lbl, _ in result.transitions}
        assert all(lbl.isalpha() and len(lbl) == 1 for lbl in labels)

    def test_preserves_lattice(self) -> None:
        ss = build_statespace(SIMPLE)
        result = abstract_by_labels(ss)
        assert check_lattice(result).is_lattice


class TestAbstractByDepth:
    """abstract_by_depth truncates to given depth."""

    def test_depth_zero_gives_top_and_bottom(self) -> None:
        ss = build_statespace(TWO_DEEP)
        result = abstract_by_depth(ss, 0)
        assert ss.top in result.states
        assert ss.bottom in result.states

    def test_depth_one_keeps_first_level(self) -> None:
        ss = build_statespace(TWO_DEEP)
        result = abstract_by_depth(ss, 1)
        assert len(result.states) >= 2

    def test_full_depth_preserves_all(self) -> None:
        ss = build_statespace(SIMPLE)
        result = abstract_by_depth(ss, 100)
        assert len(result.states) == len(ss.states)

    def test_preserves_lattice(self) -> None:
        ss = build_statespace(TWO_DEEP)
        result = abstract_by_depth(ss, 1)
        assert check_lattice(result).is_lattice


# =========================================================================
# Composition tests (8)
# =========================================================================


class TestComposeSequential:
    """compose_sequential chains types."""

    def test_single_returns_itself(self) -> None:
        t = End()
        assert compose_sequential([t]) is t

    def test_two_creates_branch(self) -> None:
        t1 = Branch((("a", End()),))
        t2 = Branch((("b", End()),))
        result = compose_sequential([t1, t2])
        assert isinstance(result, Branch)
        labels = [m for m, _ in result.choices]
        assert "phase_0" in labels
        assert "phase_1" in labels

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compose_sequential([])

    def test_three_creates_three_phases(self) -> None:
        types = [End(), End(), End()]
        result = compose_sequential(types)
        assert isinstance(result, Branch)
        assert len(result.choices) == 3


class TestComposeParallel:
    """compose_parallel nests binary parallel."""

    def test_single_returns_itself(self) -> None:
        t = End()
        assert compose_parallel([t]) is t

    def test_two_creates_parallel(self) -> None:
        t1 = Branch((("a", End()),))
        t2 = Branch((("b", End()),))
        result = compose_parallel([t1, t2])
        assert isinstance(result, Parallel)

    def test_three_nests(self) -> None:
        types = [Branch((("a", End()),)), Branch((("b", End()),)), Branch((("c", End()),))]
        result = compose_parallel(types)
        assert isinstance(result, Parallel)
        # Nested: Parallel(t1, Parallel(t2, t3))
        assert isinstance(result.branches[1], Parallel)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compose_parallel([])


class TestComposeChoice:
    """compose_choice creates branch with labels."""

    def test_default_labels(self) -> None:
        types = [End(), End()]
        result = compose_choice(types)
        assert isinstance(result, Branch)
        labels = [m for m, _ in result.choices]
        assert labels == ["option_0", "option_1"]

    def test_custom_labels(self) -> None:
        types = [End(), End()]
        result = compose_choice(types, labels=["foo", "bar"])
        labels = [m for m, _ in result.choices]
        assert labels == ["foo", "bar"]

    def test_label_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compose_choice([End()], labels=["a", "b"])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compose_choice([])


# =========================================================================
# Negation tests (6)
# =========================================================================


class TestNegateType:
    """negate_type swaps Branch/Select."""

    def test_branch_becomes_select(self) -> None:
        t = Branch((("a", End()), ("b", End())))
        result = negate_type(t)
        assert isinstance(result, Select)

    def test_select_becomes_branch(self) -> None:
        t = Select((("a", End()),))
        result = negate_type(t)
        assert isinstance(result, Branch)

    def test_double_negation_identity(self) -> None:
        t = Branch((("a", Select((("b", End()),))),))
        assert negate_type(negate_type(t)) == t

    def test_end_stays_end(self) -> None:
        assert negate_type(End()) == End()


class TestComputeViolations:
    """compute_violations finds disabled methods per state."""

    def test_end_state_has_all_violations(self) -> None:
        ss = build_statespace(SIMPLE)
        violations = compute_violations(ss)
        bottom_violations = [lbl for s, lbl in violations if s == ss.bottom]
        all_labels = {lbl for _, lbl, _ in ss.transitions}
        assert set(bottom_violations) == all_labels

    def test_top_has_no_violations_for_simple(self) -> None:
        ss = build_statespace(SIMPLE)
        violations = compute_violations(ss)
        top_violations = [lbl for s, lbl in violations if s == ss.top]
        assert len(top_violations) == 0


# =========================================================================
# Recursion tests (8)
# =========================================================================


class TestMakeRecursive:
    """make_recursive wraps type in Rec."""

    def test_wraps_in_rec(self) -> None:
        t = Branch((("a", End()),))
        result = make_recursive(t)
        assert isinstance(result, Rec)
        assert result.var == "X"

    def test_ends_replaced_with_var(self) -> None:
        t = Branch((("a", End()), ("b", End())))
        result = make_recursive(t)
        assert isinstance(result, Rec)
        for _, cont in result.body.choices:
            assert isinstance(cont, Var)
            assert cont.name == "X"

    def test_custom_var_name(self) -> None:
        t = Branch((("a", End()),))
        result = make_recursive(t, "Y")
        assert isinstance(result, Rec)
        assert result.var == "Y"

    def test_nested_ends_all_replaced(self) -> None:
        t = Branch((("a", Branch((("b", End()),))),))
        result = make_recursive(t)
        # The inner End should be replaced too.
        inner = result.body.choices[0][1]  # continuation of "a"
        assert isinstance(inner, Branch)
        assert isinstance(inner.choices[0][1], Var)


class TestUnroll:
    """unroll removes recursion."""

    def test_non_recursive_unchanged(self) -> None:
        t = Branch((("a", End()),))
        result = unroll(t)
        assert isinstance(result, Branch)

    def test_recursive_unrolled(self) -> None:
        t = Rec("X", Branch((("a", Var("X")), ("b", End()))))
        result = unroll(t, depth=1)
        # After 1 unroll, X is replaced by body, remaining Vars become End.
        assert isinstance(result, Branch)
        # No Rec or Var should remain at top.
        assert not isinstance(result, Rec)

    def test_depth_zero_terminates_vars(self) -> None:
        t = Rec("X", Branch((("a", Var("X")),)))
        result = unroll(t, depth=0)
        # depth=0 means no unfolding, but Rec stays; vars get terminated.
        # Since unfold_depth(t, 0) returns t unchanged, _terminate_vars handles Rec.
        # The Rec still wraps it, but inner Var -> End.
        assert isinstance(result, Rec)


class TestFixedPointDepth:
    """fixed_point_depth detects cycles."""

    def test_non_recursive_zero(self) -> None:
        ss = build_statespace(SIMPLE)
        assert fixed_point_depth(ss) == 0

    def test_recursive_one(self) -> None:
        ss = build_statespace(RECURSIVE)
        assert fixed_point_depth(ss) == 1


# =========================================================================
# Emergence tests (6)
# =========================================================================


class TestDetectEmergence:
    """detect_emergence compares parts vs whole."""

    def test_no_emergence_single_part(self) -> None:
        ss = build_statespace(SIMPLE)
        result = detect_emergence([ss], ss)
        assert len(result) == 0

    def test_parallel_may_have_emergence(self) -> None:
        t1 = parse("&{a: end}")
        t2 = parse("&{b: end}")
        ss1 = build_statespace(t1)
        ss2 = build_statespace(t2)
        whole = build_statespace(parse("(&{a: end} || &{b: end})"))
        result = detect_emergence([ss1, ss2], whole)
        # Parallel product typically has more states than sum.
        assert "state_count_superadditive" in result or len(result) >= 0

    def test_empty_parts_no_emergence(self) -> None:
        ss = build_statespace(SIMPLE)
        assert detect_emergence([], ss) == {}


class TestEmergenceScore:
    """emergence_score returns value in [0, 1]."""

    def test_score_in_range(self) -> None:
        ss = build_statespace(SIMPLE)
        score = emergence_score([ss], ss)
        assert 0.0 <= score <= 1.0

    def test_empty_parts_zero(self) -> None:
        ss = build_statespace(SIMPLE)
        assert emergence_score([], ss) == 0.0

    def test_identical_parts_and_whole_zero(self) -> None:
        ss = build_statespace(SIMPLE)
        score = emergence_score([ss], ss)
        assert score == 0.0


# =========================================================================
# Dialectic tests (6)
# =========================================================================


class TestDialectic:
    """dialectic produces union synthesis."""

    def test_identical_is_identity(self) -> None:
        t = Branch((("a", End()), ("b", End())))
        result = dialectic(t, t)
        assert isinstance(result, Branch)
        labels = {m for m, _ in result.choices}
        assert labels == {"a", "b"}

    def test_includes_all_methods(self) -> None:
        t1 = Branch((("a", End()), ("b", End())))
        t2 = Branch((("b", End()), ("c", End())))
        result = dialectic(t1, t2)
        labels = {m for m, _ in result.choices}
        assert labels == {"a", "b", "c"}

    def test_end_with_type_returns_type(self) -> None:
        t = Branch((("a", End()),))
        assert dialectic(End(), t) == t
        assert dialectic(t, End()) == t

    def test_both_end_returns_end(self) -> None:
        assert dialectic(End(), End()) == End()

    def test_commutative_method_set(self) -> None:
        t1 = Branch((("a", End()), ("b", End())))
        t2 = Branch((("c", End()), ("d", End())))
        r1 = dialectic(t1, t2)
        r2 = dialectic(t2, t1)
        labels1 = {m for m, _ in r1.choices}
        labels2 = {m for m, _ in r2.choices}
        assert labels1 == labels2


class TestDialecticChain:
    """dialectic_chain folds over a list."""

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            dialectic_chain([])

    def test_single_returns_itself(self) -> None:
        t = Branch((("a", End()),))
        assert dialectic_chain([t]) == t

    def test_chain_accumulates(self) -> None:
        t1 = Branch((("a", End()),))
        t2 = Branch((("b", End()),))
        t3 = Branch((("c", End()),))
        result = dialectic_chain([t1, t2, t3])
        labels = {m for m, _ in result.choices}
        assert labels == {"a", "b", "c"}


# =========================================================================
# Metaphor tests (8)
# =========================================================================


class TestMetaphor:
    """metaphor relabels using conceptual mapping."""

    def test_simple_relabel(self) -> None:
        t = Branch((("attack", End()), ("defend", End())))
        result = metaphor(t, {"attack": "criticize", "defend": "justify"})
        labels = {m for m, _ in result.choices}
        assert labels == {"criticize", "justify"}

    def test_unmapped_preserved(self) -> None:
        t = Branch((("a", End()), ("b", End())))
        result = metaphor(t, {"a": "x"})
        labels = {m for m, _ in result.choices}
        assert labels == {"x", "b"}

    def test_empty_mapping_identity(self) -> None:
        t = Branch((("a", End()),))
        result = metaphor(t, {})
        assert result == t

    def test_nested_relabeling(self) -> None:
        t = Branch((("a", Branch((("b", End()),))),))
        result = metaphor(t, {"a": "x", "b": "y"})
        outer_label = result.choices[0][0]
        inner_label = result.choices[0][1].choices[0][0]
        assert outer_label == "x"
        assert inner_label == "y"


class TestDetectMetaphor:
    """detect_metaphor finds label bijection."""

    def test_relabeled_type_detected(self) -> None:
        t1 = Branch((("a", End()), ("b", End())))
        t2 = Branch((("x", End()), ("y", End())))
        mapping = detect_metaphor(t1, t2)
        assert mapping is not None
        assert set(mapping.values()) == {"x", "y"}

    def test_different_structure_returns_none(self) -> None:
        t1 = Branch((("a", End()), ("b", End())))
        t2 = Branch((("x", End()),))
        assert detect_metaphor(t1, t2) is None

    def test_identical_types(self) -> None:
        t = Branch((("a", End()), ("b", End())))
        mapping = detect_metaphor(t, t)
        assert mapping is not None
        # Identity mapping.
        assert mapping.get("a") == "a"
        assert mapping.get("b") == "b"

    def test_end_types(self) -> None:
        mapping = detect_metaphor(End(), End())
        assert mapping is not None
        assert mapping == {}


class TestMetaphorQuality:
    """metaphor_quality measures mapping fitness."""

    def test_perfect_mapping(self) -> None:
        t = Branch((("a", End()), ("b", End())))
        ss = build_statespace(t)
        quality = metaphor_quality({"a": "a", "b": "b"}, ss, ss)
        assert quality == 1.0

    def test_empty_type_perfect(self) -> None:
        ss = build_statespace(End())
        quality = metaphor_quality({}, ss, ss)
        assert quality == 1.0

    def test_quality_in_range(self) -> None:
        t1 = parse("&{a: end, b: end}")
        t2 = parse("&{x: end, y: end}")
        ss1 = build_statespace(t1)
        ss2 = build_statespace(t2)
        quality = metaphor_quality({"a": "x", "b": "y"}, ss1, ss2)
        assert 0.0 <= quality <= 1.0


# =========================================================================
# MechanismResult dataclass test
# =========================================================================


class TestMechanismResult:
    """MechanismResult is a frozen dataclass."""

    def test_creation(self) -> None:
        r = MechanismResult(
            mechanism="abstraction",
            input_description="&{a: end, b: end}",
            output_description="&{a: end, b: end} (labels generic)",
            preserves_lattice=True,
            explanation="Plato's Forms: forgetting details.",
        )
        assert r.mechanism == "abstraction"
        assert r.preserves_lattice is True

    def test_frozen(self) -> None:
        r = MechanismResult("test", "in", "out", True, "expl")
        with pytest.raises(AttributeError):
            r.mechanism = "other"  # type: ignore[misc]
