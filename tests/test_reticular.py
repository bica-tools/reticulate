"""Tests for reticular form: characterisation and reconstruction (Step 9)."""

import pytest

from reticulate.parser import (
    Branch, End, Rec, Select, Var, parse, pretty,
)
from reticulate.statespace import StateSpace, build_statespace
from reticulate.reticular import (
    ReticularFormResult,
    StateClassification,
    check_reticular_form,
    classify_all_states,
    classify_state,
    is_reticulate,
    reconstruct,
)


# ---------------------------------------------------------------------------
# State classification
# ---------------------------------------------------------------------------


class TestClassifyState:
    """Classify individual states as branch, select, or end."""

    def test_end_state(self):
        ss = build_statespace(parse("&{a: end}"))
        c = classify_state(ss, ss.bottom)
        assert c.kind == "end"
        assert c.labels == ()

    def test_branch_state(self):
        ss = build_statespace(parse("&{a: end, b: end}"))
        c = classify_state(ss, ss.top)
        assert c.kind == "branch"
        assert set(c.labels) == {"a", "b"}

    def test_selection_state(self):
        ss = build_statespace(parse("+{a: end, b: end}"))
        c = classify_state(ss, ss.top)
        assert c.kind == "select"
        assert set(c.labels) == {"a", "b"}

    def test_nested_branch_select(self):
        """&{a: +{x: end}} — top is branch, inner is select."""
        ss = build_statespace(parse("&{a: +{x: end}}"))
        top_c = classify_state(ss, ss.top)
        assert top_c.kind == "branch"

        # Find the inner state
        inner = [t for _, l, t in ss.transitions if l == "a"][0]
        inner_c = classify_state(ss, inner)
        assert inner_c.kind == "select"


class TestClassifyAllStates:
    """Classify all states in a state space."""

    def test_simple_branch(self):
        ss = build_statespace(parse("&{a: end}"))
        classifications = classify_all_states(ss)
        kinds = {c.kind for c in classifications}
        assert "branch" in kinds or "end" in kinds

    def test_all_classified(self):
        ss = build_statespace(parse("&{a: end, b: &{c: end}}"))
        classifications = classify_all_states(ss)
        assert len(classifications) == len(ss.states)


# ---------------------------------------------------------------------------
# Reconstruction: StateSpace → SessionType
# ---------------------------------------------------------------------------


class TestReconstructBasic:
    """Reconstruct session types from their state spaces."""

    def test_end(self):
        ss = build_statespace(End())
        result = reconstruct(ss)
        assert result == End()

    def test_single_branch(self):
        ss = build_statespace(parse("&{a: end}"))
        result = reconstruct(ss)
        assert isinstance(result, Branch)
        assert len(result.choices) == 1
        assert result.choices[0][0] == "a"
        assert result.choices[0][1] == End()

    def test_single_selection(self):
        ss = build_statespace(parse("+{a: end}"))
        result = reconstruct(ss)
        assert isinstance(result, Select)
        assert len(result.choices) == 1

    def test_two_method_branch(self):
        ss = build_statespace(parse("&{a: end, b: end}"))
        result = reconstruct(ss)
        assert isinstance(result, Branch)
        labels = {l for l, _ in result.choices}
        assert labels == {"a", "b"}

    def test_two_label_selection(self):
        ss = build_statespace(parse("+{x: end, y: end}"))
        result = reconstruct(ss)
        assert isinstance(result, Select)
        labels = {l for l, _ in result.choices}
        assert labels == {"x", "y"}

    def test_nested_branch(self):
        ss = build_statespace(parse("&{a: &{b: end}}"))
        result = reconstruct(ss)
        assert isinstance(result, Branch)
        inner = dict(result.choices)["a"]
        assert isinstance(inner, Branch)
        assert dict(inner.choices)["b"] == End()

    def test_nested_mixed(self):
        """&{a: +{x: end, y: end}}."""
        ss = build_statespace(parse("&{a: +{x: end, y: end}}"))
        result = reconstruct(ss)
        assert isinstance(result, Branch)
        inner = dict(result.choices)["a"]
        assert isinstance(inner, Select)
        assert len(inner.choices) == 2


class TestReconstructRecursion:
    """Reconstruct recursive types."""

    def test_simple_recursion(self):
        """rec X . &{a: X, b: end} should reconstruct with a Rec node."""
        ss = build_statespace(parse("rec X . &{a: X, b: end}"))
        result = reconstruct(ss)
        assert isinstance(result, Rec)
        assert isinstance(result.body, Branch)
        labels = dict(result.body.choices)
        assert result.var in str(labels)  # var reference in body

    def test_recursive_selection(self):
        ss = build_statespace(parse("rec X . +{a: X, b: end}"))
        result = reconstruct(ss)
        assert isinstance(result, Rec)
        assert isinstance(result.body, Select)

    def test_recursive_iterator(self):
        """rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}."""
        ss = build_statespace(parse(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        ))
        result = reconstruct(ss)
        # Should reconstruct as some recursive type
        assert isinstance(result, Rec) or _contains_rec(result)


def _contains_rec(s) -> bool:
    """Check if a session type contains a Rec node."""
    if isinstance(s, Rec):
        return True
    if isinstance(s, (Branch, Select)):
        return any(_contains_rec(c) for _, c in s.choices)
    return False


# ---------------------------------------------------------------------------
# Round-trip: S → L(S) → S'
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Round-trip: reconstruct(build_statespace(S)) ≡ S."""

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "&{a: end, b: end}",
        "&{a: end, b: end, c: end}",
        "+{a: end}",
        "+{a: end, b: end}",
        "&{a: &{b: end}}",
        "&{a: +{x: end, y: end}}",
        "+{a: &{x: end, y: end}}",
        "&{a: &{b: end, c: end}, d: end}",
    ])
    def test_round_trip_preserves_structure(self, type_str: str):
        """S → L(S) → S' should produce an equivalent type."""
        original = parse(type_str)
        ss = build_statespace(original)
        reconstructed = reconstruct(ss)

        # The reconstructed type should have the same state space
        ss2 = build_statespace(reconstructed)
        assert len(ss.states) == len(ss2.states), (
            f"State count mismatch for {type_str}: "
            f"{len(ss.states)} vs {len(ss2.states)}"
        )
        assert len(ss.transitions) == len(ss2.transitions), (
            f"Transition count mismatch for {type_str}: "
            f"{len(ss.transitions)} vs {len(ss2.transitions)}"
        )

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "+{a: end}",
        "&{a: end, b: end}",
        "+{a: end, b: end}",
    ])
    def test_round_trip_same_labels(self, type_str: str):
        """Transition labels should be preserved."""
        original = parse(type_str)
        ss = build_statespace(original)
        reconstructed = reconstruct(ss)
        ss2 = build_statespace(reconstructed)

        labels1 = sorted(l for _, l, _ in ss.transitions)
        labels2 = sorted(l for _, l, _ in ss2.transitions)
        assert labels1 == labels2

    @pytest.mark.parametrize("type_str", [
        "rec X . &{a: X, b: end}",
        "rec X . +{a: X, b: end}",
    ])
    def test_round_trip_recursive(self, type_str: str):
        """Recursive types should round-trip."""
        original = parse(type_str)
        ss = build_statespace(original)
        reconstructed = reconstruct(ss)
        ss2 = build_statespace(reconstructed)

        assert len(ss.states) == len(ss2.states)
        assert len(ss.transitions) == len(ss2.transitions)


# ---------------------------------------------------------------------------
# Reticular form check
# ---------------------------------------------------------------------------


class TestIsReticulate:
    """Check reticular form on session type state spaces."""

    @pytest.mark.parametrize("type_str", [
        "end",
        "&{a: end}",
        "+{a: end}",
        "&{a: end, b: end}",
        "+{a: end, b: end}",
        "&{a: +{x: end, y: end}}",
        "+{a: &{x: end, y: end}}",
        "rec X . &{a: X, b: end}",
        "rec X . +{a: X, b: end}",
        "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
    ])
    def test_session_types_are_reticulates(self, type_str: str):
        """All session type state spaces should have reticular form."""
        ss = build_statespace(parse(type_str))
        assert is_reticulate(ss), f"{type_str} should be a reticulate"

    def test_product_state_is_reticulate(self):
        """A state space with mixed (product) transitions IS a reticulate.

        Product states arise from parallel composition and are legitimate.
        """
        ss = StateSpace(
            states={0, 1, 2},
            transitions=[(0, "a", 1), (0, "b", 2)],
            top=0,
            bottom=1,
            labels={0: "product", 1: "end", 2: "end2"},
            selection_transitions={(0, "a", 1)},
        )
        assert is_reticulate(ss)


class TestCheckReticularForm:
    """Detailed reticular form analysis."""

    def test_result_type(self):
        ss = build_statespace(parse("&{a: end}"))
        result = check_reticular_form(ss)
        assert isinstance(result, ReticularFormResult)

    def test_reticulate_has_reconstruction(self):
        ss = build_statespace(parse("&{a: end, b: end}"))
        result = check_reticular_form(ss)
        assert result.is_reticulate
        assert result.reconstructed is not None

    def test_product_state_has_classification(self):
        """Product states should be classified correctly."""
        ss = StateSpace(
            states={0, 1, 2},
            transitions=[(0, "a", 1), (0, "b", 2)],
            top=0,
            bottom=1,
            labels={0: "product", 1: "end", 2: "end2"},
            selection_transitions={(0, "a", 1)},
        )
        result = check_reticular_form(ss)
        assert result.is_reticulate
        product_states = [c for c in result.classifications if c.kind == "product"]
        assert len(product_states) == 1

    def test_classifications_present(self):
        ss = build_statespace(parse("&{a: +{x: end}}"))
        result = check_reticular_form(ss)
        assert len(result.classifications) == len(ss.states)


# ---------------------------------------------------------------------------
# Benchmark verification
# ---------------------------------------------------------------------------


class TestReticulateBenchmarks:
    """All 34 benchmark protocols should have reticular form."""

    @pytest.fixture
    def benchmark_types(self):
        from tests.benchmarks.protocols import BENCHMARKS
        return [(b.name, parse(b.type_string)) for b in BENCHMARKS]

    def test_all_benchmarks_are_reticulates(self, benchmark_types):
        """Every benchmark state space should have reticular form."""
        for name, s in benchmark_types:
            ss = build_statespace(s)
            result = check_reticular_form(ss)
            assert result.is_reticulate, (
                f"{name} is not a reticulate: {result.reason}"
            )

    def test_all_benchmarks_reconstruct(self, benchmark_types):
        """Every benchmark should successfully reconstruct."""
        for name, s in benchmark_types:
            ss = build_statespace(s)
            try:
                reconstructed = reconstruct(ss)
                assert reconstructed is not None, (
                    f"Reconstruction returned None for {name}"
                )
            except Exception as e:
                pytest.fail(f"Reconstruction failed for {name}: {e}")

    def test_non_parallel_benchmarks_round_trip(self, benchmark_types):
        """Non-parallel benchmarks should round-trip (same state count).

        Parallel benchmarks may not round-trip exactly because reconstruction
        cannot detect product structure from the flat state space.
        """
        from tests.benchmarks.protocols import BENCHMARKS

        for b in BENCHMARKS:
            # Skip benchmarks with parallel composition
            if "||" in b.type_string:
                continue

            s = parse(b.type_string)
            ss = build_statespace(s)
            reconstructed = reconstruct(ss)
            ss2 = build_statespace(reconstructed)
            assert len(ss.states) == len(ss2.states), (
                f"State count mismatch for {b.name}: "
                f"{len(ss.states)} → {len(ss2.states)}"
            )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestReticularProperties:
    """Properties of reticular form."""

    def test_end_is_simplest_reticulate(self):
        """end has the simplest reticulate: one state, no transitions."""
        ss = build_statespace(End())
        assert is_reticulate(ss)
        assert len(ss.states) == 1
        assert len(ss.transitions) == 0

    def test_dual_preserves_reticular_form(self):
        """If S is a reticulate, dual(S) should also be a reticulate."""
        from reticulate.duality import dual

        types = [
            "&{a: end}", "+{a: end}",
            "&{a: end, b: end}", "+{a: end, b: end}",
            "&{a: +{x: end, y: end}}",
            "rec X . &{a: X, b: end}",
        ]
        for type_str in types:
            s = parse(type_str)
            d = dual(s)
            ss_d = build_statespace(d)
            assert is_reticulate(ss_d), (
                f"dual({type_str}) should be a reticulate"
            )

    def test_subtype_preserves_reticular_form(self):
        """Subtypes should also have reticular form."""
        types = [
            ("&{a: end, b: end}", "&{a: end}"),
            ("+{a: end}", "+{a: end, b: end}"),
        ]
        for sub_str, sup_str in types:
            sub_ss = build_statespace(parse(sub_str))
            sup_ss = build_statespace(parse(sup_str))
            assert is_reticulate(sub_ss)
            assert is_reticulate(sup_ss)

    def test_reconstruction_preserves_selection_kind(self):
        """Reconstructed type should preserve branch vs selection."""
        for type_str in ["&{a: end, b: end}", "+{a: end, b: end}"]:
            original = parse(type_str)
            ss = build_statespace(original)
            reconstructed = reconstruct(ss)

            # Branch should stay branch, Select should stay Select
            assert type(original) == type(reconstructed), (
                f"Constructor mismatch for {type_str}: "
                f"{type(original).__name__} vs {type(reconstructed).__name__}"
            )
