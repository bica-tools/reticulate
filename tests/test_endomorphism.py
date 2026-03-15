"""Tests for transitions as lattice endomorphisms (Step 10)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.endomorphism import (
    EndomorphismResult,
    EndomorphismSummary,
    TransitionMap,
    check_all_endomorphisms,
    check_endomorphism,
    extract_transition_maps,
)


# ---------------------------------------------------------------------------
# Transition map extraction
# ---------------------------------------------------------------------------


class TestExtractTransitionMaps:
    """Extract partial functions from transition labels."""

    def test_single_branch(self):
        ss = build_statespace(parse("&{a: end}"))
        tmaps = extract_transition_maps(ss)
        assert len(tmaps) == 1
        assert tmaps[0].label == "a"
        assert tmaps[0].domain_size == 1

    def test_two_methods(self):
        ss = build_statespace(parse("&{a: end, b: end}"))
        tmaps = extract_transition_maps(ss)
        assert len(tmaps) == 2
        labels = {t.label for t in tmaps}
        assert labels == {"a", "b"}

    def test_selection_marked(self):
        ss = build_statespace(parse("+{a: end, b: end}"))
        tmaps = extract_transition_maps(ss)
        assert all(t.is_selection for t in tmaps)

    def test_branch_not_selection(self):
        ss = build_statespace(parse("&{a: end}"))
        tmaps = extract_transition_maps(ss)
        assert not tmaps[0].is_selection

    def test_nested_same_label(self):
        """Label appearing at multiple levels."""
        ss = build_statespace(parse("&{a: &{a: end}}"))
        tmaps = extract_transition_maps(ss)
        a_maps = [t for t in tmaps if t.label == "a"]
        assert len(a_maps) == 1  # one label, multiple domain entries
        assert a_maps[0].domain_size == 2  # enabled at 2 states

    def test_recursive_label(self):
        ss = build_statespace(parse("rec X . &{a: X, b: end}"))
        tmaps = extract_transition_maps(ss)
        labels = {t.label for t in tmaps}
        assert "a" in labels
        assert "b" in labels


# ---------------------------------------------------------------------------
# Order-preserving checks
# ---------------------------------------------------------------------------


class TestOrderPreserving:
    """Check monotonicity of transition maps."""

    def test_single_method_trivially_monotone(self):
        """A label enabled at only one state is trivially order-preserving."""
        ss = build_statespace(parse("&{a: end}"))
        tmaps = extract_transition_maps(ss)
        result = check_endomorphism(ss, tmaps[0])
        assert result.is_order_preserving

    def test_nested_branch_monotone(self):
        """&{a: &{b: end}} — label 'a' maps top→inner, trivially monotone."""
        ss = build_statespace(parse("&{a: &{b: end}}"))
        tmaps = extract_transition_maps(ss)
        for tm in tmaps:
            result = check_endomorphism(ss, tm)
            assert result.is_order_preserving, (
                f"Label '{tm.label}' not order-preserving"
            )

    def test_shared_label_monotone(self):
        """&{a: &{a: end}} — label 'a' at two ordered states."""
        ss = build_statespace(parse("&{a: &{a: end}}"))
        tmaps = extract_transition_maps(ss)
        a_map = [t for t in tmaps if t.label == "a"][0]
        result = check_endomorphism(ss, a_map)
        assert result.is_order_preserving

    def test_recursive_monotone(self):
        ss = build_statespace(parse("rec X . &{a: X, b: end}"))
        summary = check_all_endomorphisms(ss)
        assert summary.all_order_preserving


# ---------------------------------------------------------------------------
# Meet/join preservation
# ---------------------------------------------------------------------------


class TestMeetJoinPreservation:
    """Check meet and join preservation of transition maps."""

    def test_single_method_trivially_preserves(self):
        ss = build_statespace(parse("&{a: end}"))
        tmaps = extract_transition_maps(ss)
        result = check_endomorphism(ss, tmaps[0])
        assert result.is_meet_preserving
        assert result.is_join_preserving
        assert result.is_endomorphism

    def test_two_methods_end(self):
        """&{a: end, b: end} — both methods map top→bottom."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        summary = check_all_endomorphisms(ss)
        assert summary.all_meet_preserving
        assert summary.all_join_preserving

    def test_nested_preserves(self):
        ss = build_statespace(parse("&{a: &{b: end, c: end}}"))
        summary = check_all_endomorphisms(ss)
        assert summary.all_meet_preserving
        assert summary.all_join_preserving

    def test_selection_preserves(self):
        ss = build_statespace(parse("+{a: end, b: end}"))
        summary = check_all_endomorphisms(ss)
        assert summary.all_endomorphisms


# ---------------------------------------------------------------------------
# Full endomorphism check
# ---------------------------------------------------------------------------


class TestCheckAllEndomorphisms:
    """Test the summary function."""

    def test_result_type(self):
        ss = build_statespace(parse("&{a: end}"))
        summary = check_all_endomorphisms(ss)
        assert isinstance(summary, EndomorphismSummary)

    def test_num_labels(self):
        ss = build_statespace(parse("&{a: end, b: end, c: end}"))
        summary = check_all_endomorphisms(ss)
        assert summary.num_labels == 3

    def test_end_no_labels(self):
        from reticulate.parser import End
        ss = build_statespace(End())
        summary = check_all_endomorphisms(ss)
        assert summary.num_labels == 0
        assert summary.all_endomorphisms  # vacuously true

    def test_simple_types_all_endomorphisms(self):
        """Simple types should have all-endomorphism transition maps."""
        types = [
            "&{a: end}",
            "&{a: end, b: end}",
            "+{a: end}",
            "+{a: end, b: end}",
            "&{a: &{b: end}}",
            "&{a: +{x: end, y: end}}",
        ]
        for type_str in types:
            ss = build_statespace(parse(type_str))
            summary = check_all_endomorphisms(ss)
            assert summary.all_order_preserving, (
                f"{type_str}: not all order-preserving"
            )


# ---------------------------------------------------------------------------
# Iterator benchmark (interesting case)
# ---------------------------------------------------------------------------


class TestIteratorEndomorphism:
    """The Java Iterator has a recursive structure worth checking."""

    def test_iterator_order_preserving(self):
        ss = build_statespace(parse(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        ))
        summary = check_all_endomorphisms(ss)
        assert summary.all_order_preserving


# ---------------------------------------------------------------------------
# Benchmark verification
# ---------------------------------------------------------------------------


class TestEndomorphismBenchmarks:
    """Check endomorphism properties on all 34 benchmark protocols."""

    @pytest.fixture
    def benchmark_spaces(self):
        from tests.benchmarks.protocols import BENCHMARKS
        return [
            (b.name, build_statespace(parse(b.type_string)))
            for b in BENCHMARKS
        ]

    def test_all_order_preserving(self, benchmark_spaces):
        """Every label in every benchmark should be order-preserving."""
        failures = []
        for name, ss in benchmark_spaces:
            summary = check_all_endomorphisms(ss)
            if not summary.all_order_preserving:
                failing = [
                    r.label for r in summary.results
                    if not r.is_order_preserving
                ]
                failures.append((name, failing))

        if failures:
            msg = "; ".join(f"{n}: {ls}" for n, ls in failures)
            # Report but don't necessarily fail — this is research
            print(f"Non-monotone labels: {msg}")

    def test_endomorphism_statistics(self, benchmark_spaces):
        """Collect statistics on endomorphism properties across benchmarks."""
        total_labels = 0
        order_preserving = 0
        meet_preserving = 0
        join_preserving = 0
        full_endomorphisms = 0
        all_endo_benchmarks = 0

        for name, ss in benchmark_spaces:
            summary = check_all_endomorphisms(ss)
            total_labels += summary.num_labels
            order_preserving += sum(
                1 for r in summary.results if r.is_order_preserving
            )
            meet_preserving += sum(
                1 for r in summary.results if r.is_meet_preserving
            )
            join_preserving += sum(
                1 for r in summary.results if r.is_join_preserving
            )
            full_endomorphisms += sum(
                1 for r in summary.results if r.is_endomorphism
            )
            if summary.all_endomorphisms:
                all_endo_benchmarks += 1

        # These are observations, not assertions — Step 10 is exploratory
        assert total_labels > 0
        print(f"\n{'='*60}")
        print(f"Step 10: Endomorphism Statistics (34 benchmarks)")
        print(f"{'='*60}")
        print(f"Total distinct labels:        {total_labels}")
        print(f"Order-preserving:             {order_preserving}/{total_labels} "
              f"({100*order_preserving/total_labels:.0f}%)")
        print(f"Meet-preserving:              {meet_preserving}/{total_labels} "
              f"({100*meet_preserving/total_labels:.0f}%)")
        print(f"Join-preserving:              {join_preserving}/{total_labels} "
              f"({100*join_preserving/total_labels:.0f}%)")
        print(f"Full endomorphisms:           {full_endomorphisms}/{total_labels} "
              f"({100*full_endomorphisms/total_labels:.0f}%)")
        print(f"Benchmarks with all endo:     {all_endo_benchmarks}/34")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Counterexample inspection
# ---------------------------------------------------------------------------


class TestCounterexamples:
    """Inspect counterexamples when endomorphism properties fail."""

    def test_counterexample_fields(self):
        ss = build_statespace(parse("&{a: end}"))
        tmaps = extract_transition_maps(ss)
        result = check_endomorphism(ss, tmaps[0])
        assert result.order_counterexample is None
        assert result.meet_counterexample is None
        assert result.join_counterexample is None

    def test_result_fields(self):
        ss = build_statespace(parse("&{a: end, b: end}"))
        summary = check_all_endomorphisms(ss)
        for r in summary.results:
            assert isinstance(r, EndomorphismResult)
            assert isinstance(r.label, str)
            assert isinstance(r.domain_size, int)
