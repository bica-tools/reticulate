"""Tests for context-free session type analysis (Step 14)."""

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
from reticulate.context_free import (
    ChomskyClassification,
    analyze_trace_language,
    classify_chomsky,
    has_continuation_recursion,
    is_regular,
    stack_depth_bound,
)


# ---------------------------------------------------------------------------
# Regularity
# ---------------------------------------------------------------------------


class TestRegularity:
    """Test regularity checking."""

    def test_end_regular(self):
        assert is_regular(End())

    def test_simple_branch_regular(self):
        assert is_regular(parse("&{a: end, b: end}"))

    def test_tail_recursive_regular(self):
        assert is_regular(parse("rec X . &{a: X, b: end}"))

    def test_nested_tail_recursive_regular(self):
        assert is_regular(parse(
            "rec X . &{a: rec Y . &{b: Y, c: X}, d: end}"
        ))

    def test_iterator_regular(self):
        assert is_regular(parse(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        ))

    def test_parallel_without_recursion_regular(self):
        assert is_regular(parse("(&{a: wait} || &{b: wait}) . &{c: end}"))


# ---------------------------------------------------------------------------
# Context-free (continuation recursion)
# ---------------------------------------------------------------------------


class TestContextFree:
    """Test context-free detection."""

    def test_no_continuation_recursion(self):
        assert not has_continuation_recursion(parse("rec X . &{a: X}"))

    def test_no_continuation_recursion_parallel(self):
        # Parallel with recursion but not via continuation
        s = parse(
            "rec X . &{a: (&{b: wait} || &{c: wait}) . &{d: X}, e: end}"
        )
        # The recursion is in the continuation's right side, which is fine
        # because the recursive var appears after the parallel join
        assert not has_continuation_recursion(s)

    def test_parallel_non_recursive_regular(self):
        s = parse("(&{a: wait} || &{b: wait}) . &{c: end}")
        assert is_regular(s)


# ---------------------------------------------------------------------------
# Stack depth
# ---------------------------------------------------------------------------


class TestStackDepth:
    """Test stack depth bound computation."""

    def test_no_recursion_zero(self):
        assert stack_depth_bound(End()) == 0

    def test_tail_recursive_zero(self):
        assert stack_depth_bound(parse("rec X . &{a: X, b: end}")) == 0

    def test_simple_regular_zero(self):
        assert stack_depth_bound(parse("&{a: end}")) == 0


# ---------------------------------------------------------------------------
# Chomsky classification
# ---------------------------------------------------------------------------


class TestChomskyClassification:
    """Test Chomsky hierarchy classification."""

    def test_end_finite(self):
        result = classify_chomsky(End())
        assert result.level == "regular"
        assert result.is_regular
        assert "finite" in result.trace_language_class

    def test_branch_finite(self):
        result = classify_chomsky(parse("&{a: end}"))
        assert result.level == "regular"
        assert result.num_rec_binders == 0

    def test_tail_recursive_regular(self):
        result = classify_chomsky(parse("rec X . &{a: X, b: end}"))
        assert result.level == "regular"
        assert result.is_regular
        assert "tail-recursive" in result.trace_language_class

    def test_result_fields(self):
        result = classify_chomsky(parse("rec X . &{a: X, b: end}"))
        assert isinstance(result, ChomskyClassification)
        assert result.num_rec_binders == 1
        assert result.max_rec_depth == 1
        assert result.stack_depth_bound == 0


# ---------------------------------------------------------------------------
# Trace language analysis
# ---------------------------------------------------------------------------


class TestTraceLanguage:
    """Test trace language analysis."""

    def test_finite_trace(self):
        result = analyze_trace_language(parse("&{a: end}"))
        assert result["is_finite"]
        assert result["is_regular"]
        assert result["chomsky_class"] == "regular"
        assert not result["uses_recursion"]

    def test_regular_trace(self):
        result = analyze_trace_language(parse("rec X . &{a: X, b: end}"))
        assert not result["is_finite"]
        assert result["is_regular"]
        assert result["uses_recursion"]

    def test_parallel_trace(self):
        result = analyze_trace_language(
            parse("(&{a: wait} || &{b: wait}) . &{c: end}")
        )
        assert result["uses_parallel"]
        assert result["uses_continuation"]

    def test_state_count(self):
        result = analyze_trace_language(parse("&{a: end, b: end}"))
        assert result["state_count"] == 2
        assert result["transition_count"] == 2


# ---------------------------------------------------------------------------
# Benchmark classification
# ---------------------------------------------------------------------------


class TestContextFreeBenchmarks:
    """Classify all 34 benchmarks in the Chomsky hierarchy."""

    @pytest.fixture
    def benchmark_types(self):
        from tests.benchmarks.protocols import BENCHMARKS
        return [(b.name, parse(b.type_string)) for b in BENCHMARKS]

    def test_all_have_finite_statespace(self, benchmark_types):
        """All benchmarks produce finite state spaces (by construction)."""
        from reticulate.statespace import build_statespace
        for name, s in benchmark_types:
            ss = build_statespace(s)
            assert len(ss.states) < 100, f"{name}: too many states"

    def test_classification_statistics(self, benchmark_types):
        total = len(benchmark_types)
        finite = 0
        regular = 0
        context_free = 0
        with_parallel = 0
        with_continuation = 0

        for name, s in benchmark_types:
            result = classify_chomsky(s)
            trace = analyze_trace_language(s)

            if result.num_rec_binders == 0:
                finite += 1
            elif result.is_regular:
                regular += 1
            else:
                context_free += 1

            if trace["uses_parallel"]:
                with_parallel += 1
            if trace["uses_continuation"]:
                with_continuation += 1

        print(f"\n{'='*60}")
        print(f"Step 14: Chomsky Classification ({total} benchmarks)")
        print(f"{'='*60}")
        print(f"Finite (no recursion):    {finite}/{total}")
        print(f"Regular (tail-recursive): {regular}/{total}")
        print(f"Context-free:             {context_free}/{total}")
        print(f"Uses parallel:            {with_parallel}/{total}")
        print(f"Uses continuation:        {with_continuation}/{total}")
        print(f"{'='*60}")

    def test_per_benchmark_classification(self, benchmark_types):
        """Print per-benchmark classification."""
        for name, s in benchmark_types:
            result = classify_chomsky(s)
            assert result is not None, f"{name}: classification failed"
