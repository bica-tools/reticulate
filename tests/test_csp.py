"""Tests for CSP trace semantics (Step 27)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.csp import (
    CSPResult,
    FailuresRefinementResult,
    Trace,
    alphabet,
    check_failures_refinement,
    csp_analysis,
    csp_from_type,
    extract_complete_traces,
    extract_traces,
    failures,
    failures_equivalent,
    is_deterministic,
    pretty_failures,
    pretty_trace,
    trace_equivalent,
    trace_refinement,
    traces_from_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_str: str):
    return build_statespace(parse(type_str))


# ---------------------------------------------------------------------------
# 1. extract_traces
# ---------------------------------------------------------------------------

class TestExtractTraces:
    def test_end_only(self):
        ss = _ss("end")
        traces = extract_traces(ss)
        assert traces == frozenset({()})

    def test_single_branch(self):
        ss = _ss("&{a: end}")
        traces = extract_traces(ss)
        assert () in traces
        assert ("a",) in traces
        assert len(traces) == 2

    def test_two_branches(self):
        ss = _ss("&{a: end, b: end}")
        traces = extract_traces(ss)
        assert () in traces
        assert ("a",) in traces
        assert ("b",) in traces
        assert len(traces) == 3

    def test_sequential(self):
        ss = _ss("&{a: &{b: end}}")
        traces = extract_traces(ss)
        assert () in traces
        assert ("a",) in traces
        assert ("a", "b") in traces
        assert len(traces) == 3

    def test_selection(self):
        ss = _ss("+{ok: end, err: end}")
        traces = extract_traces(ss)
        assert () in traces
        assert ("ok",) in traces
        assert ("err",) in traces

    def test_max_depth_limits(self):
        ss = _ss("rec X . &{a: X}")
        traces_1 = extract_traces(ss, max_depth=1)
        traces_3 = extract_traces(ss, max_depth=3)
        assert ("a",) in traces_1
        assert ("a", "a", "a") in traces_3
        assert len(traces_3) > len(traces_1)

    def test_parallel(self):
        ss = _ss("(&{a: end} || &{b: end})")
        traces = extract_traces(ss)
        assert () in traces
        # Both interleavings
        assert ("a", "b") in traces
        assert ("b", "a") in traces


# ---------------------------------------------------------------------------
# 2. extract_complete_traces
# ---------------------------------------------------------------------------

class TestExtractCompleteTraces:
    def test_end_only(self):
        ss = _ss("end")
        ct = extract_complete_traces(ss)
        assert ct == frozenset({()})

    def test_single_branch(self):
        ss = _ss("&{a: end}")
        ct = extract_complete_traces(ss)
        assert ct == frozenset({("a",)})

    def test_two_branches(self):
        ss = _ss("&{a: end, b: end}")
        ct = extract_complete_traces(ss)
        assert ct == frozenset({("a",), ("b",)})

    def test_sequential_complete(self):
        ss = _ss("&{a: &{b: end}}")
        ct = extract_complete_traces(ss)
        assert ct == frozenset({("a", "b")})
        # Intermediate prefix (a,) is NOT complete
        assert ("a",) not in ct

    def test_recursive_complete(self):
        ss = _ss("rec X . &{a: X, b: end}")
        ct = extract_complete_traces(ss, max_depth=3)
        assert ("b",) in ct
        assert ("a", "b") in ct
        assert ("a", "a", "b") in ct


# ---------------------------------------------------------------------------
# 3. trace_refinement
# ---------------------------------------------------------------------------

class TestTraceRefinement:
    def test_same_traces(self):
        t = frozenset({(), ("a",)})
        assert trace_refinement(t, t)

    def test_subset_refines(self):
        t1 = frozenset({(), ("a",)})
        t2 = frozenset({(), ("a",), ("b",)})
        assert trace_refinement(t1, t2)

    def test_superset_does_not_refine(self):
        t1 = frozenset({(), ("a",), ("b",)})
        t2 = frozenset({(), ("a",)})
        assert not trace_refinement(t1, t2)

    def test_empty_refines_everything(self):
        t1: frozenset[Trace] = frozenset()
        t2 = frozenset({(), ("a",)})
        assert trace_refinement(t1, t2)

    def test_subtype_traces_refine_supertype(self):
        """Branch with more methods (subtype) has more traces."""
        ss_super = _ss("&{a: end}")
        ss_sub = _ss("&{a: end, b: end}")
        t_super = extract_traces(ss_super)
        t_sub = extract_traces(ss_sub)
        # Supertype refines subtype (fewer traces subset of more)
        assert trace_refinement(t_super, t_sub)
        # But subtype does NOT refine supertype
        assert not trace_refinement(t_sub, t_super)


# ---------------------------------------------------------------------------
# 4. failures
# ---------------------------------------------------------------------------

class TestFailures:
    def test_end_refuses_everything(self):
        ss = _ss("&{a: end}")
        f = failures(ss)
        # After trace (a,), we are at end: refuse {a}
        end_failures = [(t, r) for t, r in f if t == ("a",)]
        assert len(end_failures) == 1
        assert "a" in end_failures[0][1]

    def test_initial_refusals(self):
        ss = _ss("&{a: end, b: end}")
        f = failures(ss)
        # At initial state, both a and b are enabled; refusal is empty
        init_failures = [(t, r) for t, r in f if t == ()]
        assert len(init_failures) == 1
        assert init_failures[0][1] == frozenset()

    def test_partial_refusal(self):
        ss = _ss("&{a: &{b: end}}")
        alpha = alphabet(ss)
        f = failures(ss)
        # After (a,), only b is enabled, so a is refused
        after_a = [(t, r) for t, r in f if t == ("a",)]
        assert len(after_a) == 1
        assert "a" in after_a[0][1]
        assert "b" not in after_a[0][1]


# ---------------------------------------------------------------------------
# 5. check_failures_refinement
# ---------------------------------------------------------------------------

class TestFailuresRefinement:
    def test_identical_processes(self):
        ss = _ss("&{a: end}")
        result = check_failures_refinement(ss, ss)
        assert result.is_refinement
        assert result.traces_refined
        assert result.failures_refined

    def test_non_refinement(self):
        ss1 = _ss("&{a: end, b: end}")
        ss2 = _ss("&{a: end}")
        result = check_failures_refinement(ss1, ss2)
        # ss1 has trace (b,) not in ss2
        assert not result.traces_refined
        assert not result.is_refinement
        assert len(result.trace_counterexamples) > 0


# ---------------------------------------------------------------------------
# 6. traces_from_type
# ---------------------------------------------------------------------------

class TestTracesFromType:
    def test_simple(self):
        traces = traces_from_type("&{a: end}")
        assert ("a",) in traces
        assert () in traces

    def test_nested(self):
        traces = traces_from_type("&{a: &{b: end}}")
        assert ("a", "b") in traces


# ---------------------------------------------------------------------------
# 7. csp_analysis / csp_from_type
# ---------------------------------------------------------------------------

class TestCSPAnalysis:
    def test_csp_result_fields(self):
        ss = _ss("&{a: end, b: end}")
        result = csp_analysis(ss)
        assert isinstance(result, CSPResult)
        assert () in result.traces
        assert ("a",) in result.complete_traces
        assert result.alphabet == frozenset({"a", "b"})
        assert len(result.failures) > 0

    def test_csp_from_type(self):
        result = csp_from_type("&{a: end}")
        assert isinstance(result, CSPResult)
        assert ("a",) in result.traces


# ---------------------------------------------------------------------------
# 8. pretty_trace
# ---------------------------------------------------------------------------

class TestPrettyTrace:
    def test_empty_trace(self):
        assert pretty_trace(()) == "\u27e8\u27e9"

    def test_single_event(self):
        assert pretty_trace(("a",)) == "\u27e8a\u27e9"

    def test_multiple_events(self):
        assert pretty_trace(("a", "b", "c")) == "\u27e8a, b, c\u27e9"


# ---------------------------------------------------------------------------
# 9. pretty_failures
# ---------------------------------------------------------------------------

class TestPrettyFailures:
    def test_single_failure(self):
        f = frozenset({(("a",), frozenset({"b"}))})
        text = pretty_failures(f)
        assert "\u27e8a\u27e9" in text
        assert "{b}" in text

    def test_empty_failures(self):
        text = pretty_failures(frozenset())
        assert text == ""


# ---------------------------------------------------------------------------
# 10. alphabet
# ---------------------------------------------------------------------------

class TestAlphabet:
    def test_simple(self):
        ss = _ss("&{a: end, b: end}")
        assert alphabet(ss) == frozenset({"a", "b"})

    def test_nested(self):
        ss = _ss("&{a: &{b: end}}")
        assert alphabet(ss) == frozenset({"a", "b"})


# ---------------------------------------------------------------------------
# 11. is_deterministic
# ---------------------------------------------------------------------------

class TestIsDeterministic:
    def test_branch_is_deterministic(self):
        ss = _ss("&{a: end, b: end}")
        assert is_deterministic(ss)

    def test_selection_is_deterministic(self):
        ss = _ss("+{ok: end, err: end}")
        assert is_deterministic(ss)

    def test_sequential_is_deterministic(self):
        ss = _ss("&{a: &{b: end}}")
        assert is_deterministic(ss)


# ---------------------------------------------------------------------------
# 12. trace_equivalent / failures_equivalent
# ---------------------------------------------------------------------------

class TestEquivalence:
    def test_trace_equivalent_same(self):
        ss = _ss("&{a: end}")
        assert trace_equivalent(ss, ss)

    def test_trace_not_equivalent(self):
        ss1 = _ss("&{a: end}")
        ss2 = _ss("&{b: end}")
        assert not trace_equivalent(ss1, ss2)

    def test_failures_equivalent_same(self):
        ss = _ss("&{a: end, b: end}")
        assert failures_equivalent(ss, ss)

    def test_failures_not_equivalent(self):
        ss1 = _ss("&{a: end}")
        ss2 = _ss("&{a: end, b: end}")
        assert not failures_equivalent(ss1, ss2)


# ---------------------------------------------------------------------------
# 13. Integration: recursive types
# ---------------------------------------------------------------------------

class TestRecursiveTraces:
    def test_recursive_traces_bounded(self):
        ss = _ss("rec X . &{a: X}")
        traces = extract_traces(ss, max_depth=3)
        assert () in traces
        assert ("a",) in traces
        assert ("a", "a") in traces
        assert ("a", "a", "a") in traces
        # 4 traces: empty, a, aa, aaa
        assert len(traces) == 4

    def test_recursive_with_exit(self):
        ss = _ss("rec X . &{a: X, b: end}")
        ct = extract_complete_traces(ss, max_depth=4)
        assert ("b",) in ct
        assert ("a", "b") in ct
        assert ("a", "a", "b") in ct


# ---------------------------------------------------------------------------
# 14. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_parallel_complete_traces(self):
        ss = _ss("(&{a: end} || &{b: end})")
        ct = extract_complete_traces(ss)
        assert ("a", "b") in ct
        assert ("b", "a") in ct

    def test_deeply_nested(self):
        ss = _ss("&{a: &{b: &{c: end}}}")
        ct = extract_complete_traces(ss)
        assert ct == frozenset({("a", "b", "c")})

    def test_max_depth_zero(self):
        ss = _ss("&{a: end}")
        traces = extract_traces(ss, max_depth=0)
        # Only the empty trace
        assert traces == frozenset({()})
