"""Tests for coverage analysis: manual vs generated (TOPLAS Week 5).

Validates that:
1. Generated tests achieve higher coverage than manual tests
2. Coverage gaps in manual suites are identified
3. Generated tests reach 100% transition coverage
4. The coverage comparison is meaningful (not trivially high/low)
"""

import pytest
from reticulate.coverage_analysis import (
    MANUAL_SUITES,
    TraceCoverageResult,
    CoverageComparison,
    traces_to_valid_paths,
    compute_trace_coverage,
    compute_generated_coverage,
    compare_coverage,
    run_all_comparisons,
    format_coverage_report,
    format_latex_coverage_table,
)
from reticulate.python_api_extractor import extract_session_type as extract_py
from reticulate.java_api_extractor import extract_session_type as extract_java
from reticulate.parser import parse
from reticulate.statespace import build_statespace


ALL_APIS = [
    "sqlite3.Connection",
    "http.client.HTTPConnection",
    "smtplib.SMTP",
    "ftplib.FTP",
    "ssl.SSLSocket",
    "java.util.Iterator",
    "java.io.InputStream",
]


# ---------------------------------------------------------------------------
# Manual suite completeness
# ---------------------------------------------------------------------------

class TestManualSuites:
    def test_all_apis_have_manual_suites(self):
        for api in ALL_APIS:
            assert api in MANUAL_SUITES, f"No manual suite for {api}"

    @pytest.mark.parametrize("api_name", ALL_APIS)
    def test_manual_has_at_least_one_trace(self, api_name):
        assert len(MANUAL_SUITES[api_name]) >= 1

    @pytest.mark.parametrize("api_name", ALL_APIS)
    def test_manual_traces_are_lists(self, api_name):
        for trace in MANUAL_SUITES[api_name]:
            assert isinstance(trace, list)
            assert all(isinstance(l, str) for l in trace)


# ---------------------------------------------------------------------------
# Trace → ValidPath conversion
# ---------------------------------------------------------------------------

class TestTraceConversion:
    def test_sqlite3_traces_convert(self):
        r = extract_py("sqlite3.Connection")
        ast = parse(r.inferred_type)
        ss = build_statespace(ast)
        traces = MANUAL_SUITES["sqlite3.Connection"]
        paths = traces_to_valid_paths(ss, traces)
        assert len(paths) >= 1

    def test_iterator_traces_convert(self):
        r = extract_java("java.util.Iterator")
        ast = parse(r.inferred_type)
        ss = build_statespace(ast)
        traces = MANUAL_SUITES["java.util.Iterator"]
        paths = traces_to_valid_paths(ss, traces)
        assert len(paths) >= 1

    def test_invalid_trace_skipped(self):
        """Traces with invalid labels should be skipped, not error."""
        r = extract_py("sqlite3.Connection")
        ast = parse(r.inferred_type)
        ss = build_statespace(ast)
        bad_traces = [["nonexistent_method", "close"]]
        paths = traces_to_valid_paths(ss, bad_traces)
        assert len(paths) == 0


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------

class TestCoverageComputation:
    @pytest.mark.parametrize("api_name", ["sqlite3.Connection", "java.util.Iterator", "java.io.InputStream"])
    def test_manual_coverage_below_100(self, api_name):
        """Manual tests should NOT achieve 100% — that's the point."""
        if api_name.startswith("java"):
            r = extract_java(api_name)
        else:
            r = extract_py(api_name)
        manual = compute_trace_coverage(
            r.inferred_type, MANUAL_SUITES[api_name], "manual")
        assert manual.transition_coverage < 1.0, (
            f"{api_name}: manual achieves 100% — test suite too comprehensive"
        )

    @pytest.mark.parametrize("api_name", ["sqlite3.Connection", "java.util.Iterator", "java.io.InputStream"])
    def test_generated_coverage_is_100(self, api_name):
        """Generated tests should achieve 100% transition coverage."""
        if api_name.startswith("java"):
            r = extract_java(api_name)
        else:
            r = extract_py(api_name)
        gen = compute_generated_coverage(r.inferred_type)
        assert gen.transition_coverage == 1.0, (
            f"{api_name}: generated only {gen.transition_coverage:.0%}"
        )

    @pytest.mark.parametrize("api_name", ["sqlite3.Connection", "java.util.Iterator", "java.io.InputStream"])
    def test_manual_coverage_is_positive(self, api_name):
        """Manual tests should have SOME coverage (not 0%)."""
        if api_name.startswith("java"):
            r = extract_java(api_name)
        else:
            r = extract_py(api_name)
        manual = compute_trace_coverage(
            r.inferred_type, MANUAL_SUITES[api_name], "manual")
        assert manual.transition_coverage > 0.0

    @pytest.mark.parametrize("api_name", ["sqlite3.Connection", "java.util.Iterator"])
    def test_manual_has_uncovered_labels(self, api_name):
        """Manual suites should miss some transition labels."""
        if api_name.startswith("java"):
            r = extract_java(api_name)
        else:
            r = extract_py(api_name)
        manual = compute_trace_coverage(
            r.inferred_type, MANUAL_SUITES[api_name], "manual")
        assert len(manual.uncovered_labels) >= 1


# ---------------------------------------------------------------------------
# Coverage comparison
# ---------------------------------------------------------------------------

class TestCoverageComparison:
    def test_run_all_comparisons(self):
        comparisons = run_all_comparisons()
        assert len(comparisons) == 7

    def test_generated_always_beats_manual(self):
        comparisons = run_all_comparisons()
        for c in comparisons:
            assert c.generated.transition_coverage >= c.manual.transition_coverage, (
                f"{c.api_name}: generated ({c.generated.transition_coverage:.0%}) "
                f"< manual ({c.manual.transition_coverage:.0%})"
            )

    def test_average_gap_is_significant(self):
        """Average coverage gap should be significant (>30%)."""
        comparisons = run_all_comparisons()
        avg_gap = sum(c.coverage_gap for c in comparisons) / len(comparisons)
        assert avg_gap > 0.30, f"Average gap only {avg_gap:.0%}"

    def test_all_have_positive_gap(self):
        """Every API should show improvement from generated tests."""
        comparisons = run_all_comparisons()
        for c in comparisons:
            assert c.coverage_gap > 0, f"{c.api_name}: no improvement"

    def test_gaps_identified(self):
        """At least some APIs should have identified coverage gaps."""
        comparisons = run_all_comparisons()
        apis_with_gaps = sum(1 for c in comparisons if c.uncovered_by_manual)
        assert apis_with_gaps >= 3

    def test_sqlite3_gap_details(self):
        """sqlite3 manual tests should miss error paths."""
        comparisons = run_all_comparisons()
        sqlite = [c for c in comparisons if "sqlite3" in c.api_name][0]
        assert "INTEGRITY_ERROR" in sqlite.uncovered_by_manual

    def test_iterator_misses_remove(self):
        """Manual Iterator tests typically miss remove()."""
        comparisons = run_all_comparisons()
        iterator = [c for c in comparisons if "Iterator" in c.api_name][0]
        assert "REMOVED" in iterator.uncovered_by_manual or "remove" in iterator.uncovered_by_manual


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_report(self):
        comparisons = run_all_comparisons()
        report = format_coverage_report(comparisons)
        assert "COVERAGE" in report
        assert "Manual" in report or "manual" in report
        assert "Generated" in report or "generated" in report

    def test_latex_table(self):
        comparisons = run_all_comparisons()
        latex = format_latex_coverage_table(comparisons)
        assert r"\begin{table}" in latex
        assert "coverage" in latex.lower()

    def test_report_has_all_apis(self):
        comparisons = run_all_comparisons()
        report = format_coverage_report(comparisons)
        for api in ALL_APIS:
            assert api in report
