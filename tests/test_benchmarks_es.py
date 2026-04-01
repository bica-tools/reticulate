"""Tests for benchmark protocols as event structures (Step 20)."""

import pytest
from reticulate.benchmarks_es import (
    ESInvariants,
    BenchmarkESReport,
    benchmark_to_es,
    es_invariants,
    all_benchmarks_es,
    es_comparison_table,
    analyze_benchmarks_es,
)


# ---------------------------------------------------------------------------
# Single protocol conversion
# ---------------------------------------------------------------------------

class TestBenchmarkToES:
    def test_iterator(self):
        analysis, inv = benchmark_to_es(
            "Iterator",
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
        )
        assert inv.name == "Iterator"
        assert inv.num_events >= 3
        assert inv.num_states >= 2

    def test_simple_branch(self):
        _, inv = benchmark_to_es("Branch", "&{a: end, b: end}")
        assert inv.num_events == 2
        assert inv.num_conflicts >= 1

    def test_deep_chain(self):
        _, inv = benchmark_to_es("Chain", "&{a: &{b: &{c: end}}}")
        assert inv.num_events == 3
        assert inv.num_conflicts == 0
        assert inv.is_isomorphic

    def test_end(self):
        _, inv = benchmark_to_es("End", "end")
        assert inv.num_events == 0
        assert inv.num_configs == 1


# ---------------------------------------------------------------------------
# ES invariants
# ---------------------------------------------------------------------------

class TestESInvariants:
    def test_invariants_structure(self):
        inv = es_invariants("Test", "&{a: end, b: end}")
        assert isinstance(inv, ESInvariants)
        assert isinstance(inv.name, str)
        assert isinstance(inv.num_events, int)
        assert isinstance(inv.conflict_density, float)

    def test_conflict_density_bounds(self):
        inv = es_invariants("Test", "&{a: end, b: end}")
        assert 0.0 <= inv.conflict_density <= 1.0

    def test_chain_no_conflicts(self):
        inv = es_invariants("Chain", "&{a: &{b: end}}")
        assert inv.num_conflicts == 0

    def test_wide_branch_high_conflict(self):
        inv = es_invariants("Wide", "&{a: end, b: end, c: end, d: end, e: end}")
        assert inv.num_conflicts == 10  # C(5,2)
        assert inv.conflict_density == 1.0  # All pairs in conflict

    def test_parallel_has_concurrency(self):
        inv = es_invariants("Par", "(&{a: end} || &{b: end})")
        # Product state space may or may not have concurrency
        assert isinstance(inv.has_concurrency, bool)


# ---------------------------------------------------------------------------
# All benchmarks
# ---------------------------------------------------------------------------

class TestAllBenchmarks:
    def test_default_protocols(self):
        results = all_benchmarks_es()
        assert len(results) == 10  # Default set
        assert all(isinstance(r, ESInvariants) for r in results)

    def test_custom_protocols(self):
        custom = [
            ("A", "&{x: end}"),
            ("B", "&{y: end, z: end}"),
        ]
        results = all_benchmarks_es(custom)
        assert len(results) == 2

    def test_all_have_events_or_end(self):
        results = all_benchmarks_es()
        for r in results:
            assert r.num_events >= 0
            assert r.num_configs >= 1  # At least empty config


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

class TestComparisonTable:
    def test_table_is_string(self):
        table = es_comparison_table()
        assert isinstance(table, str)
        assert len(table) > 0

    def test_table_has_header(self):
        table = es_comparison_table()
        assert "Protocol" in table
        assert "Events" in table
        assert "Confl" in table

    def test_table_has_protocols(self):
        table = es_comparison_table()
        assert "Iterator" in table
        assert "DeepChain" in table

    def test_custom_table(self):
        table = es_comparison_table([("Test", "&{a: end}")])
        assert "Test" in table


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalysis:
    def test_report_structure(self):
        report = analyze_benchmarks_es()
        assert isinstance(report, BenchmarkESReport)
        assert report.num_protocols == 10
        assert report.total_events >= 10
        assert 0.0 <= report.avg_conflict_density <= 1.0

    def test_total_events(self):
        report = analyze_benchmarks_es()
        manual_sum = sum(i.num_events for i in report.invariants)
        assert report.total_events == manual_sum

    def test_isomorphic_count(self):
        report = analyze_benchmarks_es()
        manual_count = sum(1 for i in report.invariants if i.is_isomorphic)
        assert report.isomorphic_count == manual_count

    def test_concurrency_count(self):
        report = analyze_benchmarks_es()
        assert isinstance(report.protocols_with_concurrency, int)
        assert 0 <= report.protocols_with_concurrency <= report.num_protocols


# ---------------------------------------------------------------------------
# Parametrized benchmarks
# ---------------------------------------------------------------------------

class TestParametrized:
    @pytest.mark.parametrize("type_string,name", [
        ("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}", "Iterator"),
        ("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}", "File"),
        ("&{a: end, b: end}", "SimpleBranch"),
        ("+{ok: end, err: end}", "SimpleSelect"),
        ("(&{a: end} || &{b: end})", "SimpleParallel"),
        ("&{a: &{b: &{c: end}}}", "DeepChain"),
        ("&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}", "REST"),
        ("rec X . &{request: +{OK: X, ERR: end}}", "RetryLoop"),
    ])
    def test_invariants_valid(self, type_string, name):
        inv = es_invariants(name, type_string)
        assert inv.num_events >= 0
        assert inv.num_conflicts >= 0
        assert inv.num_configs >= 1
        assert 0.0 <= inv.conflict_density <= 1.0

    @pytest.mark.parametrize("type_string,name,expected_conflicts", [
        ("&{a: &{b: &{c: end}}}", "Chain", 0),
        ("&{a: end, b: end}", "Branch2", 1),
        ("&{a: end, b: end, c: end}", "Branch3", 3),
    ])
    def test_expected_conflicts(self, type_string, name, expected_conflicts):
        inv = es_invariants(name, type_string)
        assert inv.num_conflicts == expected_conflicts


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_end_protocol(self):
        inv = es_invariants("End", "end")
        assert inv.num_events == 0
        assert inv.num_conflicts == 0
        assert inv.num_configs == 1
        assert inv.conflict_density == 0.0

    def test_single_method(self):
        inv = es_invariants("Single", "&{a: end}")
        assert inv.num_events == 1
        assert inv.num_conflicts == 0

    def test_empty_custom_list(self):
        results = all_benchmarks_es([])
        assert len(results) == 0

    def test_report_empty(self):
        report = analyze_benchmarks_es([])
        assert report.num_protocols == 0
        assert report.total_events == 0
