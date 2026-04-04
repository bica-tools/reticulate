"""Tests for Universe of Lattices (Step 80f).

Tests the transitive session type analysis framework on:
1. BICA Reborn (our own project — small call graph)
2. Jedis (real Redis client — rich call graph)
"""

import os
import pytest
from reticulate.universe import (
    analyze_project,
    format_universe_report,
    _scan_java_files,
    _discover_classes,
    _build_call_graph,
    _preload_jdk_types,
    _topological_order,
    _extract_types_bottom_up,
    _check_transitive_conformance,
    CallEdge,
    ClassLattice,
    ConformanceEdge,
    UniverseReport,
    HAS_JAVALANG,
)

BICA_SRC = os.path.join(
    os.path.dirname(__file__), "..", "..", "bica", "src", "main", "java"
)

JEDIS_SRC = "/tmp/jedis-test/src/main/java"

skip_no_javalang = pytest.mark.skipif(not HAS_JAVALANG, reason="javalang not installed")
skip_no_bica = pytest.mark.skipif(not os.path.isdir(BICA_SRC), reason="BICA source not found")
skip_no_jedis = pytest.mark.skipif(not os.path.isdir(JEDIS_SRC), reason="Jedis not cloned")


# ---------------------------------------------------------------------------
# JDK preloaded types
# ---------------------------------------------------------------------------

class TestJDKPreload:
    def test_preload_returns_dict(self):
        jdk = _preload_jdk_types()
        assert isinstance(jdk, dict)

    def test_preload_has_iterator(self):
        jdk = _preload_jdk_types()
        assert "Iterator" in jdk

    def test_preload_has_connection(self):
        jdk = _preload_jdk_types()
        assert "Connection" in jdk

    def test_preload_types_parse(self):
        from reticulate.parser import parse
        jdk = _preload_jdk_types()
        for name, st in jdk.items():
            ast = parse(st)
            assert ast is not None, f"JDK type {name} failed to parse"


# ---------------------------------------------------------------------------
# Topological ordering
# ---------------------------------------------------------------------------

class TestTopologicalOrder:
    def test_simple_chain(self):
        classes = {"A", "B", "C"}
        traces = {("A", "B"): [[]], ("B", "C"): [[]]}
        order = _topological_order(classes, traces)
        assert order.index("C") < order.index("B")
        assert order.index("B") < order.index("A")

    def test_independent_classes(self):
        classes = {"X", "Y", "Z"}
        traces = {}
        order = _topological_order(classes, traces)
        assert set(order) == classes

    def test_cycle_handled(self):
        classes = {"A", "B"}
        traces = {("A", "B"): [[]], ("B", "A"): [[]]}
        order = _topological_order(classes, traces)
        assert set(order) == classes


# ---------------------------------------------------------------------------
# BICA Reborn analysis
# ---------------------------------------------------------------------------

@skip_no_javalang
@skip_no_bica
class TestBICAUniverse:
    def test_analyze_succeeds(self):
        report = analyze_project(BICA_SRC, "BICA Reborn")
        assert isinstance(report, UniverseReport)

    def test_finds_classes(self):
        report = analyze_project(BICA_SRC, "BICA Reborn")
        assert report.classes_found >= 40

    def test_finds_java_files(self):
        report = analyze_project(BICA_SRC, "BICA Reborn")
        assert report.java_files >= 50

    def test_report_formats(self):
        report = analyze_project(BICA_SRC, "BICA Reborn")
        text = format_universe_report(report)
        assert "UNIVERSE" in text
        assert "BICA" in text


# ---------------------------------------------------------------------------
# Jedis analysis (real Redis client)
# ---------------------------------------------------------------------------

@skip_no_javalang
@skip_no_jedis
class TestJedisUniverse:
    @pytest.fixture(scope="class")
    def report(self):
        return analyze_project(JEDIS_SRC, "Jedis")

    def test_analyze_succeeds(self, report):
        assert isinstance(report, UniverseReport)

    def test_finds_many_classes(self, report):
        assert report.classes_found >= 200

    def test_finds_many_java_files(self, report):
        assert report.java_files >= 300

    def test_call_graph_has_edges(self, report):
        assert report.call_edges >= 1000

    def test_mines_session_types(self, report):
        assert report.classes_with_types >= 20

    def test_high_lattice_rate(self, report):
        """Most mined session types should form lattices."""
        assert report.lattice_rate >= 0.90

    def test_has_conformance_edges(self, report):
        assert len(report.conformance_edges) >= 50

    def test_finds_violations(self, report):
        violations = [e for e in report.conformance_edges if e.violating > 0]
        assert len(violations) >= 1, "Should find at least 1 violation"

    def test_jdk_types_matched(self, report):
        assert len(report.preloaded_matches) >= 1

    def test_has_distributive_classes(self, report):
        dist = report.classification_counts.get("distributive", 0)
        assert dist >= 5

    def test_has_boolean_classes(self, report):
        bools = report.classification_counts.get("boolean", 0)
        assert bools >= 1

    def test_report_formats(self, report):
        text = format_universe_report(report)
        assert "UNIVERSE" in text
        assert "Jedis" in text
        assert "VIOLATIONS" in text

    def test_iterator_violations_detected(self, report):
        """BuilderFactory calls Iterator.next without hasNext — classic violation."""
        iter_violations = [e for e in report.conformance_edges
                           if e.callee_class == "Iterator" and e.violating > 0]
        assert len(iter_violations) >= 1

    def test_connection_used_transitively(self, report):
        """Multiple classes should use Connection transitively."""
        conn_edges = [e for e in report.conformance_edges
                      if e.callee_class == "Connection"]
        assert len(conn_edges) >= 3
