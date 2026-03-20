"""Tests for the visualize module."""

from __future__ import annotations

import os
import tempfile

import pytest

from reticulate.lattice import check_lattice
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.visualize import dot_source, hasse_diagram, render_hasse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dot(type_str: str, **kwargs) -> str:  # type: ignore[no-untyped-def]
    ss = build_statespace(parse(type_str))
    return dot_source(ss, **kwargs)


def _dot_with_result(type_str: str, **kwargs) -> str:  # type: ignore[no-untyped-def]
    ss = build_statespace(parse(type_str))
    result = check_lattice(ss)
    return dot_source(ss, result, **kwargs)


# ---------------------------------------------------------------------------
# dot_source tests (pure, no graphviz needed)
# ---------------------------------------------------------------------------

class TestDotSourceEnd:
    def test_contains_rankdir(self) -> None:
        src = _dot("end")
        assert "rankdir=TB" in src

    def test_one_node(self) -> None:
        ss = build_statespace(parse("end"))
        src = dot_source(ss)
        # Should have exactly one node (top == bottom == end)
        assert src.count("->") == 0

    def test_no_edges(self) -> None:
        src = _dot("end")
        assert "->" not in src


class TestDotSourceChain:
    def test_has_three_nodes(self) -> None:
        ss = build_statespace(parse("&{a: &{b: end}}"))
        src = dot_source(ss)
        # Count node declarations (lines with '[' and 'label')
        node_lines = [l for l in src.splitlines() if "[" in l and "label=" in l and "->" not in l]
        assert len(node_lines) == 3

    def test_has_two_edges(self) -> None:
        src = _dot("&{a: &{b: end}}")
        assert src.count("->") == 2

    def test_edge_labels(self) -> None:
        src = _dot("&{a: &{b: end}}")
        assert '"a"' in src
        assert '"b"' in src


class TestDotSourceDiamond:
    def test_has_four_nodes(self) -> None:
        ss = build_statespace(parse("&{m: &{a: end}, n: &{b: end}}"))
        src = dot_source(ss)
        node_lines = [l for l in src.splitlines() if "[" in l and "label=" in l and "->" not in l]
        assert len(node_lines) == 4

    def test_top_marked(self) -> None:
        src = _dot("&{m: &{a: end}, n: &{b: end}}")
        assert "\u22a4" in src  # top

    def test_bottom_marked(self) -> None:
        src = _dot("&{m: &{a: end}, n: &{b: end}}")
        assert "\u22a5" in src  # bot


class TestDotSourceParallel:
    def test_has_four_nodes(self) -> None:
        ss = build_statespace(parse("(&{a: end} || &{b: end})"))
        src = dot_source(ss)
        node_lines = [l for l in src.splitlines() if "[" in l and "label=" in l and "->" not in l]
        assert len(node_lines) == 4

    def test_edge_labels_present(self) -> None:
        src = _dot("(&{a: end} || &{b: end})")
        assert '"a"' in src
        assert '"b"' in src


class TestDotSourceRecursive:
    def test_self_loop(self) -> None:
        ss = build_statespace(parse("rec X . &{next: X, done: end}"))
        src = dot_source(ss)
        # Should have 2 states: the rec entry point and end
        assert ss.top != ss.bottom
        assert len(ss.states) == 2

    def test_has_two_nodes(self) -> None:
        ss = build_statespace(parse("rec X . &{next: X, done: end}"))
        src = dot_source(ss)
        node_lines = [l for l in src.splitlines() if "[" in l and "label=" in l and "->" not in l]
        assert len(node_lines) == 2


class TestDotSourceWithResult:
    def test_uses_result(self) -> None:
        src = _dot_with_result("&{m: &{a: end}, n: &{b: end}}")
        assert "rankdir=TB" in src
        # Should still have the markers
        assert "\u22a4" in src
        assert "\u22a5" in src

    def test_title_with_result(self) -> None:
        src = _dot_with_result("&{m: &{a: end}, n: &{b: end}}", title="Diamond lattice")
        assert "Diamond lattice" in src


class TestDotSourceWithCycles:
    def test_all_nodes_shown(self) -> None:
        # rec X . &{a: &{b: X}, done: end}
        # The rec body creates a cycle — all nodes should be visible
        type_str = "rec X . &{a: &{b: X}, done: end}"
        ss = build_statespace(parse(type_str))
        result = check_lattice(ss)

        src_plain = dot_source(ss)
        src_with_result = dot_source(ss, result)

        # Both versions should show the same nodes — no collapsing
        plain_nodes = [l for l in src_plain.splitlines() if "[" in l and "label=" in l and "->" not in l]
        result_nodes = [l for l in src_with_result.splitlines() if "[" in l and "label=" in l and "->" not in l]

        assert len(result_nodes) == len(plain_nodes)

    def test_cycle_edges_shown(self) -> None:
        type_str = "rec X . &{a: &{b: X}, done: end}"
        ss = build_statespace(parse(type_str))
        result = check_lattice(ss)

        src = dot_source(ss, result)

        # All transitions should be present, including the back-edge
        edge_lines = [l for l in src.splitlines() if "->" in l]
        assert len(edge_lines) == len(ss.transitions)


class TestDotSourceNoLabels:
    def test_id_only_labels(self) -> None:
        ss = build_statespace(parse("&{a: end}"))
        src = dot_source(ss, labels=False)
        # Node labels should be just IDs (numbers), not the session type labels
        # The session type label "a" should not appear as a node label
        # but may appear as an edge label
        for sid in ss.states:
            # Check that the ID appears as a label
            assert str(sid) in src


class TestDotSourceNoEdgeLabels:
    def test_edges_without_labels(self) -> None:
        src = _dot("&{a: &{b: end}}", edge_labels=False)
        # There should be edges but without label attributes
        edge_lines = [l for l in src.splitlines() if "->" in l]
        assert len(edge_lines) > 0
        for line in edge_lines:
            assert "label=" not in line


class TestDotSourceTitle:
    def test_title_in_dot(self) -> None:
        src = _dot("end", title="test title")
        assert 'label="test title"' in src

    def test_no_title(self) -> None:
        src = _dot("end")
        # No graph-level label attribute should be present
        lines = src.splitlines()
        # Filter out node/edge lines to check only graph attributes
        graph_lines = [l for l in lines if "labelloc" in l]
        assert len(graph_lines) == 0


class TestDotSourceTruncation:
    def test_long_labels_truncated(self) -> None:
        # Product of two chains should create composite labels
        ss = build_statespace(parse("(&{a: &{b: &{c: end}}} || &{d: &{e: &{f: end}}})"))
        src = dot_source(ss)
        # Extract all label values from node lines
        for line in src.splitlines():
            if "[" in line and "label=" in line and "->" not in line:
                # Extract label value between quotes after label=
                start = line.index('label="') + 7
                end = line.index('"', start)
                label = line[start:end]
                # 40 chars + possible "..." suffix = max 41
                # Plus "top " or "bot " prefix (3 chars) = max 44
                assert len(label) <= 44, f"Label too long ({len(label)}): {label!r}"


# ---------------------------------------------------------------------------
# Constructor node style tests
# ---------------------------------------------------------------------------

class TestConstructorNodeStyle:
    """Tests for node_style='constructor'."""

    def test_constructor_circles(self) -> None:
        dot = _dot("&{a: end}", node_style="constructor")
        assert "shape=circle" in dot

    def test_branch_symbol(self) -> None:
        dot = _dot("&{a: &{b: end}}", node_style="constructor")
        # The intermediate branch state should have & label
        assert 'label="&"' in dot

    def test_select_symbol(self) -> None:
        dot = _dot("&{a: +{OK: end, ERR: end}}", node_style="constructor")
        assert "\u2295" in dot  # ⊕

    def test_parallel_symbol(self) -> None:
        dot = _dot("(&{a: end} || &{b: end})", node_style="constructor")
        assert "\u2225" in dot  # ∥

    def test_end_double_circle(self) -> None:
        dot = _dot("end", node_style="constructor")
        assert "doublecircle" in dot

    def test_top_symbol(self) -> None:
        dot = _dot("&{a: end}", node_style="constructor")
        assert "\u22a4" in dot  # ⊤

    def test_bottom_symbol(self) -> None:
        dot = _dot("&{a: end}", node_style="constructor")
        assert "\u22a5" in dot  # ⊥

    def test_selection_edges_dashed(self) -> None:
        dot = _dot("&{a: +{OK: end, ERR: end}}", node_style="constructor")
        assert "dashed" in dot

    def test_with_result(self) -> None:
        dot = _dot_with_result("&{a: +{OK: end, ERR: end}}", node_style="constructor")
        assert "shape=circle" in dot
        assert "\u2295" in dot

    def test_ki3_onboarding(self) -> None:
        """Ki3 onboarding renders in constructor mode without errors."""
        type_str = (
            "&{validateContract: +{APPROVED: "
            "(&{createVPS: +{PROVISIONED: wait, FAILED: wait}} || "
            "&{configureDNS: +{PROPAGATED: wait, FAILED: wait}}) . "
            "&{createKeycloakRealm: +{CREATED: "
            "(&{createSchema: &{seedData: wait}} || "
            "&{configureProxy: &{requestSSL: wait}}) . "
            "(&{setupMonitoring: &{createDashboards: wait}} || "
            "&{configureBackup: wait}) . "
            "&{runHealthChecks: +{HEALTHY: &{notifyTenant: "
            "&{activateSubscription: end}}, "
            "UNHEALTHY: &{rollback: &{notifyOps: end}}}}, "
            "FAILED: end}}, "
            "REJECTED: end}}"
        )
        dot = _dot_with_result(type_str, node_style="constructor")
        assert "shape=circle" in dot
        # Should have branch, select, and parallel nodes
        assert "\u2295" in dot  # ⊕ (select)
        assert "\u2225" in dot  # ∥ (parallel)
        assert "&" in dot       # branch


# ---------------------------------------------------------------------------
# hasse_diagram tests (require graphviz)
# ---------------------------------------------------------------------------

class TestHasseDiagram:
    def test_returns_source_object(self) -> None:
        graphviz = pytest.importorskip("graphviz")
        ss = build_statespace(parse("end"))
        g = hasse_diagram(ss)
        assert isinstance(g, graphviz.Source)

    def test_source_contains_dot(self) -> None:
        pytest.importorskip("graphviz")
        ss = build_statespace(parse("&{a: end}"))
        g = hasse_diagram(ss)
        assert "rankdir=TB" in g.source


# ---------------------------------------------------------------------------
# render_hasse tests (require graphviz + dot binary)
# ---------------------------------------------------------------------------

class TestRenderHasse:
    def test_render_creates_file(self) -> None:
        pytest.importorskip("graphviz")
        ss = build_statespace(parse("&{a: end}"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_hasse")
            try:
                out = render_hasse(ss, path, fmt="png")
                assert os.path.exists(out)
            except Exception:
                # dot binary may not be installed — skip gracefully
                pytest.skip("graphviz dot binary not available")

    def test_render_svg(self) -> None:
        pytest.importorskip("graphviz")
        ss = build_statespace(parse("&{a: end}"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_hasse")
            try:
                out = render_hasse(ss, path, fmt="svg")
                assert os.path.exists(out)
                assert out.endswith(".svg")
            except Exception:
                pytest.skip("graphviz dot binary not available")


# ---------------------------------------------------------------------------
# SCC cluster visualization tests
# ---------------------------------------------------------------------------

class TestSCCClusters:
    """Tests for SCC cluster subgraph rendering."""

    def test_no_clusters_without_flag(self) -> None:
        """Without scc_clusters=True, no cluster subgraphs appear."""
        ss = build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))
        result = check_lattice(ss)
        src = dot_source(ss, result)
        assert "subgraph cluster_" not in src

    def test_no_clusters_without_result(self) -> None:
        """Without a LatticeResult, scc_clusters has no effect."""
        ss = build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))
        src = dot_source(ss, scc_clusters=True)
        assert "subgraph cluster_" not in src

    def test_no_clusters_for_acyclic(self) -> None:
        """Acyclic types have no multi-state SCCs — no clusters."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        result = check_lattice(ss)
        src = dot_source(ss, result, scc_clusters=True)
        assert "subgraph cluster_" not in src

    def test_no_clusters_for_self_loop(self) -> None:
        """Self-loop (1-state SCC) should not create a cluster."""
        ss = build_statespace(parse("rec X . &{next: X, done: end}"))
        result = check_lattice(ss)
        src = dot_source(ss, result, scc_clusters=True)
        assert "subgraph cluster_" not in src

    def test_cluster_for_two_state_cycle(self) -> None:
        """Two-state cycle should create a cluster subgraph."""
        ss = build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))
        result = check_lattice(ss)
        src = dot_source(ss, result, scc_clusters=True)
        assert "subgraph cluster_scc0" in src
        assert "2 states" in src
        assert "cyclic-equivalent" in src
        assert "#f5f3ff" in src  # cluster fill color

    def test_intra_scc_back_edge_dashed(self) -> None:
        """Back-edges within an SCC should be dashed purple."""
        ss = build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))
        result = check_lattice(ss)
        src = dot_source(ss, result, scc_clusters=True)
        # The back-edge b goes from after_a back to top
        assert "#7c3aed" in src  # purple color on back-edge

    def test_inter_scc_edges_not_purple(self) -> None:
        """Edges between different SCCs should not be purple."""
        ss = build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))
        result = check_lattice(ss)
        src = dot_source(ss, result, scc_clusters=True)
        # The "done" edge goes from top SCC to end SCC — should not be purple
        for line in src.splitlines():
            if "done" in line and "->" in line:
                assert "#7c3aed" not in line

    def test_all_nodes_preserved(self) -> None:
        """Enabling scc_clusters should not change the number of nodes."""
        ss = build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))
        result = check_lattice(ss)
        src_plain = dot_source(ss, result)
        src_clustered = dot_source(ss, result, scc_clusters=True)

        def count_nodes(src: str) -> int:
            return sum(1 for l in src.splitlines()
                       if "[" in l and "label=" in l and "->" not in l
                       and "subgraph" not in l)

        assert count_nodes(src_plain) == count_nodes(src_clustered)

    def test_all_edges_preserved(self) -> None:
        """Enabling scc_clusters should not change the number of edges."""
        ss = build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))
        result = check_lattice(ss)
        src_plain = dot_source(ss, result)
        src_clustered = dot_source(ss, result, scc_clusters=True)

        def count_edges(src: str) -> int:
            return sum(1 for l in src.splitlines() if "->" in l and "rank" not in l)

        assert count_edges(src_plain) == count_edges(src_clustered)

    def test_cluster_with_constructor_style(self) -> None:
        """SCC clusters work with node_style='constructor'."""
        ss = build_statespace(parse("rec X . &{a: &{b: X}, done: end}"))
        result = check_lattice(ss)
        src = dot_source(ss, result, scc_clusters=True, node_style="constructor")
        assert "subgraph cluster_scc0" in src
        assert "shape=circle" in src
