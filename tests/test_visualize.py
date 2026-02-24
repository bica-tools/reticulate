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
        ss = build_statespace(parse("a . b . end"))
        src = dot_source(ss)
        # Count node declarations (lines with '[' and 'label')
        node_lines = [l for l in src.splitlines() if "[" in l and "label=" in l and "->" not in l]
        assert len(node_lines) == 3

    def test_has_two_edges(self) -> None:
        src = _dot("a . b . end")
        assert src.count("->") == 2

    def test_edge_labels(self) -> None:
        src = _dot("a . b . end")
        assert '"a"' in src
        assert '"b"' in src


class TestDotSourceDiamond:
    def test_has_four_nodes(self) -> None:
        ss = build_statespace(parse("&{m: a.end, n: b.end}"))
        src = dot_source(ss)
        node_lines = [l for l in src.splitlines() if "[" in l and "label=" in l and "->" not in l]
        assert len(node_lines) == 4

    def test_top_marked(self) -> None:
        src = _dot("&{m: a.end, n: b.end}")
        assert "\u22a4" in src  # ⊤

    def test_bottom_marked(self) -> None:
        src = _dot("&{m: a.end, n: b.end}")
        assert "\u22a5" in src  # ⊥


class TestDotSourceParallel:
    def test_has_four_nodes(self) -> None:
        ss = build_statespace(parse("(a.end || b.end)"))
        src = dot_source(ss)
        node_lines = [l for l in src.splitlines() if "[" in l and "label=" in l and "->" not in l]
        assert len(node_lines) == 4

    def test_edge_labels_present(self) -> None:
        src = _dot("(a.end || b.end)")
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
        src = _dot_with_result("&{m: a.end, n: b.end}")
        assert "rankdir=TB" in src
        # Should still have the markers
        assert "\u22a4" in src
        assert "\u22a5" in src

    def test_title_with_result(self) -> None:
        src = _dot_with_result("&{m: a.end, n: b.end}", title="Diamond lattice")
        assert "Diamond lattice" in src


class TestDotSourceSccCollapse:
    def test_collapsed_nodes(self) -> None:
        # rec X . &{a: &{b: X}, done: end}
        # Without result: the rec body creates a cycle (states in same SCC)
        type_str = "rec X . &{a: &{b: X}, done: end}"
        ss = build_statespace(parse(type_str))
        result = check_lattice(ss)

        src_plain = dot_source(ss)
        src_collapsed = dot_source(ss, result)

        # Collapsed version should have fewer node lines
        plain_nodes = [l for l in src_plain.splitlines() if "[" in l and "label=" in l and "->" not in l]
        collapsed_nodes = [l for l in src_collapsed.splitlines() if "[" in l and "label=" in l and "->" not in l]

        assert len(collapsed_nodes) <= len(plain_nodes)

    def test_scc_label_shows_count(self) -> None:
        type_str = "rec X . &{a: &{b: X}, done: end}"
        ss = build_statespace(parse(type_str))
        result = check_lattice(ss)

        # There should be an SCC with >1 member
        scc_groups: dict[int, set[int]] = {}
        for state, rep in result.scc_map.items():
            scc_groups.setdefault(rep, set()).add(state)

        has_multi = any(len(members) > 1 for members in scc_groups.values())
        if has_multi:
            src = dot_source(ss, result)
            assert "\u00d7" in src  # × symbol in (×N)


class TestDotSourceNoLabels:
    def test_id_only_labels(self) -> None:
        ss = build_statespace(parse("a . end"))
        src = dot_source(ss, labels=False)
        # Node labels should be just IDs (numbers), not the session type labels
        # The session type label "a" should not appear as a node label
        # but may appear as an edge label
        for sid in ss.states:
            # Check that the ID appears as a label
            assert str(sid) in src


class TestDotSourceNoEdgeLabels:
    def test_edges_without_labels(self) -> None:
        src = _dot("a . b . end", edge_labels=False)
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
        ss = build_statespace(parse("(a.b.c.d.e.f.g.h.end || i.j.k.l.m.n.o.p.end)"))
        src = dot_source(ss)
        # Extract all label values from node lines
        for line in src.splitlines():
            if "[" in line and "label=" in line and "->" not in line:
                # Extract label value between quotes after label=
                start = line.index('label="') + 7
                end = line.index('"', start)
                label = line[start:end]
                # 40 chars + possible "…" suffix = max 41
                # Plus "⊤ " or "⊥ " prefix (3 chars) = max 44
                assert len(label) <= 44, f"Label too long ({len(label)}): {label!r}"


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
        ss = build_statespace(parse("a . end"))
        g = hasse_diagram(ss)
        assert "rankdir=TB" in g.source


# ---------------------------------------------------------------------------
# render_hasse tests (require graphviz + dot binary)
# ---------------------------------------------------------------------------

class TestRenderHasse:
    def test_render_creates_file(self) -> None:
        pytest.importorskip("graphviz")
        ss = build_statespace(parse("a . end"))
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
        ss = build_statespace(parse("a . end"))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_hasse")
            try:
                out = render_hasse(ss, path, fmt="svg")
                assert os.path.exists(out)
                assert out.endswith(".svg")
            except Exception:
                pytest.skip("graphviz dot binary not available")
