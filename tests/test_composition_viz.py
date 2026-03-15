"""Tests for composition visualization (Step 15 viz)."""

from __future__ import annotations

import os
import tempfile

import pytest

from reticulate.composition import (
    compose,
    synchronized_compose,
    compare_with_global,
    product_nary,
)
from reticulate.composition_viz import (
    composition_dashboard,
    level_dot,
    participant_dot,
    synchronized_dot,
)
from reticulate.duality import dual
from reticulate.parser import parse
from reticulate.statespace import build_statespace


def _parse(s: str):
    return parse(s)


# ======================================================================
# participant_dot()
# ======================================================================

class TestParticipantDot:
    """DOT generation for individual participant lattices."""

    def test_basic_output(self):
        ss = build_statespace(_parse("&{a: end, b: end}"))
        dot = participant_dot("Alice", ss)
        assert "digraph" in dot
        assert "Alice" in dot

    def test_color_index(self):
        ss = build_statespace(_parse("&{a: end}"))
        dot0 = participant_dot("A", ss, color_index=0)
        dot1 = participant_dot("B", ss, color_index=1)
        # Different colors
        assert "#2563eb" in dot0  # blue
        assert "#dc2626" in dot1  # red

    def test_top_bottom_markers(self):
        ss = build_statespace(_parse("&{a: end}"))
        dot = participant_dot("X", ss)
        assert "\u22a4" in dot  # ⊤
        assert "\u22a5" in dot  # ⊥

    def test_edge_labels_toggle(self):
        ss = build_statespace(_parse("&{a: end, b: end}"))
        dot_with = participant_dot("A", ss, edge_labels=True)
        dot_without = participant_dot("A", ss, edge_labels=False)
        assert '"a"' in dot_with
        assert '"a"' not in dot_without

    def test_selection_dashed(self):
        ss = build_statespace(_parse("+{x: end, y: end}"))
        dot = participant_dot("A", ss)
        assert "dashed" in dot

    def test_wrap_around_colors(self):
        """More participants than colors should wrap around."""
        ss = build_statespace(_parse("&{a: end}"))
        dot = participant_dot("Z", ss, color_index=100)
        assert "digraph" in dot


# ======================================================================
# synchronized_dot()
# ======================================================================

class TestSynchronizedDot:
    """DOT for synchronized product with shared edges highlighted."""

    def test_basic(self):
        ss1 = build_statespace(_parse("&{a: end, m: end}"))
        ss2 = build_statespace(_parse("&{b: end, m: end}"))
        from reticulate.composition import synchronized_product
        synced = synchronized_product(ss1, ss2)
        dot = synchronized_dot(synced, {"m"})
        assert "digraph" in dot
        assert "Synchronized" in dot

    def test_shared_edges_bold(self):
        ss1 = build_statespace(_parse("&{m: end}"))
        ss2 = build_statespace(_parse("&{m: end}"))
        from reticulate.composition import synchronized_product
        synced = synchronized_product(ss1, ss2)
        dot = synchronized_dot(synced, {"m"})
        assert 'penwidth="2.5"' in dot

    def test_no_shared_no_bold(self):
        ss1 = build_statespace(_parse("&{a: end}"))
        ss2 = build_statespace(_parse("&{b: end}"))
        from reticulate.composition import synchronized_product
        synced = synchronized_product(ss1, ss2)
        dot = synchronized_dot(synced, set())
        assert 'penwidth="2.5"' not in dot

    def test_custom_title(self):
        ss = build_statespace(_parse("&{a: end}"))
        dot = synchronized_dot(ss, set(), title="My Product")
        assert "My Product" in dot


# ======================================================================
# level_dot()
# ======================================================================

class TestLevelDot:
    """DOT for state spaces colored by composition level."""

    def test_global_level(self):
        ss = build_statespace(_parse("&{a: end}"))
        dot = level_dot(ss, "global", title="Global")
        assert "#1e40af" in dot  # global color

    def test_synchronized_level(self):
        ss = build_statespace(_parse("&{a: end}"))
        dot = level_dot(ss, "synchronized")
        assert "#7c3aed" in dot  # violet

    def test_free_level(self):
        ss = build_statespace(_parse("&{a: end}"))
        dot = level_dot(ss, "free")
        assert "#64748b" in dot  # slate

    def test_unknown_level_defaults_to_free(self):
        ss = build_statespace(_parse("&{a: end}"))
        dot = level_dot(ss, "unknown")
        assert "#64748b" in dot

    def test_edge_labels_toggle(self):
        ss = build_statespace(_parse("&{a: end}"))
        dot_with = level_dot(ss, "global", edge_labels=True)
        dot_without = level_dot(ss, "global", edge_labels=False)
        assert '"a"' in dot_with
        assert '"a"' not in dot_without


# ======================================================================
# composition_dashboard() — HTML rendering
# ======================================================================

class TestCompositionDashboard:
    """Dashboard HTML generation."""

    @pytest.fixture
    def simple_synced(self):
        server = _parse("&{req: end, ping: end}")
        client = dual(server)
        return synchronized_compose(("Server", server), ("Client", client))

    def test_renders_html(self, simple_synced, tmp_path):
        path = str(tmp_path / "dashboard.html")
        result = composition_dashboard(simple_synced, path)
        assert os.path.exists(result)
        with open(result) as f:
            html = f.read()
        assert "<!DOCTYPE html>" in html
        assert "Server" in html
        assert "Client" in html

    def test_contains_stats(self, simple_synced, tmp_path):
        path = str(tmp_path / "dashboard.html")
        composition_dashboard(simple_synced, path)
        with open(path) as f:
            html = f.read()
        assert "Participants" in html
        assert "Synced States" in html
        assert "Free States" in html
        assert "Reduction" in html

    def test_contains_compatibility(self, simple_synced, tmp_path):
        path = str(tmp_path / "dashboard.html")
        composition_dashboard(simple_synced, path)
        with open(path) as f:
            html = f.read()
        assert "Compatibility" in html
        assert "\u2713" in html or "\u2717" in html

    def test_contains_shared_labels(self, simple_synced, tmp_path):
        path = str(tmp_path / "dashboard.html")
        composition_dashboard(simple_synced, path)
        with open(path) as f:
            html = f.read()
        assert "Shared Labels" in html

    def test_contains_hierarchy(self, simple_synced, tmp_path):
        path = str(tmp_path / "dashboard.html")
        composition_dashboard(simple_synced, path)
        with open(path) as f:
            html = f.read()
        assert "Composition Hierarchy" in html
        assert "\u2286" in html  # ⊆ arrows

    def test_contains_svgs(self, simple_synced, tmp_path):
        path = str(tmp_path / "dashboard.html")
        composition_dashboard(simple_synced, path)
        with open(path) as f:
            html = f.read()
        assert "<svg" in html

    def test_custom_title(self, simple_synced, tmp_path):
        path = str(tmp_path / "dashboard.html")
        composition_dashboard(simple_synced, path, title="My Protocol")
        with open(path) as f:
            html = f.read()
        assert "My Protocol" in html

    def test_with_global_ss(self, tmp_path):
        """Dashboard with optional global state space."""
        g = "Client -> Server : {request: Server -> Client : {response: end}}"
        from reticulate.global_types import build_global_statespace, parse_global
        from reticulate.projection import project_all
        gt = parse_global(g)
        projections = project_all(gt)
        global_ss = build_global_statespace(gt)

        synced = synchronized_compose(
            *[(role, stype) for role, stype in projections.items()]
        )
        path = str(tmp_path / "dashboard.html")
        composition_dashboard(synced, path, global_ss=global_ss,
                              title="Request-Response")
        with open(path) as f:
            html = f.read()
        assert "Global" in html
        assert "Choreography" in html

    def test_three_party(self, tmp_path):
        s1 = _parse("&{a: end, m: end}")
        s2 = _parse("&{b: end, m: end}")
        s3 = _parse("&{c: end}")
        synced = synchronized_compose(("A", s1), ("B", s2), ("C", s3))
        path = str(tmp_path / "dashboard.html")
        composition_dashboard(synced, path)
        with open(path) as f:
            html = f.read()
        assert "A" in html
        assert "B" in html
        assert "C" in html

    def test_multiparty_benchmark(self, tmp_path):
        """Dashboard for a real benchmark: Delegation protocol."""
        g = (
            "Client -> Master : {task: "
            "Master -> Worker : {delegate: "
            "Worker -> Master : {result: "
            "Master -> Client : {response: end}}}}"
        )
        from reticulate.global_types import build_global_statespace, parse_global
        from reticulate.projection import project_all
        gt = parse_global(g)
        projections = project_all(gt)
        global_ss = build_global_statespace(gt)

        synced = synchronized_compose(
            *[(role, stype) for role, stype in projections.items()]
        )
        path = str(tmp_path / "delegation.html")
        result = composition_dashboard(synced, path, global_ss=global_ss,
                                       title="Delegation Protocol")
        assert os.path.exists(result)
        with open(result) as f:
            html = f.read()
        assert "Client" in html
        assert "Master" in html
        assert "Worker" in html
        # Should show reduction
        assert "Reduction" in html
