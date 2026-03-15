"""Tests for role-annotated Hasse diagrams (multiparty visualization)."""

import pytest

from reticulate.global_types import (
    build_global_statespace,
    parse_global,
)
from reticulate.visualize import (
    _extract_roles_from_transitions,
    _parse_role_label,
    role_dot_source,
)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------


class TestParseRoleLabel:
    """Test parsing of role-annotated transition labels."""

    def test_simple_label(self):
        sender, receiver, method = _parse_role_label("A->B:m")
        assert sender == "A"
        assert receiver == "B"
        assert method == "m"

    def test_non_role_label(self):
        sender, receiver, method = _parse_role_label("m")
        assert sender is None
        assert receiver is None
        assert method == "m"

    def test_complex_roles(self):
        sender, receiver, method = _parse_role_label("Buyer1->Seller:lookup")
        assert sender == "Buyer1"
        assert receiver == "Seller"
        assert method == "lookup"


# ---------------------------------------------------------------------------
# Role extraction
# ---------------------------------------------------------------------------


class TestExtractRoles:
    """Test role extraction from state-space transitions."""

    def test_two_roles(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        colors, roles = _extract_roles_from_transitions(ss)
        assert roles == {"A", "B"}
        assert len(colors) == 2

    def test_three_roles(self):
        g = parse_global("A -> B : {m: B -> C : {n: end}}")
        ss = build_global_statespace(g)
        colors, roles = _extract_roles_from_transitions(ss)
        assert roles == {"A", "B", "C"}
        assert len(colors) == 3

    def test_distinct_colors(self):
        g = parse_global("A -> B : {m: B -> C : {n: end}}")
        ss = build_global_statespace(g)
        colors, _ = _extract_roles_from_transitions(ss)
        color_values = list(colors.values())
        assert len(set(color_values)) == len(color_values)  # all distinct


# ---------------------------------------------------------------------------
# DOT generation
# ---------------------------------------------------------------------------


class TestRoleDotSource:
    """Test role-annotated DOT generation."""

    def test_produces_valid_dot(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        dot = role_dot_source(ss)
        assert dot.startswith("digraph {")
        assert dot.endswith("}")

    def test_contains_role_arrows(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        dot = role_dot_source(ss)
        assert "A\u2192B:m" in dot  # Unicode arrow in label

    def test_contains_legend(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        dot = role_dot_source(ss, show_legend=True)
        assert "cluster_legend" in dot
        assert "Roles" in dot

    def test_no_legend(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        dot = role_dot_source(ss, show_legend=False)
        assert "cluster_legend" not in dot

    def test_title(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        dot = role_dot_source(ss, title="Test Protocol")
        assert "Test Protocol" in dot

    def test_no_edge_labels(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        dot = role_dot_source(ss, edge_labels=False)
        assert "A\u2192B:m" not in dot

    def test_custom_colors(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        custom = {"A": "#ff0000", "B": "#00ff00"}
        dot = role_dot_source(ss, role_colors=custom)
        assert "#ff0000" in dot

    def test_three_roles_all_colored(self):
        g = parse_global(
            "A -> B : {m: B -> C : {n: C -> A : {reply: end}}}"
        )
        ss = build_global_statespace(g)
        dot = role_dot_source(ss)
        colors, roles = _extract_roles_from_transitions(ss)
        for role in roles:
            assert colors[role] in dot

    def test_two_buyer(self):
        g = parse_global(
            "Buyer1 -> Seller : {lookup: "
            "Seller -> Buyer1 : {price: "
            "Buyer1 -> Buyer2 : {share: "
            "Buyer2 -> Seller : {accept: end, reject: end}}}}"
        )
        ss = build_global_statespace(g)
        dot = role_dot_source(ss, title="Two-Buyer Protocol")
        assert "Two-Buyer Protocol" in dot
        assert "Buyer1" in dot
        assert "Buyer2" in dot
        assert "Seller" in dot

    def test_recursive_global(self):
        g = parse_global("rec X . A -> B : {m: X, done: end}")
        ss = build_global_statespace(g)
        dot = role_dot_source(ss)
        assert "digraph {" in dot

    def test_top_bottom_colored(self):
        g = parse_global("A -> B : {m: end}")
        ss = build_global_statespace(g)
        dot = role_dot_source(ss)
        assert "#bfdbfe" in dot  # top color
        assert "#bbf7d0" in dot  # bottom color


# ---------------------------------------------------------------------------
# Benchmark visualization
# ---------------------------------------------------------------------------


class TestRoleHasseBenchmarks:
    """Test role-annotated Hasse on all multiparty benchmarks."""

    @pytest.fixture
    def benchmarks(self):
        from tests.benchmarks.multiparty_protocols import MULTIPARTY_BENCHMARKS
        return MULTIPARTY_BENCHMARKS

    def test_all_produce_valid_dot(self, benchmarks):
        for b in benchmarks:
            g = parse_global(b.global_type_string)
            ss = build_global_statespace(g)
            dot = role_dot_source(ss, title=b.name)
            assert dot.startswith("digraph {"), f"{b.name}: invalid DOT"
            assert "cluster_legend" in dot, f"{b.name}: no legend"

    def test_all_roles_in_legend(self, benchmarks):
        for b in benchmarks:
            g = parse_global(b.global_type_string)
            ss = build_global_statespace(g)
            dot = role_dot_source(ss)
            for role in sorted(b.expected_roles):
                assert role in dot, f"{b.name}: role {role} not in DOT"
