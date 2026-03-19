"""Tests for the visual board HTML generator."""

import json
import re

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.game.visual_board import (
    generate_board_html,
    open_visual_board,
    _compute_ranks,
    _layout,
    _classify_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_str: str):
    return build_statespace(parse(type_str))


def _extract_game_data(html: str) -> dict:
    """Extract the embedded GAME_DATA JSON from the HTML."""
    match = re.search(r"const GAME_DATA = ({.*?});\s*\n", html, re.DOTALL)
    assert match, "GAME_DATA not found in HTML"
    return json.loads(match.group(1))


# ---------------------------------------------------------------------------
# _compute_ranks
# ---------------------------------------------------------------------------

class TestComputeRanks:
    def test_simple_branch(self):
        ss = _ss("&{a: end, b: end}")
        ranks = _compute_ranks(ss)
        assert ranks[ss.top] == 0
        assert ranks[ss.bottom] == 1

    def test_chain(self):
        ss = _ss("&{a: &{b: end}}")
        ranks = _compute_ranks(ss)
        assert ranks[ss.top] == 0
        assert ranks[ss.bottom] == 2

    def test_selection(self):
        ss = _ss("&{a: +{OK: end, ERR: end}}")
        ranks = _compute_ranks(ss)
        assert ranks[ss.top] == 0
        # selection node at rank 1
        select_id = [s for s in ss.states if s != ss.top and s != ss.bottom][0]
        assert ranks[select_id] == 1
        assert ranks[ss.bottom] == 2


# ---------------------------------------------------------------------------
# _layout
# ---------------------------------------------------------------------------

class TestLayout:
    def test_positions_assigned(self):
        ss = _ss("&{a: end, b: end}")
        positions = _layout(ss)
        assert len(positions) == len(ss.states)
        for sid in ss.states:
            assert sid in positions
            x, y = positions[sid]
            assert x > 0
            assert y > 0

    def test_top_above_bottom(self):
        ss = _ss("&{a: end}")
        positions = _layout(ss)
        assert positions[ss.top][1] < positions[ss.bottom][1]

    def test_same_rank_spread(self):
        """Nodes at the same rank should have different x positions."""
        ss = _ss("&{a: &{c: end}, b: &{d: end}}")
        positions = _layout(ss)
        rank1_nodes = [
            sid for sid in ss.states
            if positions[sid][1] == positions[ss.top][1] + 120
        ]
        if len(rank1_nodes) > 1:
            xs = [positions[s][0] for s in rank1_nodes]
            assert len(set(xs)) == len(xs), "Same-rank nodes should have distinct x"


# ---------------------------------------------------------------------------
# _classify_node
# ---------------------------------------------------------------------------

class TestClassifyNode:
    def test_top(self):
        ss = _ss("&{a: end}")
        assert _classify_node(ss, ss.top) == "top"

    def test_end(self):
        ss = _ss("&{a: end}")
        assert _classify_node(ss, ss.bottom) == "end"

    def test_branch(self):
        ss = _ss("&{a: &{b: end}}")
        mid = [s for s in ss.states if s != ss.top and s != ss.bottom][0]
        assert _classify_node(ss, mid) == "branch"

    def test_select(self):
        ss = _ss("&{a: +{OK: end, ERR: end}}")
        mid = [s for s in ss.states if s != ss.top and s != ss.bottom][0]
        assert _classify_node(ss, mid) == "select"


# ---------------------------------------------------------------------------
# generate_board_html
# ---------------------------------------------------------------------------

class TestGenerateBoardHtml:
    def test_returns_html(self):
        ss = _ss("&{a: end}")
        html = generate_board_html(ss)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_svg(self):
        ss = _ss("&{a: end}")
        html = generate_board_html(ss)
        assert '<svg class="board"' in html

    def test_contains_game_data(self):
        ss = _ss("&{a: end}")
        html = generate_board_html(ss)
        assert "const GAME_DATA" in html

    def test_game_data_nodes(self):
        ss = _ss("&{a: end, b: end}")
        html = generate_board_html(ss)
        data = _extract_game_data(html)
        assert len(data["nodes"]) == len(ss.states)

    def test_game_data_edges(self):
        ss = _ss("&{a: end, b: end}")
        html = generate_board_html(ss)
        data = _extract_game_data(html)
        assert len(data["edges"]) == len(ss.transitions)

    def test_game_data_top_bottom(self):
        ss = _ss("&{a: end}")
        html = generate_board_html(ss)
        data = _extract_game_data(html)
        assert data["top"] == ss.top
        assert data["bottom"] == ss.bottom

    def test_node_kinds(self):
        ss = _ss("&{a: +{OK: end, ERR: end}}")
        html = generate_board_html(ss)
        data = _extract_game_data(html)
        kinds = {n["kind"] for n in data["nodes"]}
        assert "top" in kinds
        assert "end" in kinds
        assert "select" in kinds

    def test_selection_edges_marked(self):
        ss = _ss("&{a: +{OK: end, ERR: end}}")
        html = generate_board_html(ss)
        data = _extract_game_data(html)
        selection_edges = [e for e in data["edges"] if e["is_selection"]]
        assert len(selection_edges) >= 2

    def test_title(self):
        ss = _ss("&{a: end}")
        html = generate_board_html(ss, title="My Duel")
        assert "<title>My Duel</title>" in html

    def test_mode_embedded(self):
        ss = _ss("&{a: end}")
        html = generate_board_html(ss, mode="adversarial")
        data = _extract_game_data(html)
        assert data["mode"] == "adversarial"

    def test_human_role_embedded(self):
        ss = _ss("&{a: end}")
        html = generate_board_html(ss, human_role="client")
        data = _extract_game_data(html)
        assert data["human_role"] == "client"

    def test_stats_in_data(self):
        ss = _ss("&{a: end, b: end}")
        html = generate_board_html(ss)
        data = _extract_game_data(html)
        assert data["n_states"] == len(ss.states)
        assert data["n_transitions"] == len(ss.transitions)

    def test_node_positions(self):
        ss = _ss("&{a: end}")
        html = generate_board_html(ss)
        data = _extract_game_data(html)
        for node in data["nodes"]:
            assert "x" in node
            assert "y" in node
            assert node["x"] > 0
            assert node["y"] > 0

    def test_larger_protocol(self):
        """ATM-like protocol generates valid HTML."""
        ss = _ss("&{auth: +{OK: &{balance: end, withdraw: +{OK: end, DENIED: end}}, FAIL: end}}")
        html = generate_board_html(ss)
        data = _extract_game_data(html)
        assert len(data["nodes"]) == len(ss.states)
        assert len(data["edges"]) == len(ss.transitions)

    def test_recursive_type(self):
        ss = _ss("rec X . &{next: +{val: X, done: end}}")
        html = generate_board_html(ss)
        data = _extract_game_data(html)
        assert len(data["nodes"]) > 0
        assert len(data["edges"]) > 0

    def test_javascript_functions(self):
        ss = _ss("&{a: end}")
        html = generate_board_html(ss)
        assert "function makeMove" in html
        assert "function renderBoard" in html
        assert "function updatePanel" in html
        assert "function restartGame" in html
        assert "function bfsDistance" in html
        assert "function aiMove" in html


# ---------------------------------------------------------------------------
# open_visual_board
# ---------------------------------------------------------------------------

class TestOpenVisualBoard:
    def test_output_path(self, tmp_path):
        ss = _ss("&{a: end}")
        out = str(tmp_path / "board.html")
        # Monkeypatch webbrowser.open to avoid actually opening
        import reticulate.game.visual_board as vb
        opened = []
        original = vb.webbrowser.open
        vb.webbrowser.open = lambda url: opened.append(url)
        try:
            result = open_visual_board(ss, output_path=out)
            assert result == out
            assert (tmp_path / "board.html").exists()
            content = (tmp_path / "board.html").read_text()
            assert "<!DOCTYPE html>" in content
            assert len(opened) == 1
        finally:
            vb.webbrowser.open = original

    def test_temp_path(self):
        ss = _ss("&{a: end}")
        import reticulate.game.visual_board as vb
        opened = []
        original = vb.webbrowser.open
        vb.webbrowser.open = lambda url: opened.append(url)
        try:
            result = open_visual_board(ss)
            assert result.endswith(".html")
            import os
            assert os.path.exists(result)
            os.unlink(result)
        finally:
            vb.webbrowser.open = original
