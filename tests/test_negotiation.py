"""Tests for capability negotiation via lattice meet (Step 115a)."""

from __future__ import annotations

import pytest

from reticulate import parse, build_statespace, check_lattice, pretty
from reticulate.parser import Branch, End, Select
from reticulate.negotiation import (
    NegotiationResult,
    check_negotiation,
    compatibility_score,
    compatible,
    negotiate,
    negotiate_group,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_a():
    """Agent with capabilities: read, write, close."""
    return parse("&{read: end, write: end, close: end}")


@pytest.fixture
def agent_b():
    """Agent with capabilities: read, close, delete."""
    return parse("&{read: end, close: end, delete: end}")


@pytest.fixture
def agent_c():
    """Agent with capabilities: read only."""
    return parse("&{read: end}")


@pytest.fixture
def mcp_full():
    """Full MCP-like protocol."""
    return parse("&{initialize: +{OK: &{tools_list: +{TOOLS: &{tool_call: end}}}, ERROR: end}}")


@pytest.fixture
def mcp_minimal():
    """Minimal MCP-like protocol (no tool_call)."""
    return parse("&{initialize: +{OK: end, ERROR: end}}")


# ---------------------------------------------------------------------------
# negotiate tests
# ---------------------------------------------------------------------------


class TestNegotiate:
    def test_identical_types(self, agent_a):
        """Identical types negotiate to themselves."""
        result = negotiate(agent_a, agent_a)
        assert pretty(result) == pretty(agent_a)

    def test_shared_methods_only(self, agent_a, agent_b):
        """Negotiation keeps only shared methods: read, close."""
        result = negotiate(agent_a, agent_b)
        choices = dict(result.choices) if isinstance(result, (Branch, Select)) else {}
        assert "read" in choices
        assert "close" in choices
        assert "write" not in choices
        assert "delete" not in choices

    def test_no_shared_methods(self):
        """Completely incompatible agents → End."""
        a = parse("&{x: end}")
        b = parse("&{y: end}")
        result = negotiate(a, b)
        assert isinstance(result, End)

    def test_subtype_returns_subtype(self):
        """If A <: B, negotiation returns A (more specific)."""
        wide = parse("&{a: end, b: end, c: end}")
        narrow = parse("&{a: end, b: end}")
        result = negotiate(wide, narrow)
        # Wide is subtype of narrow (more methods in Branch = subtype)
        assert pretty(result) == pretty(wide)

    def test_end_with_anything(self, agent_a):
        """End negotiated with anything = End."""
        assert isinstance(negotiate(End(), agent_a), End)
        assert isinstance(negotiate(agent_a, End()), End)

    def test_recursive_negotiation(self):
        """Negotiate types with depth > 1."""
        a = parse("&{init: +{OK: &{run: end, stop: end}, FAIL: end}}")
        b = parse("&{init: +{OK: &{run: end, pause: end}, FAIL: end}}")
        result = negotiate(a, b)
        # Should share: init → OK → run, FAIL
        assert isinstance(result, Branch)

    def test_negotiated_type_is_lattice(self, agent_a, agent_b):
        """The negotiated type should form a valid lattice."""
        result = negotiate(agent_a, agent_b)
        if not isinstance(result, End):
            ss = build_statespace(result)
            lr = check_lattice(ss)
            assert lr.is_lattice is True

    def test_select_negotiation(self):
        """Negotiate two Select types."""
        a = parse("+{x: end, y: end}")
        b = parse("+{y: end, z: end}")
        result = negotiate(a, b)
        assert isinstance(result, Select)
        choices = dict(result.choices)
        assert "y" in choices
        assert "x" not in choices
        assert "z" not in choices


# ---------------------------------------------------------------------------
# compatible tests
# ---------------------------------------------------------------------------


class TestCompatible:
    def test_identical_compatible(self, agent_a):
        assert compatible(agent_a, agent_a) is True

    def test_overlapping_compatible(self, agent_a, agent_b):
        assert compatible(agent_a, agent_b) is True

    def test_disjoint_incompatible(self):
        a = parse("&{x: end}")
        b = parse("&{y: end}")
        assert compatible(a, b) is False

    def test_end_end_compatible(self):
        assert compatible(End(), End()) is True

    def test_end_nonend_incompatible(self, agent_a):
        assert compatible(End(), agent_a) is False

    def test_subset_compatible(self, agent_a, agent_c):
        assert compatible(agent_a, agent_c) is True


# ---------------------------------------------------------------------------
# compatibility_score tests
# ---------------------------------------------------------------------------


class TestScore:
    def test_identical_score_1(self, agent_a):
        assert compatibility_score(agent_a, agent_a) == 1.0

    def test_disjoint_score_0(self):
        a = parse("&{x: end}")
        b = parse("&{y: end}")
        assert compatibility_score(a, b) == 0.0

    def test_partial_overlap(self, agent_a, agent_b):
        """a={read,write,close}, b={read,close,delete}. Shared=2, total=4. Score=0.5."""
        score = compatibility_score(agent_a, agent_b)
        assert score == 0.5

    def test_subset_score(self, agent_a, agent_c):
        """a={read,write,close}, c={read}. Shared=1, total=3. Score=0.333."""
        score = compatibility_score(agent_a, agent_c)
        assert score == 0.333

    def test_end_end_score_1(self):
        assert compatibility_score(End(), End()) == 1.0


# ---------------------------------------------------------------------------
# check_negotiation tests
# ---------------------------------------------------------------------------


class TestCheckNegotiation:
    def test_result_type(self, agent_a, agent_b):
        result = check_negotiation(agent_a, agent_b)
        assert isinstance(result, NegotiationResult)

    def test_shared_methods(self, agent_a, agent_b):
        result = check_negotiation(agent_a, agent_b)
        assert "read" in result.shared_methods
        assert "close" in result.shared_methods

    def test_dropped_methods(self, agent_a, agent_b):
        result = check_negotiation(agent_a, agent_b)
        assert "write" in result.dropped_methods
        assert "delete" in result.dropped_methods

    def test_score_matches(self, agent_a, agent_b):
        result = check_negotiation(agent_a, agent_b)
        assert result.score == 0.5


# ---------------------------------------------------------------------------
# negotiate_group tests
# ---------------------------------------------------------------------------


class TestNegotiateGroup:
    def test_empty_group(self):
        assert isinstance(negotiate_group([]), End)

    def test_single_agent(self, agent_a):
        result = negotiate_group([agent_a])
        assert pretty(result) == pretty(agent_a)

    def test_two_agents(self, agent_a, agent_b):
        result = negotiate_group([agent_a, agent_b])
        choices = dict(result.choices) if isinstance(result, (Branch, Select)) else {}
        assert "read" in choices
        assert "close" in choices

    def test_three_agents_narrows(self, agent_a, agent_b, agent_c):
        """Three agents: negotiation keeps shared methods via subtyping.

        a∧b = {read, close}. Then {read, close} <: {read} (Gay-Hole: wider branch = subtype),
        so negotiate({read,close}, {read}) returns {read,close} (the more specific type).
        """
        result = negotiate_group([agent_a, agent_b, agent_c])
        choices = dict(result.choices) if isinstance(result, (Branch, Select)) else {}
        assert "read" in choices

    def test_incompatible_in_group(self, agent_a):
        """One incompatible agent kills the group."""
        alien = parse("&{zz_alien: end}")
        result = negotiate_group([agent_a, alien])
        assert isinstance(result, End)

    def test_group_result_is_lattice(self, agent_a, agent_b, agent_c):
        """Group negotiation result should be a valid lattice."""
        result = negotiate_group([agent_a, agent_b, agent_c])
        if not isinstance(result, End):
            ss = build_statespace(result)
            lr = check_lattice(ss)
            assert lr.is_lattice is True


# ---------------------------------------------------------------------------
# MCP scenario tests
# ---------------------------------------------------------------------------


class TestMCPScenario:
    def test_mcp_full_compatible_with_minimal(self, mcp_full, mcp_minimal):
        """Full MCP and minimal MCP share 'initialize'."""
        assert compatible(mcp_full, mcp_minimal) is True

    def test_mcp_negotiation(self, mcp_full, mcp_minimal):
        """Negotiation of full and minimal MCP."""
        result = negotiate(mcp_full, mcp_minimal)
        assert not isinstance(result, End)

    def test_mcp_negotiated_is_lattice(self, mcp_full, mcp_minimal):
        result = negotiate(mcp_full, mcp_minimal)
        if not isinstance(result, End):
            ss = build_statespace(result)
            lr = check_lattice(ss)
            assert lr.is_lattice is True
