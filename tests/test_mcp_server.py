"""Tests for the MCP server tool functions (mcp_server.py).

Tests each tool function directly by importing and calling with test
session type strings. Does NOT test the MCP protocol transport layer.
"""

import pytest


# ---------------------------------------------------------------------------
# Import helpers — skip entire module if mcp dependency is missing
# ---------------------------------------------------------------------------

mcp_available = True
try:
    from reticulate.mcp_server import (
        analyze,
        hasse,
        invariants,
        petri_net,
        coverage,
        compress_type,
        analyze_global,
        ci_check,
        subtype_check,
        dual,
        trace_validate,
    )
    from reticulate.mcp_server import test_gen as mcp_test_gen
except ImportError:
    mcp_available = False

pytestmark = pytest.mark.skipif(not mcp_available, reason="mcp package not installed")


# ===================================================================
# Test data
# ===================================================================

SIMPLE_BRANCH = "&{a: end}"
TWO_BRANCH = "&{a: end, b: end}"
NESTED = "&{a: &{b: end}}"
SELECTION = "+{ok: end, err: end}"
RECURSIVE = "rec X . &{next: X, stop: end}"
PARALLEL = "(&{a: end} || &{b: end})"
INVALID = "&{a: end"  # missing closing brace


# ===================================================================
# analyze tool
# ===================================================================

class TestAnalyze:
    def test_basic_output_fields(self) -> None:
        result = analyze(SIMPLE_BRANCH)
        assert "States:" in result
        assert "Transitions:" in result
        assert "Is lattice:" in result
        assert "Distributive:" in result

    def test_simple_is_lattice(self) -> None:
        result = analyze(SIMPLE_BRANCH)
        assert "Is lattice: True" in result

    def test_two_branch_state_count(self) -> None:
        result = analyze(TWO_BRANCH)
        # &{a: end, b: end} has 2 states: top and end (bottom)
        assert "States: 2" in result

    def test_recursive_type(self) -> None:
        result = analyze(RECURSIVE)
        assert "Is lattice:" in result

    def test_parallel_type(self) -> None:
        result = analyze(PARALLEL)
        assert "Is lattice: True" in result

    def test_selection_type(self) -> None:
        result = analyze(SELECTION)
        assert "States:" in result

    def test_parse_error(self) -> None:
        result = analyze(INVALID)
        assert "Parse error" in result


# ===================================================================
# test_gen tool
# ===================================================================

class TestTestGen:
    def test_generates_java_source(self) -> None:
        result = mcp_test_gen(TWO_BRANCH)
        assert "class" in result or "void" in result

    def test_custom_class_name(self) -> None:
        result = mcp_test_gen(TWO_BRANCH, class_name="MyProto")
        assert "MyProto" in result

    def test_parse_error(self) -> None:
        result = mcp_test_gen(INVALID)
        assert "Parse error" in result


# ===================================================================
# hasse tool
# ===================================================================

class TestHasse:
    def test_returns_dot_source(self) -> None:
        result = hasse(SIMPLE_BRANCH)
        assert "digraph" in result

    def test_contains_nodes(self) -> None:
        result = hasse(TWO_BRANCH)
        assert "->" in result  # DOT edges

    def test_parse_error(self) -> None:
        result = hasse(INVALID)
        assert "Parse error" in result


# ===================================================================
# invariants tool
# ===================================================================

class TestInvariants:
    def test_returns_invariant_fields(self) -> None:
        result = invariants(SIMPLE_BRANCH)
        assert "Möbius" in result or "Mobius" in result
        assert "States:" in result

    def test_spectral_radius(self) -> None:
        result = invariants(TWO_BRANCH)
        assert "Spectral radius:" in result

    def test_parse_error(self) -> None:
        result = invariants(INVALID)
        assert "Parse error" in result


# ===================================================================
# petri_net tool
# ===================================================================

class TestPetriNet:
    def test_returns_places_transitions(self) -> None:
        result = petri_net(SIMPLE_BRANCH)
        assert "Places:" in result
        assert "Transitions:" in result

    def test_one_safe(self) -> None:
        result = petri_net(TWO_BRANCH)
        assert "1-safe: yes" in result

    def test_parse_error(self) -> None:
        result = petri_net(INVALID)
        assert "Parse error" in result


# ===================================================================
# coverage tool
# ===================================================================

class TestCoverage:
    def test_returns_coverage_fields(self) -> None:
        result = coverage(SIMPLE_BRANCH)
        assert "Transition coverage:" in result
        assert "State coverage:" in result

    def test_valid_paths_count(self) -> None:
        result = coverage(TWO_BRANCH)
        assert "Valid paths:" in result

    def test_parse_error(self) -> None:
        result = coverage(INVALID)
        assert "Parse error" in result


# ===================================================================
# compress_type tool
# ===================================================================

class TestCompressType:
    def test_returns_compression_info(self) -> None:
        result = compress_type(SIMPLE_BRANCH)
        assert "Original AST size:" in result
        assert "Compression ratio:" in result

    def test_equation_form(self) -> None:
        result = compress_type(NESTED)
        assert "Equation form" in result

    def test_parse_error(self) -> None:
        result = compress_type(INVALID)
        assert "Parse error" in result


# ===================================================================
# analyze_global tool
# ===================================================================

class TestAnalyzeGlobal:
    def test_simple_global(self) -> None:
        g = "Client -> Server : {request: end}"
        result = analyze_global(g)
        assert "Roles:" in result
        assert "Client" in result
        assert "Server" in result

    def test_global_lattice(self) -> None:
        g = "Client -> Server : {request: Server -> Client : {response: end}}"
        result = analyze_global(g)
        assert "Global is lattice:" in result

    def test_parse_error(self) -> None:
        result = analyze_global("not a valid global type {{{{")
        assert "error" in result.lower()


# ===================================================================
# ci_check tool
# ===================================================================

class TestCiCheck:
    def test_simple_pass(self) -> None:
        result = ci_check(SIMPLE_BRANCH)
        # Should produce some verdict
        assert len(result) > 0

    def test_recursive_check(self) -> None:
        result = ci_check(RECURSIVE)
        assert len(result) > 0


# ===================================================================
# subtype_check tool
# ===================================================================

class TestSubtypeCheck:
    def test_subtype_more_methods(self) -> None:
        result = subtype_check("&{a: end, b: end}", "&{a: end}")
        assert "True" in result
        assert "YES" in result

    def test_not_subtype(self) -> None:
        result = subtype_check("&{a: end}", "&{a: end, b: end}")
        assert "False" in result or "NO" in result or "reversed" in result.lower()

    def test_equivalent(self) -> None:
        result = subtype_check("&{a: end}", "&{a: end}")
        assert "equivalent" in result.lower() or "True" in result

    def test_incomparable(self) -> None:
        result = subtype_check("+{a: end, b: end}", "&{a: end, b: end}")
        assert len(result) > 0

    def test_parse_error(self) -> None:
        result = subtype_check("&{bad", "&{a: end}")
        assert "error" in result.lower()


# ===================================================================
# dual tool
# ===================================================================

class TestDual:
    def test_branch_to_select(self) -> None:
        result = dual("&{a: end, b: end}")
        assert "+{" in result or "⊕" in result

    def test_involution(self) -> None:
        result = dual("&{open: &{read: +{data: end, eof: end}}}")
        assert "Involution:  True" in result

    def test_end_is_self_dual(self) -> None:
        result = dual("end")
        assert "end" in result

    def test_recursive(self) -> None:
        result = dual("rec X . &{a: X, b: end}")
        assert "rec" in result.lower() or "μ" in result

    def test_parse_error(self) -> None:
        result = dual("not a type")
        assert "error" in result.lower()


# ===================================================================
# trace_validate tool
# ===================================================================

class TestTraceValidate:
    def test_valid_complete_trace(self) -> None:
        result = trace_validate("&{open: &{read: &{close: end}}}", "open,read,close")
        assert "VALID" in result
        assert "terminal: True" in result

    def test_valid_partial_trace(self) -> None:
        result = trace_validate("&{open: &{read: &{close: end}}}", "open,read")
        assert "VALID" in result
        assert "terminal: False" in result

    def test_invalid_trace(self) -> None:
        result = trace_validate("&{open: &{read: &{close: end}}}", "open,close")
        assert "INVALID" in result
        assert "close" in result

    def test_empty_trace(self) -> None:
        result = trace_validate("&{a: end}", "")
        assert "VALID" in result

    def test_branch_choice(self) -> None:
        result = trace_validate("&{a: end, b: end}", "a")
        assert "VALID" in result
        assert "terminal: True" in result

    def test_parse_error(self) -> None:
        result = trace_validate("bad type", "a,b")
        assert "error" in result.lower()

    def test_with_distributive_requirement(self) -> None:
        result = ci_check(TWO_BRANCH, require_distributive=True)
        assert len(result) > 0
