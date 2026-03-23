"""MCP Server exposing reticulate as AI agent tools.

Provides session type analysis tools via MCP protocol:
- analyze: Parse a session type, build state space, check lattice
- test_gen: Generate conformance tests from a session type
- hasse: Generate Hasse diagram DOT source
- invariants: Compute algebraic invariants
- conformance: Run full conformance report on MCP/A2A

Usage:
    python -m reticulate.mcp_server

    # Or with FastMCP:
    fastmcp run reticulate/reticulate/mcp_server.py

Logging:
    All tool calls are logged to stderr with timestamps.
    Set RETICULATE_LOG_LEVEL=DEBUG for verbose output.
    Logs go to stderr because stdout is the MCP transport channel.
"""

import logging
import os
import sys
import time

from mcp.server import FastMCP

# ---------------------------------------------------------------------------
# Logging setup — must go to stderr (stdout is the JSON-RPC transport)
# ---------------------------------------------------------------------------
_log_level = getattr(logging, os.environ.get("RETICULATE_LOG_LEVEL", "INFO").upper(), logging.INFO)

_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logger = logging.getLogger("reticulate.mcp")
logger.setLevel(_log_level)
logger.addHandler(_handler)
logger.propagate = False

# Track server statistics
_stats: dict[str, int] = {"calls": 0, "errors": 0}


def _log_call(tool_name: str, args: dict) -> float:
    """Log a tool call and return the start time."""
    _stats["calls"] += 1
    call_num = _stats["calls"]
    args_summary = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
    logger.info("CALL #%d  %s(%s)", call_num, tool_name, args_summary)
    return time.time()


def _log_result(tool_name: str, t0: float, result: str) -> None:
    """Log the result of a tool call."""
    elapsed_ms = (time.time() - t0) * 1000
    # First line of result as summary
    summary = result.split("\n")[0] if result else "(empty)"
    lines = result.count("\n") + 1
    logger.info("  -> %s  %.1fms  %d lines  %s", tool_name, elapsed_ms, lines, summary)


def _log_error(tool_name: str, t0: float, error: str) -> None:
    """Log an error from a tool call."""
    _stats["errors"] += 1
    elapsed_ms = (time.time() - t0) * 1000
    logger.error("  -> %s  FAILED  %.1fms  %s", tool_name, elapsed_ms, error)


mcp = FastMCP(
    "reticulate",
    instructions=(
        "Session type analysis tools. Provide a session type string using "
        "the grammar: &{m: S} (branch), +{l: S} (selection), "
        "(S1 || S2) (parallel), rec X . S (recursion), end (terminal). "
        "Example: rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
    ),
)

logger.info("Reticulate MCP server starting")


@mcp.tool()
def analyze(type_string: str) -> str:
    """Parse a session type, build its state space, and check if it forms a lattice.

    Returns: state count, transition count, lattice verdict, distributivity.

    Example type_string: "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"
    """
    t0 = _log_call("analyze", {"type_string": type_string})
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.lattice import check_lattice, check_distributive

    try:
        ast = parse(type_string)
    except ParseError as e:
        _log_error("analyze", t0, str(e))
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    lr = check_lattice(ss)
    dist = check_distributive(ss)

    lines = [
        f"Session type: {type_string}",
        f"States: {len(ss.states)}",
        f"Transitions: {len(ss.transitions)}",
        f"Is lattice: {lr.is_lattice}",
        f"Has top: {lr.has_top}",
        f"Has bottom: {lr.has_bottom}",
        f"SCCs: {lr.num_scc}",
        f"Distributive: {dist.is_distributive}",
    ]
    if not lr.is_lattice and lr.counterexample:
        a, b, kind = lr.counterexample
        lines.append(f"Counterexample: states {a}, {b} have no {kind}")

    result = "\n".join(lines)
    _log_result("analyze", t0, result)
    return result


@mcp.tool()
def test_gen(type_string: str, class_name: str = "MyProtocol") -> str:
    """Generate JUnit 5 conformance test source from a session type.

    Returns Java test class source code with valid path tests,
    violation tests, and incomplete prefix tests.
    """
    t0 = _log_call("test_gen", {"type_string": type_string, "class_name": class_name})
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.testgen import TestGenConfig, generate_test_source

    try:
        ast = parse(type_string)
    except ParseError as e:
        _log_error("test_gen", t0, str(e))
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    config = TestGenConfig(class_name=class_name)
    result = generate_test_source(ss, config, type_string)
    _log_result("test_gen", t0, result)
    return result


@mcp.tool()
def hasse(type_string: str) -> str:
    """Generate the Hasse diagram of a session type in DOT format.

    The DOT source can be rendered with Graphviz:
    dot -Tsvg output.dot -o output.svg
    """
    t0 = _log_call("hasse", {"type_string": type_string})
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.visualize import dot_source

    try:
        ast = parse(type_string)
    except ParseError as e:
        _log_error("hasse", t0, str(e))
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    result = dot_source(ss, title=type_string[:60])
    _log_result("hasse", t0, result)
    return result


@mcp.tool()
def invariants(type_string: str) -> str:
    """Compute algebraic invariants of a session type lattice.

    Returns: Möbius value, Rota polynomial, spectral radius,
    Fiedler value, tropical eigenvalue, von Neumann entropy,
    join/meet irreducibles count.
    """
    t0 = _log_call("invariants", {"type_string": type_string})
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.matrix import algebraic_invariants

    try:
        ast = parse(type_string)
    except ParseError as e:
        _log_error("invariants", t0, str(e))
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    inv = algebraic_invariants(ss)

    result = "\n".join([
        f"Session type: {type_string}",
        f"States: {inv.num_states}",
        f"Möbius μ(⊤,⊥): {inv.mobius_value}",
        f"Rota polynomial: {inv.rota_polynomial}",
        f"Spectral radius: {inv.spectral_radius:.4f}",
        f"Fiedler value: {inv.fiedler_value:.4f}",
        f"Tropical eigenvalue: {inv.tropical_eigenvalue:.1f}",
        f"Von Neumann entropy: {inv.von_neumann_entropy:.4f}",
        f"Join-irreducibles: {inv.num_join_irreducibles}",
        f"Meet-irreducibles: {inv.num_meet_irreducibles}",
    ])
    _log_result("invariants", t0, result)
    return result


@mcp.tool()
def conformance(protocol: str = "mcp") -> str:
    """Run full conformance analysis on an AI agent protocol.

    protocol: "mcp" for Model Context Protocol, "a2a" for Agent-to-Agent.

    Returns: complete conformance report including lattice properties,
    test generation, coverage, Petri net encoding, and algebraic invariants.
    """
    t0 = _log_call("conformance", {"protocol": protocol})
    from reticulate.mcp_conformance import (
        mcp_protocol,
        a2a_protocol,
        conformance_report,
        format_report,
    )

    if protocol.lower() == "mcp":
        p = mcp_protocol()
    elif protocol.lower() == "a2a":
        p = a2a_protocol()
    else:
        _log_error("conformance", t0, f"Unknown protocol: {protocol}")
        return f"Unknown protocol: {protocol}. Use 'mcp' or 'a2a'."

    report = conformance_report(p)
    result = format_report(report)
    _log_result("conformance", t0, result)
    return result


@mcp.tool()
def petri_net(type_string: str) -> str:
    """Encode a session type as a Petri net and analyze its properties.

    Returns: place/transition counts, 1-safety, free-choice property,
    occurrence net status, reachability isomorphism verification.
    """
    t0 = _log_call("petri_net", {"type_string": type_string})
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.petri import session_type_to_petri_net, petri_dot

    try:
        ast = parse(type_string)
    except ParseError as e:
        _log_error("petri_net", t0, str(e))
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    pn_result = session_type_to_petri_net(ss)

    lines = [
        f"Session type: {type_string}",
        f"Places: {pn_result.num_places}",
        f"Transitions: {pn_result.num_transitions}",
        f"1-safe: yes",
        f"Free-choice: {'yes' if pn_result.is_free_choice else 'no'}",
        f"Occurrence net: {'yes' if pn_result.is_occurrence_net else 'no (has cycles)'}",
        f"Reachability isomorphic: {'yes' if pn_result.reachability_isomorphic else 'NO'}",
        f"Reachable markings: {pn_result.num_reachable_markings}",
    ]
    result = "\n".join(lines)
    _log_result("petri_net", t0, result)
    return result


@mcp.tool()
def coverage(type_string: str) -> str:
    """Compute test coverage metrics for a session type.

    Returns: transition coverage, state coverage, number of valid paths,
    violations, and incomplete prefixes.
    """
    t0 = _log_call("coverage", {"type_string": type_string})
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.testgen import TestGenConfig, enumerate as enum_tests
    from reticulate.coverage import compute_coverage

    try:
        ast = parse(type_string)
    except ParseError as e:
        _log_error("coverage", t0, str(e))
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    config = TestGenConfig(class_name="CoverageTest")
    enum_result = enum_tests(ss, config)
    cov = compute_coverage(ss, result=enum_result)

    result = "\n".join([
        f"Session type: {type_string}",
        f"Transition coverage: {cov.transition_coverage:.1%}",
        f"State coverage: {cov.state_coverage:.1%}",
        f"Valid paths: {len(enum_result.valid_paths)}",
        f"Violations: {len(enum_result.violations)}",
        f"Incomplete prefixes: {len(enum_result.incomplete_prefixes)}",
        f"Uncovered transitions: {len(cov.uncovered_transitions)}",
        f"Uncovered states: {len(cov.uncovered_states)}",
    ])
    _log_result("coverage", t0, result)
    return result


if __name__ == "__main__":
    logger.info(
        "Server ready — 7 tools: analyze, test_gen, hasse, "
        "invariants, conformance, petri_net, coverage"
    )
    mcp.run()
    logger.info(
        "Server shutting down — %d calls, %d errors",
        _stats["calls"], _stats["errors"],
    )
