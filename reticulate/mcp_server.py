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
"""

from mcp.server import FastMCP

mcp = FastMCP(
    "reticulate",
    instructions=(
        "Session type analysis tools. Provide a session type string using "
        "the grammar: &{m: S} (branch), +{l: S} (selection), "
        "(S1 || S2) (parallel), rec X . S (recursion), end (terminal). "
        "Example: rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
    ),
)


@mcp.tool()
def analyze(type_string: str) -> str:
    """Parse a session type, build its state space, and check if it forms a lattice.

    Returns: state count, transition count, lattice verdict, distributivity.

    Example type_string: "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"
    """
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.lattice import check_lattice, check_distributive

    try:
        ast = parse(type_string)
    except ParseError as e:
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

    return "\n".join(lines)


@mcp.tool()
def test_gen(type_string: str, class_name: str = "MyProtocol") -> str:
    """Generate JUnit 5 conformance test source from a session type.

    Returns Java test class source code with valid path tests,
    violation tests, and incomplete prefix tests.
    """
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.testgen import TestGenConfig, generate_test_source

    try:
        ast = parse(type_string)
    except ParseError as e:
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    config = TestGenConfig(class_name=class_name)
    return generate_test_source(ss, config, type_string)


@mcp.tool()
def hasse(type_string: str) -> str:
    """Generate the Hasse diagram of a session type in DOT format.

    The DOT source can be rendered with Graphviz:
    dot -Tsvg output.dot -o output.svg
    """
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.visualize import dot_source

    try:
        ast = parse(type_string)
    except ParseError as e:
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    return dot_source(ss, title=type_string[:60])


@mcp.tool()
def invariants(type_string: str) -> str:
    """Compute algebraic invariants of a session type lattice.

    Returns: Möbius value, Rota polynomial, spectral radius,
    Fiedler value, tropical eigenvalue, von Neumann entropy,
    join/meet irreducibles count.
    """
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.matrix import algebraic_invariants

    try:
        ast = parse(type_string)
    except ParseError as e:
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    inv = algebraic_invariants(ss)

    return "\n".join([
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


@mcp.tool()
def conformance(protocol: str = "mcp") -> str:
    """Run full conformance analysis on an AI agent protocol.

    protocol: "mcp" for Model Context Protocol, "a2a" for Agent-to-Agent.

    Returns: complete conformance report including lattice properties,
    test generation, coverage, Petri net encoding, and algebraic invariants.
    """
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
        return f"Unknown protocol: {protocol}. Use 'mcp' or 'a2a'."

    report = conformance_report(p)
    return format_report(report)


@mcp.tool()
def petri_net(type_string: str) -> str:
    """Encode a session type as a Petri net and analyze its properties.

    Returns: place/transition counts, 1-safety, free-choice property,
    occurrence net status, reachability isomorphism verification.
    """
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.petri import session_type_to_petri_net, petri_dot

    try:
        ast = parse(type_string)
    except ParseError as e:
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    result = session_type_to_petri_net(ss)

    lines = [
        f"Session type: {type_string}",
        f"Places: {result.num_places}",
        f"Transitions: {result.num_transitions}",
        f"1-safe: yes",
        f"Free-choice: {'yes' if result.is_free_choice else 'no'}",
        f"Occurrence net: {'yes' if result.is_occurrence_net else 'no (has cycles)'}",
        f"Reachability isomorphic: {'yes' if result.reachability_isomorphic else 'NO'}",
        f"Reachable markings: {result.num_reachable_markings}",
    ]
    return "\n".join(lines)


@mcp.tool()
def coverage(type_string: str) -> str:
    """Compute test coverage metrics for a session type.

    Returns: transition coverage, state coverage, number of valid paths,
    violations, and incomplete prefixes.
    """
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace
    from reticulate.testgen import TestGenConfig, enumerate as enum_tests
    from reticulate.coverage import compute_coverage

    try:
        ast = parse(type_string)
    except ParseError as e:
        return f"Parse error: {e}"

    ss = build_statespace(ast)
    config = TestGenConfig(class_name="CoverageTest")
    result = enum_tests(ss, config)
    cov = compute_coverage(ss, result=result)

    return "\n".join([
        f"Session type: {type_string}",
        f"Transition coverage: {cov.transition_coverage:.1%}",
        f"State coverage: {cov.state_coverage:.1%}",
        f"Valid paths: {len(result.valid_paths)}",
        f"Violations: {len(result.violations)}",
        f"Incomplete prefixes: {len(result.incomplete_prefixes)}",
        f"Uncovered transitions: {len(cov.uncovered_transitions)}",
        f"Uncovered states: {len(cov.uncovered_states)}",
    ])


if __name__ == "__main__":
    mcp.run()
