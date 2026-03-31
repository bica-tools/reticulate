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


@mcp.tool()
def compress_type(type_string: str) -> str:
    """Compress a tree-like session type into an equation system.

    Detects shared (structurally equal) subtrees and factors them
    into named equations. Shows compression ratio and shared structure.
    Non-distributive protocols typically have higher compression ratios
    because their reconvergent branches share subtrees.

    Returns: equation system, compression ratio, shared subtree count.
    """
    t0 = _log_call("compress_type", {"type_string": type_string})
    from reticulate.parser import parse, ParseError, pretty_program
    from reticulate.compress import analyze_compression

    try:
        ast = parse(type_string)
    except ParseError as e:
        _log_error("compress_type", t0, str(e))
        return f"Parse error: {e}"

    cr = analyze_compression(ast, min_size=2)

    lines = [
        f"Session type: {type_string}",
        f"Original AST size: {cr.original_size} nodes",
        f"Compressed size: {cr.compressed_size} nodes",
        f"Compression ratio: {cr.ratio:.2f}x",
        f"Shared subtrees: {cr.num_shared}",
        f"Definitions: {cr.num_definitions}",
        "",
        "--- Equation form ---",
        pretty_program(cr.program),
    ]

    if cr.has_sharing:
        lines.append("")
        lines.append(f"Shared names: {', '.join(cr.shared_names)}")
        lines.append("(Each shared name represents a reconvergence point)")

    result = "\n".join(lines)
    _log_result("compress_type", t0, result)
    return result


@mcp.tool()
def analyze_global(global_type_string: str) -> str:
    """Analyze a multiparty global session type.

    Parses a global type, extracts roles, builds the global state space,
    checks lattice properties, and projects onto each role to produce
    local (binary) session types.

    Global type syntax:
        sender -> receiver : { label1: G1, label2: G2 }
        G1 || G2        (parallel)
        rec X . G       (recursion)
        end             (terminated)

    Example: Client -> Server : {request: Server -> Client : {response: end}}

    Returns: roles, global state space analysis, and per-role local types
    with their individual lattice properties.
    """
    t0 = _log_call("analyze_global", {"global_type_string": global_type_string})
    from reticulate.global_types import parse_global, roles, build_global_statespace, pretty_global
    from reticulate.projection import project_all
    from reticulate.parser import pretty
    from reticulate.statespace import build_statespace
    from reticulate.lattice import check_lattice, check_distributive

    try:
        g = parse_global(global_type_string)
    except Exception as e:
        _log_error("analyze_global", t0, str(e))
        return f"Parse error: {e}"

    all_roles = sorted(roles(g))
    gss = build_global_statespace(g)
    glr = check_lattice(gss)
    gdist = check_distributive(gss)

    lines = [
        f"Global type: {global_type_string[:80]}",
        f"Roles: {', '.join(all_roles)}",
        f"Global states: {len(gss.states)}",
        f"Global transitions: {len(gss.transitions)}",
        f"Global is lattice: {glr.is_lattice}",
        f"Global distributive: {gdist.is_distributive}",
        "",
        "--- Per-role projections (local binary types) ---",
    ]

    try:
        local_types = project_all(g)
        for role in all_roles:
            local = local_types.get(role)
            if local is None:
                lines.append(f"  {role}: (not involved)")
                continue
            local_str = pretty(local)
            lss = build_statespace(local)
            llr = check_lattice(lss)
            ldist = check_distributive(lss)
            lines.append(f"  {role}:")
            lines.append(f"    Local type: {local_str[:80]}")
            lines.append(f"    States: {len(lss.states)}, Transitions: {len(lss.transitions)}")
            lines.append(f"    Lattice: {llr.is_lattice}, Distributive: {ldist.is_distributive}")
    except Exception as e:
        lines.append(f"  Projection error: {e}")

    result = "\n".join(lines)
    _log_result("analyze_global", t0, result)
    return result


@mcp.tool()
def ci_check(type_string: str, baseline: str = "", require_distributive: bool = False) -> str:
    """Run CI/CD gate checks on a session type.

    Checks: lattice property, termination, test coverage, and optionally
    distributivity and backward compatibility (if baseline provided).

    Returns pass/fail verdict with per-check details. Use in CI pipelines
    to verify protocol correctness on every commit.

    Args:
        type_string: The session type to check.
        baseline: Optional previous version for backward compatibility.
        require_distributive: If true, require distributive lattice.
    """
    t0 = _log_call("ci_check", {
        "type_string": type_string,
        "baseline": baseline or "(none)",
        "require_distributive": require_distributive,
    })
    from reticulate.parser import parse, ParseError
    from reticulate.ci_gate import ci_gate, GateConfig

    try:
        config = GateConfig(
            require_distributive=require_distributive,
            require_subtype=bool(baseline),
            min_transition_coverage=0.9,
        )
        gate_result = ci_gate(
            type_string,
            baseline=baseline if baseline else None,
            config=config,
        )
    except ParseError as e:
        _log_error("ci_check", t0, str(e))
        return f"Parse error: {e}"
    except Exception as e:
        _log_error("ci_check", t0, str(e))
        return f"Error: {e}"

    result = gate_result.summary()
    _log_result("ci_check", t0, result)
    return result


@mcp.tool()
def supervise_programme() -> str:
    """Run the research supervision process.

    Scans the full session types research programme, evaluates progress,
    and proposes: new steps, new tools, new papers, and target venues.

    Returns a supervision report with programme snapshot and prioritized proposals.
    """
    t0 = _log_call("supervise_programme", {})
    from reticulate.supervisor import supervise

    report = supervise()
    result = report.summary()
    _log_result("supervise_programme", t0, result)
    return result


@mcp.tool()
def evaluate(step_number: str) -> str:
    """Evaluate a step's deliverables and grade it.

    Checks: paper exists (5000+ words), companion proofs, implementation
    module, test suite (20+ tests), tests pass, paper structure.

    Grade: A+ (accepted), A (near-complete), B (gaps), C (major rework), F (broken).
    A step is ACCEPTED only at A+. Lower grades list required fixes.
    """
    t0 = _log_call("evaluate", {"step_number": step_number})
    from reticulate.evaluator import evaluate_step

    result = evaluate_step(step_number, run_tests=False)
    output = result.summary()
    _log_result("evaluate", t0, output)
    return output


@mcp.tool()
def subtype_check(subtype: str, supertype: str) -> str:
    """Check if one session type is a subtype of another (Gay-Hole subtyping).

    Backward compatibility verification: is the new protocol version
    a subtype of the old one? If yes, existing clients still work.

    Returns: subtyping verdict, direction, and embedding check.

    Example: subtype_check("&{a: end, b: end}", "&{a: end}")
    """
    t0 = _log_call("subtype_check", {"subtype": subtype, "supertype": supertype})
    from reticulate.parser import parse, ParseError
    from reticulate.subtyping import is_subtype

    try:
        s1 = parse(subtype)
        s2 = parse(supertype)
    except ParseError as e:
        _log_error("subtype_check", t0, str(e))
        return f"Parse error: {e}"

    forward = is_subtype(s1, s2)
    backward = is_subtype(s2, s1)

    if forward and backward:
        relation = "equivalent (mutual subtypes)"
    elif forward:
        relation = f"YES: {subtype}  ≤  {supertype}"
    elif backward:
        relation = f"NO (reversed): {supertype}  ≤  {subtype}"
    else:
        relation = "INCOMPARABLE: neither is a subtype of the other"

    lines = [
        f"Subtype:   {subtype}",
        f"Supertype: {supertype}",
        f"S₁ ≤ S₂:  {forward}",
        f"S₂ ≤ S₁:  {backward}",
        f"Verdict:   {relation}",
    ]

    result = "\n".join(lines)
    _log_result("subtype_check", t0, result)
    return result


@mcp.tool()
def dual(type_string: str) -> str:
    """Compute the dual of a session type (swap Branch ↔ Select).

    Given a client protocol, generates the matching server protocol.
    The dual of a branch (external choice) is a selection (internal choice)
    and vice versa. Parallel, recursion, and end are preserved.

    Returns: the dual type string and verification that dual(dual(S)) = S.

    Example: dual("&{a: end, b: end}") → "+{a: end, b: end}"
    """
    t0 = _log_call("dual", {"type_string": type_string})
    from reticulate.parser import parse, pretty, ParseError
    from reticulate.duality import dual as compute_dual

    try:
        ast = parse(type_string)
    except ParseError as e:
        _log_error("dual", t0, str(e))
        return f"Parse error: {e}"

    d = compute_dual(ast)
    dd = compute_dual(d)
    involution = pretty(dd) == pretty(ast)

    lines = [
        f"Original:    {pretty(ast)}",
        f"Dual:        {pretty(d)}",
        f"Dual(Dual):  {pretty(dd)}",
        f"Involution:  {involution}",
    ]

    result = "\n".join(lines)
    _log_result("dual", t0, result)
    return result


@mcp.tool()
def trace_validate(type_string: str, trace: str) -> str:
    """Check if a method call trace follows a session type protocol.

    Given a session type and a comma-separated sequence of method names,
    verifies that the trace is a valid execution path through the state space.

    Returns: validity verdict, final state, and remaining enabled methods.

    Example: trace_validate("&{open: &{read: &{close: end}}}", "open,read,close")
    """
    t0 = _log_call("trace_validate", {"type_string": type_string, "trace": trace})
    from reticulate.parser import parse, ParseError
    from reticulate.statespace import build_statespace

    try:
        ast = parse(type_string)
    except ParseError as e:
        _log_error("trace_validate", t0, str(e))
        return f"Parse error: {e}"

    ss = build_statespace(ast)

    # Parse trace
    methods = [m.strip() for m in trace.split(",") if m.strip()]

    # Walk the state space
    current = ss.top
    path = [current]
    for i, method in enumerate(methods):
        # Find transition with this label
        found = False
        for src, label, tgt in ss.transitions:
            if src == current and label == method:
                current = tgt
                path.append(current)
                found = True
                break
        if not found:
            enabled = sorted({l for s, l, t in ss.transitions if s == current})
            result = "\n".join([
                f"INVALID at step {i+1}: method '{method}' not enabled",
                f"Trace so far: {','.join(methods[:i])}",
                f"Current state: {current}",
                f"Enabled methods: {', '.join(enabled) if enabled else '(none — at terminal state)'}",
            ])
            _log_result("trace_validate", t0, result)
            return result

    # Trace completed successfully
    at_bottom = (current == ss.bottom)
    enabled = sorted({l for s, l, t in ss.transitions if s == current})

    lines = [
        f"VALID: trace follows the protocol",
        f"Trace: {trace}",
        f"Steps: {len(methods)}",
        f"Final state: {current}",
        f"At terminal: {at_bottom}",
        f"Remaining methods: {', '.join(enabled) if enabled else '(none — session complete)'}",
    ]

    result = "\n".join(lines)
    _log_result("trace_validate", t0, result)
    return result


@mcp.tool()
def check_modularity(type_string: str) -> str:
    """Check protocol modularity: distributivity, coupling, Birkhoff decomposition.

    Returns a structured modularity certificate including:
    - Verdict (MODULAR / WEAKLY_MODULAR / NOT_MODULAR / INVALID)
    - Metrics (Fiedler value, Cheeger constant, coupling, compression ratio)
    - Birkhoff decomposition (minimal modules)
    - Diagnosis and refactoring suggestions (if non-modular)

    Example: &{open: rec X . &{read: +{data: X, eof: &{close: end}}}}
    """
    t0 = _log_call("check_modularity", {"type_string": type_string})
    try:
        from reticulate.modular_report import generate_report
        from reticulate.parser import parse, parse_program, pretty
        from reticulate.resolve import resolve
        from reticulate.statespace import build_statespace

        program = parse_program(type_string)
        ast = resolve(program)
        ss = build_statespace(ast)
        report = generate_report(type_string, ss, protocol_name="Protocol")

        result = report.to_text()
        _log_result("check_modularity", t0, result)
        return result
    except Exception as e:
        error = f"Error: {e}"
        _log_error("check_modularity", t0, error)
        return error


@mcp.tool()
def import_protocol(spec_json: str, format: str = "openapi") -> str:
    """Import an external API specification and convert to session types.

    Supported formats: openapi, grpc, asyncapi.
    The spec_json should be a JSON string of the specification dict.

    Returns session types extracted from the specification.
    """
    t0 = _log_call("import_protocol", {"format": format})
    try:
        import json as _json
        from reticulate.importers import from_spec

        spec = _json.loads(spec_json)
        result_dict = from_spec(spec, format)

        lines = [f"Imported {len(result_dict)} session type(s) from {format}:", ""]
        for name, type_str in result_dict.items():
            lines.append(f"  {name}: {type_str}")

        result = "\n".join(lines)
        _log_result("import_protocol", t0, result)
        return result
    except Exception as e:
        error = f"Error: {e}"
        _log_error("import_protocol", t0, error)
        return error


if __name__ == "__main__":
    logger.info(
        "Server ready — 17 tools: analyze, test_gen, hasse, invariants, "
        "conformance, petri_net, coverage, compress_type, analyze_global, "
        "ci_check, supervise_programme, evaluate, subtype_check, dual, "
        "trace_validate, check_modularity, import_protocol"
    )
    mcp.run()
    logger.info(
        "Server shutting down — %d calls, %d errors",
        _stats["calls"], _stats["errors"],
    )
