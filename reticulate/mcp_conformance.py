"""MCP/A2A conformance testing for AI agent protocols (Step 70).

The first formal conformance testing tool for AI agent protocols.
Models MCP (Anthropic) and A2A (Google) as session types, generates
conformance test suites, computes coverage, and produces full
analysis reports including lattice properties, Petri net encoding,
and algebraic invariants.

Usage:
    python -m reticulate.agent_test --protocol mcp --report
    python -m reticulate.agent_test --protocol a2a --test-gen
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.coverage import CoverageResult, compute_coverage
from reticulate.lattice import (
    DistributivityResult,
    LatticeResult,
    check_distributive,
    check_lattice,
)
from reticulate.marking_lattice import MarkingLatticeResult, analyze_marking_lattice
from reticulate.matrix import AlgebraicInvariants, algebraic_invariants
from reticulate.parser import SessionType, parse
from reticulate.petri import PetriNetResult, session_type_to_petri_net
from reticulate.statespace import StateSpace, build_statespace
from reticulate.testgen import (
    EnumerationResult,
    TestGenConfig,
    enumerate as enumerate_tests,
    generate_test_source,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Canonical protocol type strings
# ---------------------------------------------------------------------------

MCP_TYPE_STRING: str = (
    "&{initialize: (rec X . &{callTool: +{RESULT: X, ERROR: X}, "
    "listTools: X, shutdown: end} || "
    "rec Y . +{NOTIFICATION: Y, DONE: end})}"
)

A2A_TYPE_STRING: str = (
    "&{sendTask: rec X . +{WORKING: &{getStatus: X, cancel: end}, "
    "COMPLETED: &{getArtifact: end}, FAILED: end}}"
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransitionAnnotation:
    """Protocol-level metadata for a single transition.

    Attributes:
        source: Source state ID.
        label: Transition label (method/message name).
        target: Target state ID.
        initiator: "client" or "server".
        message_kind: "request", "response", or "notification".
        description: Human-readable description.
    """
    source: int
    label: str
    target: int
    initiator: str
    message_kind: str
    description: str


@dataclass(frozen=True)
class ProtocolModel:
    """A named protocol with its session type and annotations.

    Attributes:
        name: Protocol name (e.g., "MCP", "A2A").
        version: Protocol version string.
        type_string: Canonical session type string.
        ast: Parsed session type AST.
        description: Human-readable protocol description.
        annotations: Per-transition metadata.
    """
    name: str
    version: str
    type_string: str
    ast: SessionType
    description: str
    annotations: tuple[TransitionAnnotation, ...]


@dataclass(frozen=True)
class ConformanceReport:
    """Complete conformance analysis of a protocol.

    Attributes:
        protocol: The protocol model.
        state_space: Constructed state space.
        lattice_result: Lattice property check.
        distributivity: Distributivity check.
        enumeration: Test path enumeration.
        test_source: Generated JUnit 5 test source.
        coverage: Coverage analysis.
        petri: Petri net encoding result.
        invariants: Algebraic invariants.
        marking: Marking lattice analysis.
        num_valid_paths: Count of valid execution paths.
        num_violations: Count of protocol violation points.
        num_incomplete_prefixes: Count of incomplete executions.
    """
    protocol: ProtocolModel
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    enumeration: EnumerationResult
    test_source: str
    coverage: CoverageResult
    petri: PetriNetResult
    invariants: AlgebraicInvariants
    marking: MarkingLatticeResult
    num_valid_paths: int
    num_violations: int
    num_incomplete_prefixes: int


# ---------------------------------------------------------------------------
# Annotation tables
# ---------------------------------------------------------------------------

_MCP_ANNOTATIONS: dict[str, tuple[str, str, str]] = {
    # label: (initiator, message_kind, description)
    "initialize": ("client", "request", "Client initializes the MCP session"),
    "callTool": ("client", "request", "Client invokes a tool"),
    "listTools": ("client", "request", "Client lists available tools"),
    "shutdown": ("client", "request", "Client shuts down the session"),
    "RESULT": ("server", "response", "Server returns tool call result"),
    "ERROR": ("server", "response", "Server returns tool call error"),
    "NOTIFICATION": ("server", "notification", "Server sends async notification"),
    "DONE": ("server", "notification", "Server signals notification stream complete"),
}

_A2A_ANNOTATIONS: dict[str, tuple[str, str, str]] = {
    "sendTask": ("client", "request", "Client sends a task to the agent"),
    "getStatus": ("client", "request", "Client polls task status"),
    "cancel": ("client", "request", "Client cancels the task"),
    "getArtifact": ("client", "request", "Client retrieves task artifacts"),
    "WORKING": ("server", "response", "Agent reports task in progress"),
    "COMPLETED": ("server", "response", "Agent reports task completed"),
    "FAILED": ("server", "response", "Agent reports task failed"),
}


def _annotate(
    ss: StateSpace,
    table: dict[str, tuple[str, str, str]],
) -> tuple[TransitionAnnotation, ...]:
    """Build transition annotations from a label lookup table."""
    annotations: list[TransitionAnnotation] = []
    for src, label, tgt in ss.transitions:
        if label in table:
            initiator, kind, desc = table[label]
        else:
            initiator, kind, desc = "unknown", "unknown", f"Transition {label}"
        annotations.append(TransitionAnnotation(
            source=src,
            label=label,
            target=tgt,
            initiator=initiator,
            message_kind=kind,
            description=desc,
        ))
    return tuple(annotations)


# ---------------------------------------------------------------------------
# Protocol model constructors
# ---------------------------------------------------------------------------

def mcp_protocol() -> ProtocolModel:
    """Return the canonical MCP protocol model with annotations."""
    ast = parse(MCP_TYPE_STRING)
    ss = build_statespace(ast)
    return ProtocolModel(
        name="MCP",
        version="2024-11-05",
        type_string=MCP_TYPE_STRING,
        ast=ast,
        description=(
            "Model Context Protocol: AI agent protocol for tool invocation, "
            "resource access, and prompt management. Supports concurrent "
            "tool operations and server-initiated notifications."
        ),
        annotations=_annotate(ss, _MCP_ANNOTATIONS),
    )


def a2a_protocol() -> ProtocolModel:
    """Return the canonical A2A protocol model with annotations."""
    ast = parse(A2A_TYPE_STRING)
    ss = build_statespace(ast)
    return ProtocolModel(
        name="A2A",
        version="1.0",
        type_string=A2A_TYPE_STRING,
        ast=ast,
        description=(
            "Agent-to-Agent Protocol: Google's protocol for inter-agent "
            "task delegation with status polling, cancellation, and "
            "artifact retrieval."
        ),
        annotations=_annotate(ss, _A2A_ANNOTATIONS),
    )


def custom_protocol(
    name: str,
    type_string: str,
    description: str = "",
) -> ProtocolModel:
    """Create a protocol model from a custom type string (no annotations)."""
    ast = parse(type_string)
    return ProtocolModel(
        name=name,
        version="custom",
        type_string=type_string,
        ast=ast,
        description=description or f"Custom protocol: {name}",
        annotations=(),
    )


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def conformance_report(
    protocol: ProtocolModel,
    config: TestGenConfig | None = None,
) -> ConformanceReport:
    """Run the full conformance analysis pipeline.

    Orchestrates all existing reticulate modules to produce a
    comprehensive conformance report.
    """
    if config is None:
        config = TestGenConfig(
            class_name=f"{protocol.name}ProtocolTest",
            package_name="com.agent.conformance",
        )

    # 1. Build state space
    ss = build_statespace(protocol.ast)

    # 2. Lattice analysis
    lr = check_lattice(ss)
    dist = check_distributive(ss)

    # 3. Test generation
    enum = enumerate_tests(ss, config)
    test_src = generate_test_source(ss, config, protocol.type_string)

    # 4. Coverage
    cov = compute_coverage(ss, result=enum)

    # 5. Petri net
    petri = session_type_to_petri_net(ss)

    # 6. Algebraic invariants
    inv = algebraic_invariants(ss)

    # 7. Marking lattice
    ml = analyze_marking_lattice(ss)

    return ConformanceReport(
        protocol=protocol,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        enumeration=enum,
        test_source=test_src,
        coverage=cov,
        petri=petri,
        invariants=inv,
        marking=ml,
        num_valid_paths=len(enum.valid_paths),
        num_violations=len(enum.violations),
        num_incomplete_prefixes=len(enum.incomplete_prefixes),
    )


def generate_conformance_tests(
    protocol: ProtocolModel,
    config: TestGenConfig | None = None,
) -> str:
    """Generate JUnit 5 conformance test source for a protocol."""
    if config is None:
        config = TestGenConfig(
            class_name=f"{protocol.name}ProtocolTest",
            package_name="com.agent.conformance",
        )
    ss = build_statespace(protocol.ast)
    return generate_test_source(ss, config, protocol.type_string)


# ---------------------------------------------------------------------------
# Text report formatter
# ---------------------------------------------------------------------------

def format_report(report: ConformanceReport) -> str:
    """Format a ConformanceReport as structured text for terminal output."""
    lines: list[str] = []
    r = report
    p = r.protocol

    lines.append("=" * 70)
    lines.append(f"  CONFORMANCE REPORT: {p.name} Protocol (v{p.version})")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {p.description}")
    lines.append("")

    # Session type
    lines.append("--- Session Type ---")
    lines.append(f"  {p.type_string}")
    lines.append("")

    # State space
    lines.append("--- State Space ---")
    lines.append(f"  States:      {len(r.state_space.states)}")
    lines.append(f"  Transitions: {len(r.state_space.transitions)}")
    lines.append(f"  Top (init):  {r.state_space.top}")
    lines.append(f"  Bottom (end):{r.state_space.bottom}")
    lines.append("")

    # Lattice
    lattice_str = "YES" if r.lattice_result.is_lattice else "NO"
    dist_str = "yes" if r.distributivity.is_distributive else "no"
    lines.append("--- Lattice Properties ---")
    lines.append(f"  Is lattice:     {lattice_str}")
    lines.append(f"  Has top:        {r.lattice_result.has_top}")
    lines.append(f"  Has bottom:     {r.lattice_result.has_bottom}")
    lines.append(f"  Distributive:   {dist_str}")
    lines.append(f"  SCCs:           {r.lattice_result.num_scc}")
    lines.append("")

    # Test generation
    lines.append("--- Conformance Tests ---")
    lines.append(f"  Valid paths:          {r.num_valid_paths}")
    lines.append(f"  Violation points:     {r.num_violations}")
    lines.append(f"  Incomplete prefixes:  {r.num_incomplete_prefixes}")
    lines.append(f"  Transition coverage:  {r.coverage.transition_coverage:.1%}")
    lines.append(f"  State coverage:       {r.coverage.state_coverage:.1%}")
    lines.append("")

    # Petri net
    lines.append("--- Petri Net ---")
    lines.append(f"  Places:       {r.petri.num_places}")
    lines.append(f"  Transitions:  {r.petri.num_transitions}")
    lines.append(f"  Occurrence:   {'yes' if r.petri.is_occurrence_net else 'no (has cycles)'}")
    lines.append(f"  Free-choice:  {'yes' if r.petri.is_free_choice else 'no'}")
    lines.append(f"  1-safe:       yes")
    lines.append(f"  Reachability isomorphic: {'yes' if r.petri.reachability_isomorphic else 'NO'}")
    lines.append("")

    # Algebraic invariants
    lines.append("--- Algebraic Invariants ---")
    lines.append(f"  Mobius mu(top,bot): {r.invariants.mobius_value}")
    lines.append(f"  Rota polynomial:   {r.invariants.rota_polynomial}")
    lines.append(f"  Spectral radius:   {r.invariants.spectral_radius:.4f}")
    lines.append(f"  Fiedler value:     {r.invariants.fiedler_value:.4f}")
    lines.append(f"  Tropical eigenval: {r.invariants.tropical_eigenvalue:.1f}")
    lines.append(f"  VN entropy:        {r.invariants.von_neumann_entropy:.4f}")
    lines.append(f"  Join-irreducibles: {r.invariants.num_join_irreducibles}")
    lines.append(f"  Meet-irreducibles: {r.invariants.num_meet_irreducibles}")
    lines.append("")

    # Marking lattice
    lines.append("--- Marking Lattice ---")
    lines.append(f"  Width:        {r.marking.width}")
    lines.append(f"  Height:       {r.marking.height}")
    lines.append(f"  Chain count:  {r.marking.chain_count}")
    lines.append(f"  Isomorphic:   {'yes' if r.marking.is_isomorphic_to_statespace else 'NO'}")
    lines.append("")

    # Annotations
    if p.annotations:
        lines.append("--- Transition Annotations ---")
        for ann in p.annotations:
            arrow = ">>>" if ann.initiator == "client" else "<<<"
            lines.append(
                f"  {arrow} {ann.label:20s} "
                f"[{ann.initiator}/{ann.message_kind}] "
                f"{ann.description}"
            )
        lines.append("")

    # Verdict
    lines.append("=" * 70)
    if r.lattice_result.is_lattice and r.petri.reachability_isomorphic:
        lines.append(f"  VERDICT: {p.name} protocol is WELL-FORMED")
        lines.append(f"  The state space forms a bounded lattice.")
        lines.append(f"  {r.num_valid_paths} conformance tests generated.")
        lines.append(f"  {r.num_violations} violation scenarios detected.")
    else:
        lines.append(f"  VERDICT: {p.name} protocol has ISSUES")
        if not r.lattice_result.is_lattice:
            lines.append(f"  WARNING: State space does NOT form a lattice!")
        if not r.petri.reachability_isomorphic:
            lines.append(f"  WARNING: Petri net reachability NOT isomorphic!")
    lines.append("=" * 70)

    return "\n".join(lines)
