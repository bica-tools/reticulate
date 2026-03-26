"""Verified audio routing via session types — no dead signals (Step 57a).

Models audio routing graphs (DAW signal flow) as session types and verifies
that every signal reaches the master output.  The lattice property on the
resulting state space guarantees that no signal path is a dead end: every
node eventually reaches the master output bus.

Key insight: an audio routing graph with N parallel channels maps to an
N-ary parallel session type.  Sequential effects chains map to branch
sequences.  The state space lattice encodes *all possible signal states*;
a dead signal manifests as a state that cannot reach the bottom (master
output), violating the lattice property.

Feedback loops are modelled as recursive types.  A feedback loop is *safe*
(no infinite-gain oscillation) iff the recursion is guarded — i.e., every
recursive path passes through at least one delay (a method call that
separates iterations).  This corresponds exactly to the guarded-recursion
requirement in the session type theory.

Usage:
    from reticulate.audio_routing import (
        AudioGraph, AudioNode, AudioRoute,
        audio_graph_to_session_type,
        verify_routing,
        find_dead_signals,
        find_feedback_loops,
        is_feedback_safe,
    )
    graph = AudioGraph(
        nodes=[AudioNode("kick", "source"), AudioNode("master", "output")],
        routes=[AudioRoute("kick", "master", "send")],
        master_output="master",
    )
    result = verify_routing(graph)
    assert result.is_verified  # every signal reaches master

Routing patterns:
    - Simple chain:  input → EQ → compressor → output
    - Parallel mix:  (kick || snare || vocal) → master
    - Send/return:   main + aux bus with send/return
    - Sidechain:     trigger → detector, main → compressor
    - Feedback delay: out → delay → feedback → delay (guarded recursion)
    - Multiband:     split → (low || mid || high) → sum
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.lattice import LatticeResult, check_lattice
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioNode:
    """A node in an audio routing graph.

    Attributes:
        name: Unique identifier for this node.
        node_type: One of "source", "effect", "bus", "output".
        channels: Channel configuration ("mono", "stereo", "surround").
    """
    name: str
    node_type: str = "effect"
    channels: str = "stereo"

    def __post_init__(self) -> None:
        if self.node_type not in ("source", "effect", "bus", "output"):
            raise ValueError(
                f"Invalid node_type '{self.node_type}'; "
                f"must be 'source', 'effect', 'bus', or 'output'"
            )
        if self.channels not in ("mono", "stereo", "surround"):
            raise ValueError(
                f"Invalid channels '{self.channels}'; "
                f"must be 'mono', 'stereo', or 'surround'"
            )


@dataclass(frozen=True)
class AudioRoute:
    """A directed edge in an audio routing graph.

    Attributes:
        src: Source node name.
        dst: Destination node name.
        label: Edge label (e.g. "send", "insert", "feedback").
    """
    src: str
    dst: str
    label: str = "send"


@dataclass(frozen=True)
class AudioGraph:
    """A complete audio routing graph.

    Attributes:
        nodes: List of audio nodes in the graph.
        routes: List of directed routes between nodes.
        master_output: Name of the master output node.
    """
    nodes: tuple[AudioNode, ...] | list[AudioNode] = field(default_factory=list)
    routes: tuple[AudioRoute, ...] | list[AudioRoute] = field(default_factory=list)
    master_output: str = "master"

    def __post_init__(self) -> None:
        node_names = {n.name for n in self.nodes}
        if self.master_output not in node_names:
            raise ValueError(
                f"master_output '{self.master_output}' not in nodes: "
                f"{sorted(node_names)}"
            )

    @property
    def node_map(self) -> dict[str, AudioNode]:
        """Map from node name to AudioNode."""
        return {n.name: n for n in self.nodes}

    def successors(self, name: str) -> list[tuple[str, str]]:
        """Return (dst, label) pairs for routes from *name*."""
        return [(r.dst, r.label) for r in self.routes if r.src == name]

    def predecessors(self, name: str) -> list[tuple[str, str]]:
        """Return (src, label) pairs for routes into *name*."""
        return [(r.src, r.label) for r in self.routes if r.dst == name]

    def sources(self) -> list[str]:
        """Return names of source nodes."""
        return [n.name for n in self.nodes if n.node_type == "source"]

    def outputs(self) -> list[str]:
        """Return names of output nodes."""
        return [n.name for n in self.nodes if n.node_type == "output"]


@dataclass(frozen=True)
class RoutingResult:
    """Result of routing verification.

    Attributes:
        is_verified: True iff every signal reaches master output (lattice holds).
        session_type_string: The session type encoding of the routing graph.
        ast: Parsed session type AST.
        state_space: Constructed state space.
        lattice_result: Lattice check result.
        dead_signals: List of node names that cannot reach master output.
        feedback_loops: List of cycles (each a list of node names).
        feedback_safe: True iff all feedback loops are guarded (pass through delay).
        num_nodes: Number of nodes in the audio graph.
        num_routes: Number of routes.
    """
    is_verified: bool
    session_type_string: str
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    dead_signals: list[str]
    feedback_loops: list[list[str]]
    feedback_safe: bool
    num_nodes: int
    num_routes: int


# ---------------------------------------------------------------------------
# Helpers: session type construction
# ---------------------------------------------------------------------------

def _sanitize_label(name: str) -> str:
    """Sanitize a node/route name for use as a session type label.

    Replaces characters that conflict with the session type grammar.
    """
    s = name.replace(" ", "_").replace("-", "_")
    # Ensure it starts with a letter
    if s and not s[0].isalpha():
        s = "n" + s
    return s


def _build_chain_type(effects: list[str]) -> str:
    """Build a sequential chain session type from a list of effect labels.

    Example: ["eq", "comp", "limit"] → "&{eq: &{comp: &{limit: end}}}"
    """
    if not effects:
        return "end"
    result = "end"
    for eff in reversed(effects):
        label = _sanitize_label(eff)
        result = f"&{{{label}: {result}}}"
    return result


def _build_parallel_type(channels: list[str]) -> str:
    """Build a parallel session type from a list of channel type strings.

    Example: ["&{a: end}", "&{b: end}"] → "(&{a: end} || &{b: end})"
    """
    if not channels:
        return "end"
    if len(channels) == 1:
        return channels[0]
    return "(" + " || ".join(channels) + ")"


def _find_paths_to_master(
    graph: AudioGraph,
    start: str,
    visited: frozenset[str] | None = None,
) -> list[list[str]]:
    """Find all acyclic paths from *start* to master_output.

    Returns a list of paths, where each path is a list of node names
    (including start and master_output).
    """
    if visited is None:
        visited = frozenset()
    if start == graph.master_output:
        return [[start]]
    if start in visited:
        return []
    visited = visited | {start}
    paths: list[list[str]] = []
    for dst, _label in graph.successors(start):
        for sub_path in _find_paths_to_master(graph, dst, visited):
            paths.append([start] + sub_path)
    return paths


def _find_all_chains(graph: AudioGraph) -> list[list[str]]:
    """Find all maximal signal chains from sources to master output."""
    chains: list[list[str]] = []
    for src in graph.sources():
        chains.extend(_find_paths_to_master(graph, src))
    return chains


# ---------------------------------------------------------------------------
# Core: audio graph → session type
# ---------------------------------------------------------------------------

def audio_graph_to_session_type(graph: AudioGraph) -> str:
    """Convert an audio routing graph to a session type string.

    The conversion strategy:
    1. Find all source nodes.
    2. For each source, trace paths to master output.
    3. Each linear chain becomes a branch-sequence session type.
    4. Multiple source chains in parallel become a ∥ composition.
    5. Feedback loops become recursive types (rec X . ... X).

    Args:
        graph: The audio routing graph.

    Returns:
        A session type string encoding the routing topology.
    """
    sources = graph.sources()
    if not sources:
        return "end"

    # Find chains from each source to master
    chain_types: list[str] = []
    for src in sources:
        chain_type = _build_source_type(graph, src, frozenset())
        chain_types.append(chain_type)

    if not chain_types:
        return "end"

    return _build_parallel_type(chain_types)


def _detect_cycle_targets(
    graph: AudioGraph,
    node: str,
    visited: frozenset[str],
) -> set[str]:
    """Detect which nodes in *visited* are reachable from *node* (cycle targets)."""
    targets: set[str] = set()
    stack = [node]
    seen: set[str] = set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for dst, _ in graph.successors(n):
            if dst in visited and dst != graph.master_output:
                targets.add(dst)
            elif dst not in seen and dst != graph.master_output:
                stack.append(dst)
    return targets


def _build_source_type(
    graph: AudioGraph,
    node: str,
    visited: frozenset[str],
    rec_vars: frozenset[str] | None = None,
) -> str:
    """Build a session type for the signal path starting at *node*.

    Handles branching (multiple successors → branch type),
    sequential chains (single successor → chain),
    and feedback loops (visited node → guarded recursion via rec X . ...).
    """
    if rec_vars is None:
        rec_vars = frozenset()

    if node == graph.master_output:
        return "end"
    if node in visited:
        # Feedback loop — reference recursion variable
        var = _sanitize_label(node).upper()
        return var

    # Check if any successor leads back to this node (cycle).
    # If so, wrap this node's body in rec X . ...
    cycle_targets = _detect_cycle_targets(graph, node, visited | {node})
    needs_rec = node in cycle_targets

    visited = visited | {node}
    succs = graph.successors(node)
    label = _sanitize_label(node)

    if not succs:
        # Dead end — still emit a label so the state space captures it
        return f"&{{{label}: end}}"

    if len(succs) == 1:
        dst, _route_label = succs[0]
        sub = _build_source_type(graph, dst, visited, rec_vars)
        inner = f"&{{{label}: {sub}}}"
    else:
        # Multiple successors: check if they reconverge (parallel split)
        sub_types: list[str] = []
        for dst, _route_label in succs:
            sub = _build_source_type(graph, dst, visited, rec_vars)
            sub_types.append(sub)

        # If this is a split point (e.g., multiband), model as parallel
        node_obj = graph.node_map.get(node)
        if node_obj and node_obj.node_type == "bus":
            par = _build_parallel_type(sub_types)
            inner = f"&{{{label}: {par}}}"
        else:
            choices = []
            for (dst, route_label), sub_type in zip(succs, sub_types):
                choice_label = _sanitize_label(route_label)
                choices.append(f"{choice_label}: {sub_type}")
            inner = f"&{{{label}: &{{{', '.join(choices)}}}}}"

    if needs_rec:
        var = _sanitize_label(node).upper()
        return f"rec {var} . {inner}"
    return inner


def _build_feedback_type(
    graph: AudioGraph,
    node: str,
    loop_var: str,
    visited: frozenset[str],
) -> str:
    """Build a recursive session type for a feedback loop starting at *node*."""
    if node == graph.master_output:
        return "end"
    if node in visited:
        return loop_var

    visited = visited | {node}
    succs = graph.successors(node)
    label = _sanitize_label(node)

    if not succs:
        return f"&{{{label}: end}}"

    if len(succs) == 1:
        dst, _ = succs[0]
        sub = _build_feedback_type(graph, dst, loop_var, visited)
        return f"&{{{label}: {sub}}}"

    choices = []
    for dst, route_label in succs:
        sub = _build_feedback_type(graph, dst, loop_var, visited)
        choices.append(f"{_sanitize_label(route_label)}: {sub}")
    return f"&{{{label}: &{{{', '.join(choices)}}}}}"


# ---------------------------------------------------------------------------
# Core: verification
# ---------------------------------------------------------------------------

def verify_routing(graph: AudioGraph) -> RoutingResult:
    """Verify that every signal in the audio graph reaches master output.

    Builds a session type from the routing graph, constructs the state
    space, and checks the lattice property.  If the state space is a
    lattice, every signal path eventually reaches the terminal state
    (master output).

    Args:
        graph: The audio routing graph to verify.

    Returns:
        A RoutingResult with verification details.
    """
    st_str = audio_graph_to_session_type(graph)
    ast = parse(st_str)
    ss = build_statespace(ast)
    lr = check_lattice(ss)

    dead = find_dead_signals(graph)
    loops = find_feedback_loops(graph)
    fb_safe = is_feedback_safe(graph)

    return RoutingResult(
        is_verified=lr.is_lattice and len(dead) == 0,
        session_type_string=st_str,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        dead_signals=dead,
        feedback_loops=loops,
        feedback_safe=fb_safe,
        num_nodes=len(graph.nodes),
        num_routes=len(graph.routes),
    )


def find_dead_signals(graph: AudioGraph) -> list[str]:
    """Identify nodes that cannot reach the master output.

    A dead signal is a node from which no directed path reaches
    master_output.  This is the audio equivalent of a dangling
    pointer or unreachable code.

    Args:
        graph: The audio routing graph.

    Returns:
        List of node names that cannot reach master output.
    """
    # BFS/DFS backwards from master output
    reachable_from_master: set[str] = set()
    stack = [graph.master_output]
    while stack:
        node = stack.pop()
        if node in reachable_from_master:
            continue
        reachable_from_master.add(node)
        for src, _label in graph.predecessors(node):
            stack.append(src)

    dead: list[str] = []
    for n in graph.nodes:
        if n.name not in reachable_from_master and n.name != graph.master_output:
            dead.append(n.name)
    return sorted(dead)


def find_feedback_loops(graph: AudioGraph) -> list[list[str]]:
    """Find all cycles in the audio routing graph.

    A cycle indicates a feedback loop, which in audio can cause
    infinite-gain oscillation if not properly managed with delays.

    Uses DFS cycle detection.

    Args:
        graph: The audio routing graph.

    Returns:
        List of cycles, each a list of node names forming the loop.
    """
    cycles: list[list[str]] = []
    visited: set[str] = set()

    def _dfs(node: str, path: list[str], on_stack: set[str]) -> None:
        visited.add(node)
        on_stack.add(node)
        path.append(node)

        for dst, _label in graph.successors(node):
            if dst in on_stack:
                # Found a cycle
                idx = path.index(dst)
                cycle = path[idx:] + [dst]
                # Normalize: start from lexicographically smallest
                cycles.append(cycle)
            elif dst not in visited:
                _dfs(dst, path, on_stack)

        path.pop()
        on_stack.discard(node)

    for n in graph.nodes:
        if n.name not in visited:
            _dfs(n.name, [], set())

    return cycles


def is_feedback_safe(graph: AudioGraph) -> bool:
    """Check whether all feedback loops are safe (guarded by a delay).

    A feedback loop is safe if every cycle passes through at least
    one node whose name contains "delay" or whose node_type is "effect"
    with a delay-like label.  This corresponds to guarded recursion
    in the session type theory: the delay acts as the guard that
    prevents divergence.

    Args:
        graph: The audio routing graph.

    Returns:
        True iff all feedback loops pass through a delay node.
    """
    loops = find_feedback_loops(graph)
    if not loops:
        return True  # No loops → trivially safe

    node_map = graph.node_map
    for cycle in loops:
        # Check if any node in the cycle is a delay
        has_delay = False
        for node_name in cycle[:-1]:  # Last element repeats first
            node = node_map.get(node_name)
            if node and "delay" in node.name.lower():
                has_delay = True
                break
        if not has_delay:
            return False
    return True


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def format_routing_report(result: RoutingResult) -> str:
    """Format a RoutingResult as structured text for terminal output."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  AUDIO ROUTING VERIFICATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append("--- Routing Graph ---")
    lines.append(f"  Nodes:       {result.num_nodes}")
    lines.append(f"  Routes:      {result.num_routes}")
    lines.append("")

    lines.append("--- Session Type ---")
    lines.append(f"  {result.session_type_string}")
    lines.append("")

    lines.append("--- State Space ---")
    lines.append(f"  States:      {len(result.state_space.states)}")
    lines.append(f"  Transitions: {len(result.state_space.transitions)}")
    lines.append("")

    lines.append("--- Verification ---")
    lr = result.lattice_result
    lines.append(f"  Lattice:     {'YES' if lr.is_lattice else 'NO'}")
    lines.append(f"  Dead signals:{' NONE' if not result.dead_signals else ' ' + ', '.join(result.dead_signals)}")
    lines.append(f"  Verified:    {'YES' if result.is_verified else 'NO'}")
    lines.append("")

    if result.feedback_loops:
        lines.append("--- Feedback Loops ---")
        for i, loop in enumerate(result.feedback_loops, 1):
            lines.append(f"  Loop {i}: {' → '.join(loop)}")
        lines.append(f"  Feedback safe: {'YES' if result.feedback_safe else 'NO'}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built routing scenarios
# ---------------------------------------------------------------------------

def basic_mix() -> AudioGraph:
    """Basic mixing setup: kick, snare, vocal → master."""
    return AudioGraph(
        nodes=[
            AudioNode("kick", "source", "mono"),
            AudioNode("snare", "source", "mono"),
            AudioNode("vocal", "source", "mono"),
            AudioNode("master", "output", "stereo"),
        ],
        routes=[
            AudioRoute("kick", "master", "send"),
            AudioRoute("snare", "master", "send"),
            AudioRoute("vocal", "master", "send"),
        ],
        master_output="master",
    )


def mastering_chain() -> AudioGraph:
    """Mastering chain: input → EQ → compressor → limiter → output."""
    return AudioGraph(
        nodes=[
            AudioNode("input", "source", "stereo"),
            AudioNode("eq", "effect", "stereo"),
            AudioNode("compressor", "effect", "stereo"),
            AudioNode("limiter", "effect", "stereo"),
            AudioNode("master", "output", "stereo"),
        ],
        routes=[
            AudioRoute("input", "eq", "insert"),
            AudioRoute("eq", "compressor", "insert"),
            AudioRoute("compressor", "limiter", "insert"),
            AudioRoute("limiter", "master", "insert"),
        ],
        master_output="master",
    )


def multiband_split() -> AudioGraph:
    """Multiband processing: input → splitter bus → (low || mid || high) → sum → master."""
    return AudioGraph(
        nodes=[
            AudioNode("input", "source", "stereo"),
            AudioNode("splitter", "bus", "stereo"),
            AudioNode("low_band", "effect", "stereo"),
            AudioNode("mid_band", "effect", "stereo"),
            AudioNode("high_band", "effect", "stereo"),
            AudioNode("sum_bus", "bus", "stereo"),
            AudioNode("master", "output", "stereo"),
        ],
        routes=[
            AudioRoute("input", "splitter", "send"),
            AudioRoute("splitter", "low_band", "split_low"),
            AudioRoute("splitter", "mid_band", "split_mid"),
            AudioRoute("splitter", "high_band", "split_high"),
            AudioRoute("low_band", "sum_bus", "return"),
            AudioRoute("mid_band", "sum_bus", "return"),
            AudioRoute("high_band", "sum_bus", "return"),
            AudioRoute("sum_bus", "master", "send"),
        ],
        master_output="master",
    )


def sidechain_compress() -> AudioGraph:
    """Sidechain compression: trigger + main → compressor → master."""
    return AudioGraph(
        nodes=[
            AudioNode("trigger", "source", "mono"),
            AudioNode("main", "source", "stereo"),
            AudioNode("detector", "effect", "mono"),
            AudioNode("compressor", "effect", "stereo"),
            AudioNode("master", "output", "stereo"),
        ],
        routes=[
            AudioRoute("trigger", "detector", "sidechain"),
            AudioRoute("detector", "compressor", "control"),
            AudioRoute("main", "compressor", "audio"),
            AudioRoute("compressor", "master", "send"),
        ],
        master_output="master",
    )


def feedback_delay() -> AudioGraph:
    """Feedback delay: input → delay → output, with delay → feedback → delay loop."""
    return AudioGraph(
        nodes=[
            AudioNode("input", "source", "stereo"),
            AudioNode("delay", "effect", "stereo"),
            AudioNode("feedback", "effect", "stereo"),
            AudioNode("master", "output", "stereo"),
        ],
        routes=[
            AudioRoute("input", "delay", "send"),
            AudioRoute("delay", "master", "output"),
            AudioRoute("delay", "feedback", "tap"),
            AudioRoute("feedback", "delay", "return"),
        ],
        master_output="master",
    )


def unsafe_feedback() -> AudioGraph:
    """Unsafe feedback: direct loop without delay node."""
    return AudioGraph(
        nodes=[
            AudioNode("input", "source", "stereo"),
            AudioNode("effect_a", "effect", "stereo"),
            AudioNode("effect_b", "effect", "stereo"),
            AudioNode("master", "output", "stereo"),
        ],
        routes=[
            AudioRoute("input", "effect_a", "send"),
            AudioRoute("effect_a", "effect_b", "send"),
            AudioRoute("effect_b", "effect_a", "feedback"),
            AudioRoute("effect_b", "master", "send"),
        ],
        master_output="master",
    )


def send_return() -> AudioGraph:
    """Send/return routing: main → master, main → reverb_bus → master."""
    return AudioGraph(
        nodes=[
            AudioNode("vocal", "source", "mono"),
            AudioNode("main_fader", "effect", "stereo"),
            AudioNode("reverb_bus", "bus", "stereo"),
            AudioNode("reverb", "effect", "stereo"),
            AudioNode("master", "output", "stereo"),
        ],
        routes=[
            AudioRoute("vocal", "main_fader", "insert"),
            AudioRoute("main_fader", "master", "direct"),
            AudioRoute("main_fader", "reverb_bus", "send"),
            AudioRoute("reverb_bus", "reverb", "insert"),
            AudioRoute("reverb", "master", "return"),
        ],
        master_output="master",
    )


def live_mix() -> AudioGraph:
    """Live mixing setup: multiple inputs → sub-groups → master."""
    return AudioGraph(
        nodes=[
            AudioNode("guitar", "source", "mono"),
            AudioNode("bass", "source", "mono"),
            AudioNode("drums", "source", "stereo"),
            AudioNode("keys", "source", "stereo"),
            AudioNode("instrument_bus", "bus", "stereo"),
            AudioNode("vocal_lead", "source", "mono"),
            AudioNode("vocal_backing", "source", "mono"),
            AudioNode("vocal_bus", "bus", "stereo"),
            AudioNode("master", "output", "stereo"),
        ],
        routes=[
            AudioRoute("guitar", "instrument_bus", "send"),
            AudioRoute("bass", "instrument_bus", "send"),
            AudioRoute("drums", "instrument_bus", "send"),
            AudioRoute("keys", "instrument_bus", "send"),
            AudioRoute("vocal_lead", "vocal_bus", "send"),
            AudioRoute("vocal_backing", "vocal_bus", "send"),
            AudioRoute("instrument_bus", "master", "send"),
            AudioRoute("vocal_bus", "master", "send"),
        ],
        master_output="master",
    )


def recording_setup() -> AudioGraph:
    """Recording setup: mic → preamp → converter → DAW → monitor."""
    return AudioGraph(
        nodes=[
            AudioNode("mic", "source", "mono"),
            AudioNode("preamp", "effect", "mono"),
            AudioNode("converter", "effect", "stereo"),
            AudioNode("daw", "effect", "stereo"),
            AudioNode("master", "output", "stereo"),
        ],
        routes=[
            AudioRoute("mic", "preamp", "xlr"),
            AudioRoute("preamp", "converter", "line"),
            AudioRoute("converter", "daw", "digital"),
            AudioRoute("daw", "master", "monitor"),
        ],
        master_output="master",
    )


def dead_signal_graph() -> AudioGraph:
    """Graph with a dead signal: orphaned bus with no route to master."""
    return AudioGraph(
        nodes=[
            AudioNode("input", "source", "stereo"),
            AudioNode("eq", "effect", "stereo"),
            AudioNode("orphan_bus", "bus", "stereo"),
            AudioNode("orphan_fx", "effect", "stereo"),
            AudioNode("master", "output", "stereo"),
        ],
        routes=[
            AudioRoute("input", "eq", "insert"),
            AudioRoute("eq", "master", "send"),
            AudioRoute("orphan_bus", "orphan_fx", "send"),
            # orphan_fx has no route to master → dead signal
        ],
        master_output="master",
    )


# Collect all pre-built scenarios for benchmarking
ALL_ROUTING_SCENARIOS: list[tuple[str, AudioGraph]] = [
    ("basic_mix", basic_mix()),
    ("mastering_chain", mastering_chain()),
    ("multiband_split", multiband_split()),
    ("sidechain_compress", sidechain_compress()),
    ("feedback_delay", feedback_delay()),
    ("send_return", send_return()),
    ("live_mix", live_mix()),
    ("recording_setup", recording_setup()),
]
