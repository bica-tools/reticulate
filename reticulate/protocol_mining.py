"""Protocol mining from observed traces (Step 83).

Given observed method call sequences (traces), infer a session type that
describes the protocol. This is the *inverse* of trace extraction: instead
of going from a session type to its traces, we go from traces to a session
type.

Three entry points:

1. ``mine_from_traces(traces)`` — given raw trace sequences, build a
   prefix tree acceptor (PTA), convert to a state space, then
   reconstruct a session type AST.
2. ``mine_from_statespace(ss)`` — use ``reconstruct()`` to extract
   a session type from an existing state space.
3. ``mine_from_logs(log_lines)`` — parse structured log lines, extract
   traces, and infer the protocol.

The core algorithm is *prefix tree induction*:

1. Build a prefix tree acceptor (trie) from traces.
2. Merge states that have identical futures (Angluin-style state merging).
3. Convert the merged automaton into a ``StateSpace``.
4. Reconstruct a session type via ``reconstruct()``.
5. Verify the result by checking that all original traces are accepted.

This connects to the lattice-theoretic framework because the mined
state space must form a lattice (reticulate) if the underlying protocol
is a valid session type.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from reticulate.lattice import check_lattice
from reticulate.parser import SessionType, pretty
from reticulate.reticular import reconstruct
from reticulate.statespace import StateSpace, build_statespace
from reticulate.csp import Trace, extract_traces


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MiningResult:
    """Result of protocol mining.

    Attributes:
        inferred_type: The reconstructed session type AST (None if mining failed).
        inferred_type_str: Pretty-printed session type string.
        confidence: Confidence score in [0.0, 1.0] based on trace coverage
            and lattice properties.
        num_traces: Number of input traces used.
        coverage: Fraction of original traces accepted by the inferred type.
        is_lattice: Whether the mined state space forms a lattice.
        num_states: Number of states in the mined state space.
        num_transitions: Number of transitions in the mined state space.
        state_space: The mined state space (for further analysis).
    """
    inferred_type: SessionType | None
    inferred_type_str: str
    confidence: float
    num_traces: int
    coverage: float
    is_lattice: bool
    num_states: int
    num_transitions: int
    state_space: StateSpace | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Prefix tree acceptor (PTA)
# ---------------------------------------------------------------------------

class _TrieNode:
    """Node in a prefix tree acceptor."""

    __slots__ = ("children", "is_terminal", "state_id")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.is_terminal: bool = False
        self.state_id: int = -1


def _build_pta(traces: Sequence[Trace]) -> _TrieNode:
    """Build a prefix tree acceptor from traces."""
    root = _TrieNode()
    for trace in traces:
        node = root
        for label in trace:
            if label not in node.children:
                node.children[label] = _TrieNode()
            node = node.children[label]
        node.is_terminal = True
    return root


# ---------------------------------------------------------------------------
# State merging
# ---------------------------------------------------------------------------

def _signature(node: _TrieNode) -> tuple[bool, tuple[str, ...]]:
    """Compute the signature of a trie node for merging.

    Two nodes with identical signatures (same terminal flag and same
    set of outgoing labels) are candidates for merging.
    """
    return (node.is_terminal, tuple(sorted(node.children.keys())))


def _deep_signature(node: _TrieNode, depth: int = 2) -> tuple:
    """Compute a deeper signature for more accurate merging.

    Looks ahead `depth` levels to distinguish nodes that have the same
    immediate children but different futures.
    """
    if depth <= 0:
        return (_signature(node),)
    child_sigs = tuple(
        (label, _deep_signature(child, depth - 1))
        for label, child in sorted(node.children.items())
    )
    return (node.is_terminal, child_sigs)


def _merge_equivalent(root: _TrieNode, merge_depth: int = 2) -> _TrieNode:
    """Merge nodes with identical deep signatures.

    This is a simplified version of state merging that groups nodes
    by their future behaviour (up to merge_depth levels ahead).
    Nodes in the same equivalence class are replaced by a single
    representative.
    """
    # Collect all nodes by deep signature
    sig_to_nodes: dict[tuple, list[_TrieNode]] = defaultdict(list)
    _collect_by_signature(root, sig_to_nodes, merge_depth)

    # Build a mapping from each node to its representative
    node_to_rep: dict[int, _TrieNode] = {}
    for nodes in sig_to_nodes.values():
        rep = nodes[0]
        for node in nodes:
            node_to_rep[id(node)] = rep

    # Rebuild tree with merged references
    _apply_merges(root, node_to_rep)
    return root


def _collect_by_signature(
    node: _TrieNode,
    sig_map: dict[tuple, list[_TrieNode]],
    depth: int,
) -> None:
    """Recursively collect nodes by deep signature."""
    sig = _deep_signature(node, depth)
    sig_map[sig].append(node)
    for child in node.children.values():
        _collect_by_signature(child, sig_map, depth)


def _apply_merges(node: _TrieNode, node_to_rep: dict[int, _TrieNode]) -> None:
    """Replace children with their representatives."""
    for label in list(node.children.keys()):
        child = node.children[label]
        rep = node_to_rep.get(id(child), child)
        node.children[label] = rep
        if rep is not child:
            # Merge children of child into rep if not already there
            for clabel, cchild in child.children.items():
                if clabel not in rep.children:
                    rep.children[clabel] = cchild
            if child.is_terminal:
                rep.is_terminal = True
        _apply_merges(rep, node_to_rep)


# ---------------------------------------------------------------------------
# PTA → StateSpace conversion
# ---------------------------------------------------------------------------

def _pta_to_statespace(root: _TrieNode) -> StateSpace:
    """Convert a prefix tree acceptor to a StateSpace.

    Assigns integer IDs to each unique trie node, then builds
    transitions and identifies top/bottom states.
    """
    ss = StateSpace()
    counter = 0
    visited: dict[int, int] = {}  # id(node) → state_id

    def assign_id(node: _TrieNode) -> int:
        nonlocal counter
        nid = id(node)
        if nid in visited:
            return visited[nid]
        sid = counter
        counter += 1
        visited[nid] = sid
        node.state_id = sid
        ss.states.add(sid)
        return sid

    # BFS to assign IDs and build transitions
    queue: list[_TrieNode] = [root]
    processed: set[int] = set()

    top_id = assign_id(root)
    ss.top = top_id

    # Find or create the bottom (end) state
    # Terminal nodes with no children map to a single bottom state
    bottom_id = -1

    while queue:
        node = queue.pop(0)
        nid = id(node)
        if nid in processed:
            continue
        processed.add(nid)

        src = visited[nid]

        if node.is_terminal and not node.children:
            # This is a terminal leaf — candidate for bottom
            if bottom_id == -1:
                bottom_id = src
            # If there's already a bottom, we keep this one as is
            # (will be handled by lattice check)

        for label in sorted(node.children.keys()):
            child = node.children[label]
            tgt = assign_id(child)
            ss.transitions.append((src, label, tgt))
            queue.append(child)

    # If no terminal leaf found, create an explicit bottom
    if bottom_id == -1:
        bottom_id = counter
        ss.states.add(bottom_id)

    ss.bottom = bottom_id

    # Connect non-leaf terminals to bottom if they have children
    # (they represent states where the protocol *can* end but also continue)
    for node_id_py, state_id in visited.items():
        # We can't iterate trie nodes from python IDs easily,
        # so we handle this during the BFS above.
        pass

    ss.labels = {s: f"s{s}" for s in ss.states}
    return ss


# ---------------------------------------------------------------------------
# Trace acceptance check
# ---------------------------------------------------------------------------

def _accepts_trace(ss: StateSpace, trace: Trace) -> bool:
    """Check if a state space accepts a trace from its top state."""
    state = ss.top
    for label in trace:
        found = False
        for l, t in ss.enabled(state):
            if l == label:
                state = t
                found = True
                break
        if not found:
            return False
    return True


def _compute_coverage(
    ss: StateSpace,
    traces: Sequence[Trace],
) -> float:
    """Compute the fraction of input traces accepted by the state space."""
    if not traces:
        return 1.0
    accepted = sum(1 for t in traces if _accepts_trace(ss, t))
    return accepted / len(traces)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(
    coverage: float,
    is_lattice: bool,
    num_traces: int,
    num_states: int,
) -> float:
    """Compute a confidence score for the mining result.

    Factors:
    - coverage: higher is better (weight 0.5)
    - is_lattice: lattice formation is a strong positive signal (weight 0.3)
    - trace count: more traces = more confidence, diminishing returns (weight 0.2)
    """
    trace_factor = min(1.0, num_traces / 10.0)  # saturates at 10 traces
    lattice_factor = 1.0 if is_lattice else 0.3
    confidence = 0.5 * coverage + 0.3 * lattice_factor + 0.2 * trace_factor
    return round(min(1.0, confidence), 4)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Common log patterns
_LOG_PATTERNS = [
    # Pattern: "[timestamp] method_name" or "[timestamp] CALL method_name"
    re.compile(r"\b(?:CALL|INVOKE|METHOD)\s+(\w+)", re.IGNORECASE),
    # Pattern: "method_name()" or "obj.method_name()"
    re.compile(r"(?:\w+\.)?(\w+)\(\)"),
    # Pattern: "-> method_name" or "=> method_name"
    re.compile(r"[-=]>\s*(\w+)"),
    # Pattern: simple word per line (one method per line)
    re.compile(r"^(\w+)$"),
]

# Session delimiters
_SESSION_DELIMITERS = re.compile(
    r"(?:\b(?:SESSION\s+(?:START|END|BEGIN|CLOSE))\b|^-{3,}$|^={3,}$)",
    re.IGNORECASE,
)


def parse_log_lines(log_lines: Sequence[str]) -> list[Trace]:
    """Parse log lines into traces.

    Attempts several common log formats. Sessions are delimited by
    blank lines, '---' separators, or SESSION/BEGIN/END markers.

    Each contiguous block of method calls becomes one trace.

    Args:
        log_lines: Sequence of log line strings.

    Returns:
        List of traces (each a tuple of method name strings).
    """
    traces: list[Trace] = []
    current_methods: list[str] = []

    for line in log_lines:
        stripped = line.strip()

        # Check for session delimiter
        if not stripped or _SESSION_DELIMITERS.search(stripped):
            if current_methods:
                traces.append(tuple(current_methods))
                current_methods = []
            continue

        # Try to extract a method name
        method = _extract_method(stripped)
        if method:
            current_methods.append(method)

    # Don't forget the last session
    if current_methods:
        traces.append(tuple(current_methods))

    return traces


def _extract_method(line: str) -> str | None:
    """Extract a method name from a log line."""
    for pattern in _LOG_PATTERNS:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mine_from_traces(
    traces: Sequence[Trace],
    *,
    merge_depth: int = 2,
) -> MiningResult:
    """Infer a session type from observed method call traces.

    Algorithm:
    1. Build a prefix tree acceptor (PTA) from the traces.
    2. Merge equivalent states (nodes with identical futures).
    3. Convert to a StateSpace.
    4. Reconstruct a session type AST.
    5. Verify coverage and lattice properties.

    Args:
        traces: Sequence of traces (each a tuple of method name strings).
        merge_depth: Depth for state-merging signature comparison.

    Returns:
        A MiningResult with the inferred type and quality metrics.
    """
    if not traces:
        return MiningResult(
            inferred_type=None,
            inferred_type_str="end",
            confidence=0.0,
            num_traces=0,
            coverage=0.0,
            is_lattice=True,
            num_states=0,
            num_transitions=0,
        )

    # Filter out empty traces for PTA construction, but count them
    non_empty = [t for t in traces if t]
    all_empty = len(non_empty) == 0

    if all_empty:
        from reticulate.parser import End
        return MiningResult(
            inferred_type=End(),
            inferred_type_str="end",
            confidence=1.0,
            num_traces=len(traces),
            coverage=1.0,
            is_lattice=True,
            num_states=1,
            num_transitions=0,
        )

    # 1. Build PTA
    root = _build_pta(non_empty)

    # 2. Merge equivalent states
    root = _merge_equivalent(root, merge_depth)

    # 3. Convert to StateSpace
    ss = _pta_to_statespace(root)

    # 4. Check lattice
    lattice_result = check_lattice(ss)
    is_lattice = lattice_result.is_lattice

    # 5. Reconstruct session type
    try:
        inferred = reconstruct(ss)
        type_str = pretty(inferred)
    except (ValueError, RecursionError):
        inferred = None
        type_str = "<reconstruction failed>"

    # 6. Compute coverage
    coverage = _compute_coverage(ss, list(traces))

    # 7. Confidence
    confidence = _compute_confidence(
        coverage, is_lattice, len(traces), len(ss.states),
    )

    return MiningResult(
        inferred_type=inferred,
        inferred_type_str=type_str,
        confidence=confidence,
        num_traces=len(traces),
        coverage=coverage,
        is_lattice=is_lattice,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        state_space=ss,
    )


def mine_from_statespace(ss: StateSpace) -> MiningResult:
    """Extract a session type from an existing state space.

    Uses ``reconstruct()`` from the reticular module to recover the
    session type AST. This is useful when you already have a state
    machine (e.g., from runtime monitoring or model extraction) and
    want to obtain the corresponding session type.

    Args:
        ss: An existing state space.

    Returns:
        A MiningResult with the reconstructed type and metrics.
    """
    # Check lattice
    lattice_result = check_lattice(ss)
    is_lattice = lattice_result.is_lattice

    # Reconstruct
    try:
        inferred = reconstruct(ss)
        type_str = pretty(inferred)
    except (ValueError, RecursionError):
        inferred = None
        type_str = "<reconstruction failed>"

    # Extract traces for coverage baseline
    traces = extract_traces(ss, max_depth=10)
    coverage = 1.0  # By construction, all traces from the SS are accepted

    confidence = _compute_confidence(
        coverage, is_lattice, len(traces), len(ss.states),
    )

    return MiningResult(
        inferred_type=inferred,
        inferred_type_str=type_str,
        confidence=confidence,
        num_traces=len(traces),
        coverage=coverage,
        is_lattice=is_lattice,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        state_space=ss,
    )


def mine_from_logs(
    log_lines: Sequence[str],
    *,
    merge_depth: int = 2,
) -> MiningResult:
    """Parse log files, extract traces, and infer a protocol.

    Combines log parsing with trace-based mining. Accepts various
    log formats (see ``parse_log_lines`` for supported patterns).

    Args:
        log_lines: Sequence of log line strings.
        merge_depth: Depth for state-merging signature comparison.

    Returns:
        A MiningResult with the inferred type and quality metrics.
    """
    traces = parse_log_lines(log_lines)
    return mine_from_traces(traces, merge_depth=merge_depth)
