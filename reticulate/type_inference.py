"""Session type inference from traces (Step 97).

Given observed method call traces, infer a session type AST that accepts
all observed traces.  The algorithm:

1. Build a prefix tree (trie) from the observed traces.
2. Detect loops: repeated subsequences that suggest recursion.
3. Merge compatible states: states with identical futures are collapsed.
4. Convert the merged prefix tree to a session type AST.
5. Validate: every original trace must be accepted by the inferred type.

Directions:
- "send" labels become Select choices (internal choice — we choose).
- "receive" labels become Branch choices (external choice — environment chooses).
- If all labels at a node share the same direction, the node is pure
  Branch or pure Select.  Mixed directions at the same node are allowed
  (the dominant direction wins, or we split).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from reticulate.parser import (
    Branch,
    End,
    Rec,
    Select,
    SessionType,
    Var,
    parse,
    pretty,
)
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TraceStep:
    """A single step in a trace: a method label and a direction.

    Attributes:
        label: The method/message name.
        direction: Either "send" or "receive".
    """
    label: str
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in ("send", "receive"):
            raise ValueError(
                f"direction must be 'send' or 'receive', got {self.direction!r}"
            )


@dataclass(frozen=True)
class Trace:
    """An observed trace: a sequence of method call steps.

    Attributes:
        steps: Tuple of TraceStep values.
    """
    steps: tuple[TraceStep, ...]

    @staticmethod
    def from_labels(labels: list[str], direction: str = "receive") -> "Trace":
        """Create a trace from plain labels with uniform direction."""
        return Trace(tuple(TraceStep(l, direction) for l in labels))

    @staticmethod
    def from_pairs(pairs: list[tuple[str, str]]) -> "Trace":
        """Create a trace from (label, direction) pairs."""
        return Trace(tuple(TraceStep(l, d) for l, d in pairs))

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int | slice) -> "TraceStep | tuple[TraceStep, ...]":
        return self.steps[idx]


@dataclass(frozen=True)
class InferenceResult:
    """Result of type inference from traces.

    Attributes:
        inferred: The inferred session type AST.
        pretty_type: Pretty-printed form.
        num_traces: Number of input traces.
        num_states: Number of states in prefix tree before merging.
        num_merged_states: Number of states after merging.
        has_recursion: Whether the inferred type contains recursion.
        all_traces_valid: Whether all input traces pass validation.
        confidence: Confidence score (0.0 to 1.0).
    """
    inferred: SessionType
    pretty_type: str
    num_traces: int
    num_states: int
    num_merged_states: int
    has_recursion: bool
    all_traces_valid: bool
    confidence: float


# ---------------------------------------------------------------------------
# Prefix tree
# ---------------------------------------------------------------------------

class PrefixNode:
    """A node in the prefix tree (trie) built from traces.

    Each node has children keyed by (label, direction) pairs.
    A node is terminal if at least one trace ends here.
    """

    def __init__(self, node_id: int = 0) -> None:
        self.node_id: int = node_id
        self.children: dict[tuple[str, str], PrefixNode] = {}
        self.is_terminal: bool = False
        self.count: int = 0  # how many traces pass through this node

    def __repr__(self) -> str:
        kids = list(self.children.keys())
        return f"PrefixNode(id={self.node_id}, terminal={self.is_terminal}, children={kids})"


class PrefixTree:
    """A prefix tree (trie) built from observed traces."""

    def __init__(self) -> None:
        self._next_id = 0
        self.root = self._new_node()

    def _new_node(self) -> PrefixNode:
        node = PrefixNode(self._next_id)
        self._next_id += 1
        return node

    @property
    def num_nodes(self) -> int:
        return self._next_id

    def insert(self, trace: Trace) -> None:
        """Insert a trace into the prefix tree."""
        node = self.root
        node.count += 1
        for step in trace.steps:
            key = (step.label, step.direction)
            if key not in node.children:
                node.children[key] = self._new_node()
            node = node.children[key]
            node.count += 1
        node.is_terminal = True

    def all_nodes(self) -> list[PrefixNode]:
        """Return all nodes in BFS order."""
        result: list[PrefixNode] = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in node.children.values():
                queue.append(child)
        return result


def build_prefix_tree(traces: list[Trace]) -> PrefixTree:
    """Build a prefix tree from a list of traces.

    Args:
        traces: List of observed traces.

    Returns:
        A PrefixTree containing all traces.

    Raises:
        ValueError: If traces is empty.
    """
    if not traces:
        raise ValueError("Cannot build prefix tree from empty trace list")
    tree = PrefixTree()
    for trace in traces:
        tree.insert(trace)
    return tree


# ---------------------------------------------------------------------------
# Signature-based state merging
# ---------------------------------------------------------------------------

def _node_signature(node: PrefixNode, depth: int = 0, max_depth: int = 10) -> object:
    """Compute a hashable signature for a prefix-tree node.

    Two nodes with the same signature have the same future behavior
    (same set of children with the same labels and recursively same
    continuations), so they can be merged.
    """
    if depth >= max_depth:
        # Truncate at max depth to avoid infinite recursion in loops
        child_keys = tuple(sorted(node.children.keys()))
        return ("TRUNC", child_keys, node.is_terminal)

    child_sigs: list[tuple[tuple[str, str], object]] = []
    for key in sorted(node.children.keys()):
        child = node.children[key]
        child_sigs.append((key, _node_signature(child, depth + 1, max_depth)))
    return ("NODE", tuple(child_sigs), node.is_terminal)


def merge_states(tree: PrefixTree) -> dict[int, int]:
    """Identify equivalent nodes in the prefix tree for merging.

    Returns a mapping from node_id to canonical node_id.
    Nodes with identical signatures (same future behavior) are mapped
    to the same canonical representative.
    """
    nodes = tree.all_nodes()
    sig_to_canonical: dict[object, int] = {}
    merge_map: dict[int, int] = {}

    for node in nodes:
        sig = _node_signature(node)
        if sig not in sig_to_canonical:
            sig_to_canonical[sig] = node.node_id
        merge_map[node.node_id] = sig_to_canonical[sig]

    return merge_map


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoopInfo:
    """Information about a detected loop in the prefix tree.

    Attributes:
        entry_node: The node ID where the loop starts.
        back_edge_from: The node ID that has a back-edge to entry_node.
        pattern_length: Length of the repeating pattern.
        label_pattern: The repeating sequence of labels.
    """
    entry_node: int
    back_edge_from: int
    pattern_length: int
    label_pattern: tuple[tuple[str, str], ...]


def detect_loops(tree: PrefixTree) -> list[LoopInfo]:
    """Detect loops in the prefix tree by finding back-edges in the merge map.

    A loop exists when a child node maps to the same canonical node as
    an ancestor. We also detect loops by checking if suffixes of a path
    repeat.
    """
    merge_map = merge_states(tree)
    loops: list[LoopInfo] = []
    _detect_loops_dfs(tree.root, [], set(), merge_map, loops)
    return loops


def _detect_loops_dfs(
    node: PrefixNode,
    path: list[tuple[tuple[str, str], PrefixNode]],
    seen_canonicals: set[int],
    merge_map: dict[int, int],
    loops: list[LoopInfo],
) -> None:
    """DFS to find back-edges in the merged prefix tree."""
    canonical = merge_map[node.node_id]

    if canonical in seen_canonicals:
        # Found a loop: walk back to find the entry point
        for i, (key, ancestor) in enumerate(path):
            if merge_map[ancestor.node_id] == canonical:
                pattern = tuple(k for k, _ in path[i:])
                loops.append(LoopInfo(
                    entry_node=merge_map[ancestor.node_id],
                    back_edge_from=merge_map[path[-1][1].node_id] if path else canonical,
                    pattern_length=len(pattern),
                    label_pattern=pattern,
                ))
                return
        return

    seen_canonicals_copy = seen_canonicals | {canonical}

    for key, child in sorted(node.children.items()):
        _detect_loops_dfs(
            child,
            path + [(key, node)],
            seen_canonicals_copy,
            merge_map,
            loops,
        )


# ---------------------------------------------------------------------------
# AST construction from prefix tree
# ---------------------------------------------------------------------------

def _build_ast_from_tree(
    node: PrefixNode,
    merge_map: dict[int, int],
    in_progress: set[int] | None = None,
    var_names: dict[int, str] | None = None,
    cache: dict[int, SessionType] | None = None,
    var_counter: list[int] | None = None,
) -> SessionType:
    """Convert a prefix-tree node (with merging) to a session type AST.

    Uses the merge_map to collapse equivalent nodes. Detects cycles
    (via in_progress set) and emits Rec/Var for recursion.
    """
    if in_progress is None:
        in_progress = set()
    if var_names is None:
        var_names = {}
    if cache is None:
        cache = {}
    if var_counter is None:
        var_counter = [0]

    canonical = merge_map[node.node_id]

    # Cycle detection: if canonical is in progress, emit Var
    if canonical in in_progress:
        if canonical not in var_names:
            idx = var_counter[0]
            var_counter[0] += 1
            name = chr(ord("X") + idx % 3)
            if idx >= 3:
                name += str(idx // 3)
            var_names[canonical] = name
        return Var(var_names[canonical])

    # Check cache
    if canonical in cache:
        return cache[canonical]

    # Terminal with no children → End
    if not node.children:
        result = End()
        cache[canonical] = result
        return result

    in_progress.add(canonical)

    # Group children by direction to determine Branch vs Select
    send_children: list[tuple[str, PrefixNode]] = []
    recv_children: list[tuple[str, PrefixNode]] = []
    for (label, direction), child in sorted(node.children.items()):
        if direction == "send":
            send_children.append((label, child))
        else:
            recv_children.append((label, child))

    # Build child ASTs
    def make_choices(
        children: list[tuple[str, PrefixNode]],
    ) -> tuple[tuple[str, SessionType], ...]:
        choices: list[tuple[str, SessionType]] = []
        for label, child in children:
            child_ast = _build_ast_from_tree(
                child, merge_map, in_progress, var_names, cache, var_counter
            )
            choices.append((label, child_ast))
        return tuple(choices)

    if send_children and not recv_children:
        result = Select(make_choices(send_children))
    elif recv_children and not send_children:
        result = Branch(make_choices(recv_children))
    elif recv_children and send_children:
        # Mixed: treat as nested Branch { ... Select { ... } }
        # We create a Branch for receive, then nest Select inside
        # But if there are both, we just use Branch for all since
        # the distinction is at the choice level
        all_choices = make_choices(recv_children) + make_choices(send_children)
        result = Branch(all_choices)
    else:
        result = End()

    in_progress.discard(canonical)

    # Wrap in Rec if this node was referenced recursively
    if canonical in var_names:
        result = Rec(var_names[canonical], result)

    cache[canonical] = result
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inference(inferred: SessionType, traces: list[Trace]) -> bool:
    """Validate that all traces are accepted by the inferred session type.

    Builds a state space from the inferred type and checks that each
    trace corresponds to a valid path from top to bottom.
    """
    ss = build_statespace(inferred)
    for trace in traces:
        if not _trace_accepted(ss, trace):
            return False
    return True


def _trace_accepted(ss: StateSpace, trace: Trace) -> bool:
    """Check if a single trace is accepted by the state space.

    Follows transitions from top; at the end, the current state must
    be bottom or must have reached the end of the trace at a valid state.
    """
    current_states = {ss.top}

    for step in trace.steps:
        next_states: set[int] = set()
        for state in current_states:
            for label, target in ss.enabled(state):
                if label == step.label:
                    next_states.add(target)
        if not next_states:
            return False
        current_states = next_states

    # After consuming all steps, we should be at bottom
    # (or at a state from which bottom is reachable with no further input)
    return bool(current_states & {ss.bottom})


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(
    traces: list[Trace],
    inferred: SessionType,
    tree: PrefixTree,
    merge_map: dict[int, int],
) -> float:
    """Compute a confidence score for the inference.

    Factors:
    - Number of traces (more traces → higher confidence)
    - Coverage of branching points
    - Validation success
    """
    if not traces:
        return 0.0

    # Base confidence from number of traces
    n = len(traces)
    trace_score = min(1.0, n / 10.0)  # saturates at 10 traces

    # Validation score
    try:
        valid = validate_inference(inferred, traces)
        valid_score = 1.0 if valid else 0.0
    except Exception:
        valid_score = 0.0

    # Merge ratio: how much the tree was compressed
    num_nodes = tree.num_nodes
    num_canonical = len(set(merge_map.values()))
    merge_score = 1.0 - (num_canonical / max(num_nodes, 1))
    merge_score = max(0.0, merge_score)

    # Weighted average
    confidence = 0.5 * valid_score + 0.3 * trace_score + 0.2 * merge_score
    return round(confidence, 3)


# ---------------------------------------------------------------------------
# Recursion detection in AST
# ---------------------------------------------------------------------------

def _has_recursion(node: SessionType) -> bool:
    """Check if a session type AST contains Rec nodes."""
    if isinstance(node, Rec):
        return True
    if isinstance(node, Branch):
        return any(_has_recursion(s) for _, s in node.choices)
    if isinstance(node, Select):
        return any(_has_recursion(s) for _, s in node.choices)
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_from_traces(traces: list[Trace]) -> SessionType:
    """Infer a session type from a list of observed traces.

    Args:
        traces: List of observed traces. Must be non-empty.

    Returns:
        A session type AST that accepts all observed traces.

    Raises:
        ValueError: If traces is empty.
    """
    if not traces:
        raise ValueError("Cannot infer from empty trace list")

    # Handle all-empty traces
    if all(len(t) == 0 for t in traces):
        return End()

    tree = build_prefix_tree(traces)
    merge_map = merge_states(tree)
    return _build_ast_from_tree(tree.root, merge_map)


def infer_from_statespace(ss: StateSpace) -> SessionType:
    """Infer (reconstruct) a session type from a state space.

    This is a thin wrapper around reticular.reconstruct, provided for
    API consistency within the inference module.

    Args:
        ss: A state space.

    Returns:
        A session type AST.
    """
    from reticulate.reticular import reconstruct
    return reconstruct(ss)


def analyze_inference(traces: list[Trace]) -> InferenceResult:
    """Full inference analysis with confidence metrics.

    Args:
        traces: List of observed traces. Must be non-empty.

    Returns:
        An InferenceResult with the inferred type and metadata.

    Raises:
        ValueError: If traces is empty.
    """
    if not traces:
        raise ValueError("Cannot analyze empty trace list")

    # Handle all-empty traces
    if all(len(t) == 0 for t in traces):
        inferred = End()
        return InferenceResult(
            inferred=inferred,
            pretty_type=pretty(inferred),
            num_traces=len(traces),
            num_states=1,
            num_merged_states=1,
            has_recursion=False,
            all_traces_valid=True,
            confidence=0.5,
        )

    tree = build_prefix_tree(traces)
    merge_map = merge_states(tree)
    inferred = _build_ast_from_tree(tree.root, merge_map)

    num_canonical = len(set(merge_map.values()))
    all_valid = validate_inference(inferred, traces)
    confidence = _compute_confidence(traces, inferred, tree, merge_map)

    return InferenceResult(
        inferred=inferred,
        pretty_type=pretty(inferred),
        num_traces=len(traces),
        num_states=tree.num_nodes,
        num_merged_states=num_canonical,
        has_recursion=_has_recursion(inferred),
        all_traces_valid=all_valid,
        confidence=confidence,
    )
