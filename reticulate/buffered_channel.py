"""Buffered channel session types (Step 158).

Models single-endpoint buffered channels where messages are enqueued into
a FIFO buffer of bounded capacity before being consumed.  Unlike the
bidirectional async channel model (Step 157b), a buffered channel models
a *single* session type with an explicit buffer that decouples the
producer (selection/output) from the consumer (branch/input).

State = (protocol_state, buffer_contents)
  - At a Select node: the sender enqueues the chosen label (if buffer not full).
  - At a Branch node: the consumer dequeues the head label (if buffer non-empty).
  - Buffer overflow: attempting to enqueue when |buffer| == capacity.
  - Buffer underflow: attempting to dequeue from an empty buffer.

Key analyses:
  1. Buffer safety: no reachable state has overflow or underflow.
  2. Growth analysis: how state-space size grows with buffer capacity.
  3. Optimal buffer size: minimum capacity for deadlock freedom.
  4. Lattice preservation: buffered state space remains a lattice.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

from reticulate.lattice import LatticeResult, check_lattice
from reticulate.parser import SessionType, pretty
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BufferedChannelResult:
    """Result of buffered channel analysis for a session type.

    Attributes:
        type_str: Pretty-printed session type.
        capacity: Buffer bound K.
        unbuffered_states: Number of states in L(S) (capacity=0 / synchronous).
        buffered_states: Number of states in the buffered state space.
        buffered_transitions: Number of transitions in the buffered state space.
        is_lattice: True iff buffered state space is a lattice.
        is_buffer_safe: True iff no overflow or underflow in reachable states.
        overflow_states: Number of states where overflow would occur.
        underflow_states: Number of states where underflow would occur.
        max_buffer_occupancy: Maximum buffer fill level observed.
        buffer_distribution: buffer_length -> count of states with that fill.
        deadlock_states: Number of states with no outgoing transitions
            that are not the bottom state.
        lattice_result: Full LatticeResult for detailed inspection.
    """

    type_str: str
    capacity: int
    unbuffered_states: int
    buffered_states: int
    buffered_transitions: int
    is_lattice: bool
    is_buffer_safe: bool
    overflow_states: int
    underflow_states: int
    max_buffer_occupancy: int
    buffer_distribution: dict[int, int]
    deadlock_states: int
    lattice_result: LatticeResult


class GrowthPoint(NamedTuple):
    """A single data point in buffer growth analysis."""

    capacity: int
    states: int
    transitions: int
    is_lattice: bool
    deadlocks: int


@dataclass(frozen=True)
class GrowthAnalysis:
    """How the buffered state space grows with capacity.

    Attributes:
        type_str: Pretty-printed session type.
        max_capacity: Maximum capacity tested.
        points: List of GrowthPoint, one per capacity 0..max_capacity.
        growth_ratios: Ratio states[k] / states[k-1] for k >= 1.
        optimal_capacity: Minimum capacity for zero deadlocks (or -1).
    """

    type_str: str
    max_capacity: int
    points: list[GrowthPoint]
    growth_ratios: list[float]
    optimal_capacity: int


# ---------------------------------------------------------------------------
# Buffered state key
# ---------------------------------------------------------------------------

# (protocol_state_id, buffer_contents_as_tuple)
BufferedStateKey = tuple[int, tuple[str, ...]]


# ---------------------------------------------------------------------------
# Core: build_buffered_statespace
# ---------------------------------------------------------------------------

def build_buffered_statespace(
    ast: SessionType,
    capacity: int,
) -> StateSpace:
    """Build a buffered channel state space via BFS.

    The base state space L(S) provides the protocol skeleton.  Each state
    in the buffered state space is a pair (protocol_state, buffer_contents).

    Transitions:
      - At a selection state: for each label l, if |buffer| < capacity,
        enqueue l and advance the protocol.  Label: "enqueue:l".
      - At a branch state: if buffer is non-empty and head matches a
        branch label, dequeue and advance.  Label: "dequeue:l".
      - At a non-selection, non-branch state (e.g., end): no transitions
        (terminal or handled by other means).

    Args:
        ast: Session type AST.
        capacity: Maximum buffer size K (>= 0).  K=0 means synchronous
            (no buffering; selections and branches must be matched directly).

    Returns:
        StateSpace with buffered states.
    """
    base_ss = build_statespace(ast)
    return _build_buffered_from_ss(base_ss, capacity)


def _build_buffered_from_ss(
    base_ss: StateSpace,
    capacity: int,
) -> StateSpace:
    """Build buffered state space from an existing base state space."""
    # Pre-compute adjacency + classify states
    adj: dict[int, list[tuple[str, int]]] = {s: [] for s in base_ss.states}
    for src, lbl, tgt in base_ss.transitions:
        adj[src].append((lbl, tgt))

    # State ID allocation
    key_to_id: dict[BufferedStateKey, int] = {}
    id_labels: dict[int, str] = {}
    transitions: list[tuple[int, str, int]] = []
    selection_transitions: set[tuple[int, str, int]] = set()
    next_id = 0

    def get_id(key: BufferedStateKey) -> int:
        nonlocal next_id
        if key not in key_to_id:
            sid = next_id
            next_id += 1
            key_to_id[key] = sid
            proto_state, buf = key
            base_label = base_ss.labels.get(proto_state, str(proto_state))
            if buf:
                id_labels[sid] = f"({base_label}, [{','.join(buf)}])"
            else:
                id_labels[sid] = f"({base_label}, [])"
        return key_to_id[key]

    # BFS from (top, ())
    start: BufferedStateKey = (base_ss.top, ())
    get_id(start)
    queue: deque[BufferedStateKey] = deque([start])
    visited: set[BufferedStateKey] = set()

    while queue:
        key = queue.popleft()
        if key in visited:
            continue
        visited.add(key)

        proto_state, buf = key
        src = key_to_id[key]

        for lbl, tgt in adj[proto_state]:
            is_sel = base_ss.is_selection(proto_state, lbl, tgt)

            if is_sel:
                # Selection = enqueue into buffer
                if len(buf) < capacity:
                    new_buf = buf + (lbl,)
                    new_key: BufferedStateKey = (tgt, new_buf)
                    tgt_id = get_id(new_key)
                    tr = (src, f"enqueue:{lbl}", tgt_id)
                    transitions.append(tr)
                    selection_transitions.add(tr)
                    if new_key not in visited:
                        queue.append(new_key)
                elif capacity == 0:
                    # Synchronous mode: selection transitions pass through
                    # directly (no buffer)
                    new_key = (tgt, buf)
                    tgt_id = get_id(new_key)
                    tr = (src, lbl, tgt_id)
                    transitions.append(tr)
                    selection_transitions.add(tr)
                    if new_key not in visited:
                        queue.append(new_key)
            else:
                # Branch = dequeue from buffer
                if buf and buf[0] == lbl:
                    # Head of buffer matches branch label
                    new_buf = buf[1:]
                    new_key = (tgt, new_buf)
                    tgt_id = get_id(new_key)
                    tr = (src, f"dequeue:{lbl}", tgt_id)
                    transitions.append(tr)
                    if new_key not in visited:
                        queue.append(new_key)
                elif not buf:
                    if capacity == 0:
                        # Synchronous: branch transitions pass through directly
                        new_key = (tgt, buf)
                        tgt_id = get_id(new_key)
                        tr = (src, lbl, tgt_id)
                        transitions.append(tr)
                        if new_key not in visited:
                            queue.append(new_key)
                    # else: underflow — cannot dequeue from empty buffer
                    # (blocked, not an error — just no transition)

    # Top and bottom
    top = key_to_id[start]
    bottom_key: BufferedStateKey = (base_ss.bottom, ())
    if bottom_key in key_to_id:
        bottom = key_to_id[bottom_key]
    else:
        bottom = get_id(bottom_key)

    return StateSpace(
        states=set(key_to_id.values()),
        transitions=transitions,
        top=top,
        bottom=bottom,
        labels=id_labels,
        selection_transitions=selection_transitions,
    )


# ---------------------------------------------------------------------------
# Buffer safety check
# ---------------------------------------------------------------------------

def check_buffer_safety(
    ast: SessionType,
    capacity: int,
) -> BufferedChannelResult:
    """Check whether a buffered channel has overflow/underflow issues.

    Overflow: a selection state where |buffer| == capacity (cannot enqueue).
    Underflow: a branch state where buffer is empty (cannot dequeue).

    For capacity == 0 (synchronous), there are no overflow/underflow issues
    by definition since transitions pass through directly.

    Args:
        ast: Session type AST.
        capacity: Buffer bound K.

    Returns:
        BufferedChannelResult with safety analysis.
    """
    base_ss = build_statespace(ast)
    buffered_ss = _build_buffered_from_ss(base_ss, capacity)
    lattice_result = check_lattice(buffered_ss)

    # Classify base states
    base_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in base_ss.states}
    for src, lbl, tgt in base_ss.transitions:
        base_adj[src].append((lbl, tgt))

    # Re-extract key mapping
    key_to_id = _extract_key_mapping(base_ss, capacity)

    overflow_count = 0
    underflow_count = 0
    buf_dist: dict[int, int] = {}
    max_occ = 0

    for key, sid in key_to_id.items():
        proto_state, buf = key
        buf_len = len(buf)
        buf_dist[buf_len] = buf_dist.get(buf_len, 0) + 1
        if buf_len > max_occ:
            max_occ = buf_len

        # Check overflow: selection state with full buffer
        if capacity > 0:
            has_selection = any(
                base_ss.is_selection(proto_state, lbl, tgt)
                for lbl, tgt in base_adj.get(proto_state, [])
            )
            if has_selection and buf_len >= capacity:
                overflow_count += 1

        # Check underflow: branch state with empty buffer
        if capacity > 0:
            has_branch = any(
                not base_ss.is_selection(proto_state, lbl, tgt)
                for lbl, tgt in base_adj.get(proto_state, [])
            )
            if has_branch and buf_len == 0:
                underflow_count += 1

    # Deadlock detection
    out_degree: dict[int, int] = {s: 0 for s in buffered_ss.states}
    for src, _, _ in buffered_ss.transitions:
        out_degree[src] += 1
    deadlocks = sum(
        1 for s, deg in out_degree.items()
        if deg == 0 and s != buffered_ss.bottom
    )

    is_safe = overflow_count == 0 and underflow_count == 0

    return BufferedChannelResult(
        type_str=pretty(ast),
        capacity=capacity,
        unbuffered_states=len(base_ss.states),
        buffered_states=len(buffered_ss.states),
        buffered_transitions=len(buffered_ss.transitions),
        is_lattice=lattice_result.is_lattice,
        is_buffer_safe=is_safe,
        overflow_states=overflow_count,
        underflow_states=underflow_count,
        max_buffer_occupancy=max_occ,
        buffer_distribution=buf_dist,
        deadlock_states=deadlocks,
        lattice_result=lattice_result,
    )


def _extract_key_mapping(
    base_ss: StateSpace,
    capacity: int,
) -> dict[BufferedStateKey, int]:
    """Re-run BFS to extract key->ID mapping (mirrors _build_buffered_from_ss)."""
    adj: dict[int, list[tuple[str, int]]] = {s: [] for s in base_ss.states}
    for src, lbl, tgt in base_ss.transitions:
        adj[src].append((lbl, tgt))

    key_to_id: dict[BufferedStateKey, int] = {}
    next_id = 0

    def get_id(key: BufferedStateKey) -> int:
        nonlocal next_id
        if key not in key_to_id:
            key_to_id[key] = next_id
            next_id += 1
        return key_to_id[key]

    start: BufferedStateKey = (base_ss.top, ())
    get_id(start)
    queue: deque[BufferedStateKey] = deque([start])
    visited: set[BufferedStateKey] = set()

    while queue:
        key = queue.popleft()
        if key in visited:
            continue
        visited.add(key)

        proto_state, buf = key

        for lbl, tgt in adj[proto_state]:
            is_sel = base_ss.is_selection(proto_state, lbl, tgt)

            if is_sel:
                if len(buf) < capacity:
                    new_key: BufferedStateKey = (tgt, buf + (lbl,))
                    get_id(new_key)
                    if new_key not in visited:
                        queue.append(new_key)
                elif capacity == 0:
                    new_key = (tgt, buf)
                    get_id(new_key)
                    if new_key not in visited:
                        queue.append(new_key)
            else:
                if buf and buf[0] == lbl:
                    new_key = (tgt, buf[1:])
                    get_id(new_key)
                    if new_key not in visited:
                        queue.append(new_key)
                elif not buf and capacity == 0:
                    new_key = (tgt, buf)
                    get_id(new_key)
                    if new_key not in visited:
                        queue.append(new_key)

    bottom_key: BufferedStateKey = (base_ss.bottom, ())
    get_id(bottom_key)

    return key_to_id


# ---------------------------------------------------------------------------
# Growth analysis
# ---------------------------------------------------------------------------

def buffer_growth_analysis(
    ast: SessionType,
    max_capacity: int = 10,
) -> GrowthAnalysis:
    """Analyze how the buffered state space grows with buffer capacity.

    Args:
        ast: Session type AST.
        max_capacity: Maximum capacity to test (inclusive).

    Returns:
        GrowthAnalysis with per-capacity data points and growth ratios.
    """
    points: list[GrowthPoint] = []

    for k in range(max_capacity + 1):
        result = check_buffer_safety(ast, k)
        points.append(GrowthPoint(
            capacity=k,
            states=result.buffered_states,
            transitions=result.buffered_transitions,
            is_lattice=result.is_lattice,
            deadlocks=result.deadlock_states,
        ))

    # Compute growth ratios
    ratios: list[float] = []
    for i in range(1, len(points)):
        prev = points[i - 1].states
        curr = points[i].states
        if prev > 0:
            ratios.append(curr / prev)
        else:
            ratios.append(float('inf'))

    # Find optimal capacity (first k with zero deadlocks)
    optimal = -1
    for p in points:
        if p.deadlocks == 0:
            optimal = p.capacity
            break

    return GrowthAnalysis(
        type_str=pretty(ast),
        max_capacity=max_capacity,
        points=points,
        growth_ratios=ratios,
        optimal_capacity=optimal,
    )


# ---------------------------------------------------------------------------
# Optimal buffer size
# ---------------------------------------------------------------------------

def optimal_buffer_size(
    ast: SessionType,
    max_search: int = 20,
) -> int:
    """Find the minimum buffer capacity for deadlock freedom.

    Searches capacities 0..max_search for the first capacity where the
    buffered state space has no deadlock states (states with no outgoing
    transitions other than the bottom state).

    Args:
        ast: Session type AST.
        max_search: Maximum capacity to search.

    Returns:
        Minimum capacity for deadlock freedom, or -1 if none found
        within the search range.
    """
    for k in range(max_search + 1):
        result = check_buffer_safety(ast, k)
        if result.deadlock_states == 0:
            return k
    return -1
