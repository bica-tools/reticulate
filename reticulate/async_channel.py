"""Async channels: message buffers as lattice extensions (Step 157b).

Extends the synchronous binary channel model from Step 157a to
**asynchronous channels** with bounded FIFO buffers.  In the async model,
output (selection) is non-blocking: the sender advances and appends the
chosen label to a buffer.  Input (branch) is blocking: the receiver
waits until the buffer is non-empty, then pops the head.

Two-buffer model (bidirectional):
  State: (s_A, s_B, buf_AB, buf_BA)
  - A sends: A at Select, |buf_AB| < K → A advances, label → buf_AB
  - B receives: buf_AB non-empty, B has Branch on head → B advances, pop
  - B sends: B at Select, |buf_BA| < K → B advances, label → buf_BA
  - A receives: buf_BA non-empty, A has Branch on head → A advances, pop

Key results:
  1. Async channel state space is always a bounded lattice (for well-formed types).
  2. Synchronous channel embeds into async channel (empty-buffer slice).
  3. Buffer capacity K controls state-space growth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.channel import build_channel
from reticulate.duality import dual
from reticulate.lattice import check_lattice
from reticulate.morphism import is_order_preserving, is_order_reflecting
from reticulate.parser import SessionType, pretty
from reticulate.product import product_statespace
from reticulate.statespace import StateSpace, build_statespace

if TYPE_CHECKING:
    from reticulate.global_types import GlobalType


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AsyncChannelResult:
    """Result of async channel analysis for a session type S.

    Attributes:
        type_str: Pretty-printed S.
        dual_str: Pretty-printed dual(S).
        capacity: Buffer bound K.
        sender_states: |L(S)|.
        receiver_states: |L(dual(S))|.
        async_channel_states: Reachable (s_A, s_B, buf_AB, buf_BA) states.
        sync_channel_states: |L(S) × L(dual(S))| for comparison.
        is_lattice: True iff async state space is a lattice.
        sync_embeds: True iff sync channel embeds in async.
        max_buffer_occupancy: Max total |buf_AB| + |buf_BA| seen.
        buffer_distribution: total_buffer_length → state_count.
    """

    type_str: str
    dual_str: str
    capacity: int
    sender_states: int
    receiver_states: int
    async_channel_states: int
    sync_channel_states: int
    is_lattice: bool
    sync_embeds: bool
    max_buffer_occupancy: int
    buffer_distribution: dict[int, int]


# ---------------------------------------------------------------------------
# Async state key type
# ---------------------------------------------------------------------------

# State key: (s_A, s_B, buf_AB, buf_BA)
# buf_AB and buf_BA are tuples of label strings (FIFO, head = index 0)
AsyncStateKey = tuple[int, int, tuple[str, ...], tuple[str, ...]]


# ---------------------------------------------------------------------------
# Core: build_async_statespace
# ---------------------------------------------------------------------------

def build_async_statespace(
    ss_s: StateSpace,
    ss_d: StateSpace,
    capacity: int,
) -> StateSpace:
    """Build the async channel state space via BFS.

    Args:
        ss_s: State space L(S) for role A.
        ss_d: State space L(dual(S)) for role B.
        capacity: Maximum buffer size K (per direction).

    Returns:
        StateSpace with states (s_A, s_B, buf_AB, buf_BA), transitions
        labeled "send_AB:m", "recv_AB:m", "send_BA:m", "recv_BA:m".
    """
    # Pre-compute adjacency lists for both state spaces
    a_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in ss_s.states}
    for src, lbl, tgt in ss_s.transitions:
        a_adj[src].append((lbl, tgt))

    b_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in ss_d.states}
    for src, lbl, tgt in ss_d.transitions:
        b_adj[src].append((lbl, tgt))

    # State ID allocation
    key_to_id: dict[AsyncStateKey, int] = {}
    id_labels: dict[int, str] = {}
    transitions: list[tuple[int, str, int]] = []
    selection_transitions: set[tuple[int, str, int]] = set()
    next_id = 0

    def get_id(key: AsyncStateKey) -> int:
        nonlocal next_id
        if key not in key_to_id:
            sid = next_id
            next_id += 1
            key_to_id[key] = sid
            s_a, s_b, buf_ab, buf_ba = key
            la = ss_s.labels.get(s_a, str(s_a))
            lb = ss_d.labels.get(s_b, str(s_b))
            buf_str = ""
            if buf_ab or buf_ba:
                buf_str = f" [{','.join(buf_ab)}|{','.join(buf_ba)}]"
            id_labels[sid] = f"({la}, {lb}{buf_str})"
        return key_to_id[key]

    # BFS from (top_S, top_dual, (), ())
    start: AsyncStateKey = (ss_s.top, ss_d.top, (), ())
    get_id(start)
    queue: deque[AsyncStateKey] = deque([start])
    visited: set[AsyncStateKey] = set()

    while queue:
        key = queue.popleft()
        if key in visited:
            continue
        visited.add(key)

        s_a, s_b, buf_ab, buf_ba = key
        src = key_to_id[key]

        # --- A sends (A has selection transitions, buf_AB not full) ---
        if len(buf_ab) < capacity:
            for lbl, s_a_tgt in a_adj[s_a]:
                if ss_s.is_selection(s_a, lbl, s_a_tgt):
                    new_buf_ab = buf_ab + (lbl,)
                    new_key: AsyncStateKey = (s_a_tgt, s_b, new_buf_ab, buf_ba)
                    tgt = get_id(new_key)
                    tr = (src, f"send_AB:{lbl}", tgt)
                    transitions.append(tr)
                    selection_transitions.add(tr)
                    if new_key not in visited:
                        queue.append(new_key)

        # --- B receives (buf_AB non-empty, B has branch on head label) ---
        if buf_ab:
            head = buf_ab[0]
            rest_ab = buf_ab[1:]
            for lbl, s_b_tgt in b_adj[s_b]:
                if not ss_d.is_selection(s_b, lbl, s_b_tgt) and lbl == head:
                    new_key = (s_a, s_b_tgt, rest_ab, buf_ba)
                    tgt = get_id(new_key)
                    tr = (src, f"recv_AB:{lbl}", tgt)
                    transitions.append(tr)
                    # Receiving is a branch transition, not selection
                    if new_key not in visited:
                        queue.append(new_key)

        # --- B sends (B has selection transitions, buf_BA not full) ---
        if len(buf_ba) < capacity:
            for lbl, s_b_tgt in b_adj[s_b]:
                if ss_d.is_selection(s_b, lbl, s_b_tgt):
                    new_buf_ba = buf_ba + (lbl,)
                    new_key = (s_a, s_b_tgt, buf_ab, new_buf_ba)
                    tgt = get_id(new_key)
                    tr = (src, f"send_BA:{lbl}", tgt)
                    transitions.append(tr)
                    selection_transitions.add(tr)
                    if new_key not in visited:
                        queue.append(new_key)

        # --- A receives (buf_BA non-empty, A has branch on head label) ---
        if buf_ba:
            head = buf_ba[0]
            rest_ba = buf_ba[1:]
            for lbl, s_a_tgt in a_adj[s_a]:
                if not ss_s.is_selection(s_a, lbl, s_a_tgt) and lbl == head:
                    new_key = (s_a_tgt, s_b, buf_ab, rest_ba)
                    tgt = get_id(new_key)
                    tr = (src, f"recv_BA:{lbl}", tgt)
                    transitions.append(tr)
                    if new_key not in visited:
                        queue.append(new_key)

    # Top and bottom
    top = key_to_id[start]
    bottom_key: AsyncStateKey = (ss_s.bottom, ss_d.bottom, (), ())
    if bottom_key in key_to_id:
        bottom = key_to_id[bottom_key]
    else:
        # Bottom might not be reachable
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
# Sync embedding check
# ---------------------------------------------------------------------------

def check_sync_embedding(
    sync_ss: StateSpace,
    async_ss: StateSpace,
    ss_s: StateSpace,
    ss_d: StateSpace,
    async_key_to_id: dict[AsyncStateKey, int],
) -> bool:
    """Verify that empty-buffer async states preserve sync ordering.

    In the async model, not all sync product states are reachable with
    empty buffers (e.g. B cannot advance before A sends).  We check
    that the empty-buffer slice of the async state space preserves the
    ordering of the corresponding sync states: for all sync pairs
    (s_A, s_B) that appear as empty-buffer async states, the order
    relationships are preserved and reflected.

    Args:
        sync_ss: The synchronous product L(S) × L(dual(S)).
        async_ss: The async channel state space.
        ss_s: L(S) state space.
        ss_d: L(dual(S)) state space.
        async_key_to_id: The key→ID mapping from async construction.

    Returns:
        True iff the mappable sync states form an order-embedding.
    """
    # Reconstruct sync pair_to_id (product_statespace uses sorted order)
    left_states = sorted(ss_s.states)
    right_states = sorted(ss_d.states)
    sync_pair_to_id: dict[tuple[int, int], int] = {}
    next_id = 0
    for s1 in left_states:
        for s2 in right_states:
            sync_pair_to_id[(s1, s2)] = next_id
            next_id += 1

    # Build embedding: only include sync states that exist as empty-buffer
    # async states (reachable in async with empty buffers)
    embedding: dict[int, int] = {}
    for (s_a, s_b), sync_id in sync_pair_to_id.items():
        if sync_id not in sync_ss.states:
            continue
        async_key: AsyncStateKey = (s_a, s_b, (), ())
        if async_key in async_key_to_id:
            embedding[sync_id] = async_key_to_id[async_key]
        # Skip sync states not reachable as empty-buffer async states

    if not embedding:
        return True  # Trivial

    # Check order-preserving and order-reflecting using reachability
    # in the FULL sync and async state spaces, but only for mapped states.
    # s1 ≥ s2 in sync iff s2 reachable from s1 in sync_ss.
    sync_reach = {s: sync_ss.reachable_from(s) for s in embedding}
    async_reach = {s: async_ss.reachable_from(s) for s in set(embedding.values())}

    for s1, a1 in embedding.items():
        for s2, a2 in embedding.items():
            # Order-preserving: s2 reachable from s1 in sync → a2 reachable from a1 in async
            if s2 in sync_reach[s1]:
                if a2 not in async_reach[a1]:
                    return False
            # Order-reflecting: a2 reachable from a1 in async → s2 reachable from s1 in sync
            if a2 in async_reach[a1]:
                if s2 not in sync_reach[s1]:
                    return False

    return True


# ---------------------------------------------------------------------------
# Orchestrator: build_async_channel
# ---------------------------------------------------------------------------

def build_async_channel(
    s: SessionType,
    capacity: int = 1,
) -> AsyncChannelResult:
    """Build async channel and verify properties.

    Args:
        s: Session type for role A.
        capacity: Buffer bound K (default 1).

    Returns:
        AsyncChannelResult with lattice check and sync embedding.
    """
    d = dual(s)
    ss_s = build_statespace(s)
    ss_d = build_statespace(d)

    # Sync channel for comparison
    sync_ss = product_statespace(ss_s, ss_d)

    # Async channel
    async_ss = build_async_statespace(ss_s, ss_d, capacity)

    # Get key_to_id for embedding check — rebuild it
    # We need to re-extract the mapping. Rather than re-running BFS,
    # we reconstruct from the labels or run the BFS and return the mapping.
    # For simplicity, run BFS again with mapping extraction.
    key_to_id = _build_key_mapping(ss_s, ss_d, capacity)

    # Lattice check
    lattice_result = check_lattice(async_ss)

    # Sync embedding check
    sync_emb = check_sync_embedding(sync_ss, async_ss, ss_s, ss_d, key_to_id)

    # Buffer distribution
    buf_dist: dict[int, int] = {}
    max_occ = 0
    for key in key_to_id:
        total_buf = len(key[2]) + len(key[3])
        buf_dist[total_buf] = buf_dist.get(total_buf, 0) + 1
        if total_buf > max_occ:
            max_occ = total_buf

    return AsyncChannelResult(
        type_str=pretty(s),
        dual_str=pretty(d),
        capacity=capacity,
        sender_states=len(ss_s.states),
        receiver_states=len(ss_d.states),
        async_channel_states=len(async_ss.states),
        sync_channel_states=len(sync_ss.states),
        is_lattice=lattice_result.is_lattice,
        sync_embeds=sync_emb,
        max_buffer_occupancy=max_occ,
        buffer_distribution=buf_dist,
    )


def _build_key_mapping(
    ss_s: StateSpace,
    ss_d: StateSpace,
    capacity: int,
) -> dict[AsyncStateKey, int]:
    """Re-run BFS to extract the key→ID mapping (matches build_async_statespace)."""
    a_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in ss_s.states}
    for src, lbl, tgt in ss_s.transitions:
        a_adj[src].append((lbl, tgt))

    b_adj: dict[int, list[tuple[str, int]]] = {s: [] for s in ss_d.states}
    for src, lbl, tgt in ss_d.transitions:
        b_adj[src].append((lbl, tgt))

    key_to_id: dict[AsyncStateKey, int] = {}
    next_id = 0

    def get_id(key: AsyncStateKey) -> int:
        nonlocal next_id
        if key not in key_to_id:
            key_to_id[key] = next_id
            next_id += 1
        return key_to_id[key]

    start: AsyncStateKey = (ss_s.top, ss_d.top, (), ())
    get_id(start)
    queue: deque[AsyncStateKey] = deque([start])
    visited: set[AsyncStateKey] = set()

    while queue:
        key = queue.popleft()
        if key in visited:
            continue
        visited.add(key)

        s_a, s_b, buf_ab, buf_ba = key

        if len(buf_ab) < capacity:
            for lbl, s_a_tgt in a_adj[s_a]:
                if ss_s.is_selection(s_a, lbl, s_a_tgt):
                    new_key: AsyncStateKey = (s_a_tgt, s_b, buf_ab + (lbl,), buf_ba)
                    get_id(new_key)
                    if new_key not in visited:
                        queue.append(new_key)

        if buf_ab:
            head = buf_ab[0]
            rest_ab = buf_ab[1:]
            for lbl, s_b_tgt in b_adj[s_b]:
                if not ss_d.is_selection(s_b, lbl, s_b_tgt) and lbl == head:
                    new_key = (s_a, s_b_tgt, rest_ab, buf_ba)
                    get_id(new_key)
                    if new_key not in visited:
                        queue.append(new_key)

        if len(buf_ba) < capacity:
            for lbl, s_b_tgt in b_adj[s_b]:
                if ss_d.is_selection(s_b, lbl, s_b_tgt):
                    new_key = (s_a, s_b_tgt, buf_ab, buf_ba + (lbl,))
                    get_id(new_key)
                    if new_key not in visited:
                        queue.append(new_key)

        if buf_ba:
            head = buf_ba[0]
            rest_ba = buf_ba[1:]
            for lbl, s_a_tgt in a_adj[s_a]:
                if not ss_s.is_selection(s_a, lbl, s_a_tgt) and lbl == head:
                    new_key = (s_a_tgt, s_b, buf_ab, rest_ba)
                    get_id(new_key)
                    if new_key not in visited:
                        queue.append(new_key)

    # Ensure bottom is present
    bottom_key: AsyncStateKey = (ss_s.bottom, ss_d.bottom, (), ())
    get_id(bottom_key)

    return key_to_id


# ---------------------------------------------------------------------------
# Growth ratio
# ---------------------------------------------------------------------------

def async_growth_ratio(s: SessionType, capacity: int) -> float:
    """Compute the ratio async_states / sync_states.

    Returns:
        The growth ratio (≥ 1.0).
    """
    r = build_async_channel(s, capacity)
    if r.sync_channel_states == 0:
        return 1.0
    return r.async_channel_states / r.sync_channel_states


# ---------------------------------------------------------------------------
# Global type bridge
# ---------------------------------------------------------------------------

def async_channel_from_global(
    g: GlobalType,
    sender: str,
    receiver: str,
    capacity: int = 1,
) -> AsyncChannelResult:
    """Build an async channel from a binary global type via projection.

    Args:
        g: A global type (should be binary between sender and receiver).
        sender: The sender role name.
        receiver: The receiver role name.
        capacity: Buffer bound K.

    Returns:
        AsyncChannelResult for the projected types.

    Raises:
        ValueError: If projection fails.
    """
    from reticulate.projection import project

    local_a = project(g, sender)
    return build_async_channel(local_a, capacity)
